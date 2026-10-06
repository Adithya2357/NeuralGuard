"""Training pipeline: build a dataset, fit the model, evaluate it honestly, save it.

Two data sources are supported:

* **Simulated traffic** (the default): the model is trained on several independent
  :class:`~neuralguard.simulator.TrafficSimulator` streams - each its own network, hosts
  and attack episodes - and evaluated on *separate* held-out streams with seeds of their
  own (see :func:`simulated_plan`), not on a random split of the same traffic. The first
  ``EPISODE_WARMUP_PACKETS`` packets of every attack episode are left out of training:
  they carry no evidence of the attack yet (a flood's first SYN looks exactly like a new
  visitor's), and labelled as attacks they taught the model that ordinary first contacts
  are attacks. Besides per-packet metrics, the evaluation reports their spread over the
  held-out streams, how many attack episodes were detected, and the false alerts per
  hour on attack-free traffic through the detector's own alert throttling.
* **Labelled JSON Lines** (``data_path``): one traffic record per line, each with a
  ``label``. Records are sorted by time, features are computed over the whole stream
  (exactly as the live detector would see it) and the data is split *chronologically*:
  the first ``1 - test_size`` for training, the rest for testing. A random split would
  leak, because neighbouring packets of the same episode share almost identical window
  features.

Features always come from the same :class:`~neuralguard.features.FeatureExtractor` the
detector uses, so there is no train/serve skew.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neuralguard.config import Settings
from neuralguard.features import FEATURE_NAMES, FeatureExtractor
from neuralguard.model import ThreatModel
from neuralguard.schema import (
    ATTACK_TYPES,
    LABELS,
    NORMAL_LABEL,
    InvalidRecordError,
    normalize_record,
)
from neuralguard.simulator import TrafficSimulator

logger = logging.getLogger(__name__)

DEFAULT_START_TIME = 1_700_000_000.0
DEFAULT_SAMPLES = 240_000
# Simulated training data comes in streams of at most this many records (about four
# minutes of traffic), each with its own seed: more independent networks and attack
# episodes - every variant seen several times - rather than one ever longer stream.
STREAM_RECORDS = 60_000
MAX_TEST_STREAMS = 4
EPISODE_WARMUP_PACKETS = 10
# Attack-free simulated records the false alerts per hour are measured on: twice as many
# as the test set has, within these bounds (120k is about ten minutes of traffic).
ATTACK_FREE_RECORDS = (10_000, 120_000)
METRIC_SUMMARY_KEYS = (
    "accuracy",
    "roc_auc",
    "detection_rate",
    "false_positive_rate",
    "episode_detection_rate",
    "false_alerts_per_hour",
)
TOP_FEATURES = 10
_MIN_SIMULATED_TEST_SAMPLES = 1_000
_SEED_STRIDE = 100  # training stream i uses seed + 100 * i, test stream j seed + 1 + j

SIMULATED_CAVEAT = (
    "Note: these metrics were measured on synthetic traffic and are optimistic; retrain on "
    "real labelled captures (--data FILE.jsonl) before relying on the model in production."
)


@dataclass
class TrainingResult:
    """A trained model plus how it was trained and how well it did on held-out data.

    ``class_counts`` counts the *training* labels; ``simulated`` is False when the model
    was trained on a labelled JSONL file.
    """

    model: ThreatModel
    metrics: dict[str, Any]
    train_size: int
    test_size: int
    class_counts: dict[str, int] = field(default_factory=dict)
    simulated: bool = True


def build_dataset(
    records: Iterable[Mapping[str, Any]], window_seconds: float
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix and labels for time-ordered labelled records.

    Every record goes through ``normalize_record`` and one fresh ``FeatureExtractor``
    (window ``window_seconds``). Raises ``InvalidRecordError`` for a malformed record and
    ``ValueError`` for a record without a label; both name the record's index.
    """
    extractor = FeatureExtractor(window_seconds=window_seconds)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, raw in enumerate(records):
        try:
            record = normalize_record(raw)
        except InvalidRecordError as exc:
            raise InvalidRecordError(f"record {index}: {exc}") from exc
        label = record.get("label")
        if label is None:
            raise ValueError(f"record {index} has no label; training needs labelled records")
        rows.append(extractor.transform_one(record))
        labels.append(label)
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), labels
    return np.vstack(rows), labels


def simulated_plan(
    n_samples: int, seed: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """``(seed, records)`` of each simulated training stream and held-out test stream.

    ``n_samples`` training records are split evenly into streams of at most
    ``STREAM_RECORDS``, with seeds ``seed``, ``seed + 100``, ... Each of the first
    ``MAX_TEST_STREAMS`` of them gets a held-out test stream a quarter as long (at least
    1000 records) with seed ``seed + 1``, ``seed + 2``, ...: never a training seed.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    streams = -(-n_samples // STREAM_RECORDS)  # rounded up
    size, extra = divmod(n_samples, streams)
    training = [(seed + _SEED_STRIDE * i, size + (i < extra)) for i in range(streams)]
    test_records = max(size // 4, min(size, _MIN_SIMULATED_TEST_SAMPLES))
    test = [(seed + 1 + i, test_records) for i in range(min(streams, MAX_TEST_STREAMS))]
    return training, test


def simulated_dataset(
    n_samples: int,
    *,
    seed: int,
    attack_ratio: float,
    window_seconds: float,
    start_time: float = DEFAULT_START_TIME,
    skip_episode_starts: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """``build_dataset`` over ``n_samples`` records of a seeded traffic simulation.

    ``skip_episode_starts`` leaves the first that many packets of every attack episode
    out of the result (they still count in the windows of the packets after them).
    """
    stream = _simulated_stream(
        n_samples,
        seed=seed,
        attack_ratio=attack_ratio,
        window_seconds=window_seconds,
        start_time=start_time,
    )
    if not skip_episode_starts:
        return stream.X, stream.labels
    keep = (stream.positions < 0) | (stream.positions >= skip_episode_starts)
    labels = [label for label, kept in zip(stream.labels, keep, strict=True) if kept]
    return stream.X[keep], labels


@dataclass(frozen=True)
class _SimulatedStream:
    seed: int
    X: np.ndarray
    labels: list[str]
    episodes: np.ndarray  # attack episode of each record, 0 for normal traffic
    positions: np.ndarray  # each record's place in its episode, -1 for normal traffic


def _simulated_stream(
    n_samples: int,
    *,
    seed: int,
    attack_ratio: float,
    window_seconds: float,
    start_time: float = DEFAULT_START_TIME,
) -> _SimulatedStream:
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    simulator = TrafficSimulator(seed=seed, attack_ratio=attack_ratio, start_time=start_time)
    items = list(simulator.records_with_episodes(n_samples))
    X, labels = build_dataset((record for record, _, _ in items), window_seconds)
    return _SimulatedStream(
        seed=seed,
        X=X,
        labels=labels,
        episodes=np.array([episode for _, episode, _ in items], dtype=np.int64),
        positions=np.array([position for _, _, position in items], dtype=np.int64),
    )


def load_jsonl(path: str | Path, *, require_label: bool = True) -> list[dict[str, Any]]:
    """Read labelled traffic records from a JSON Lines file, sorted by timestamp.

    Blank lines are ignored. Each record is validated with ``normalize_record`` and must
    have a ``timestamp`` (window features need real times) and, unless
    ``require_label=False``, a ``label``. Any problem raises ``ValueError`` naming the
    line number. Records with equal timestamps keep their file order.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.strip()
                if text:
                    records.append(_parse_line(text, path, lineno, require_label))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror or exc}") from exc
    records.sort(key=lambda record: record["timestamp"])
    logger.info("loaded %d records from %s", len(records), path)
    return records


def train_model(
    *,
    n_samples: int = DEFAULT_SAMPLES,
    seed: int = 42,
    attack_ratio: float = 0.3,
    window_seconds: float = 10.0,
    n_estimators: int = 200,
    data_path: str | Path | None = None,
    test_size: float = 0.25,
) -> TrainingResult:
    """Build the dataset, fit a :class:`ThreatModel` and evaluate it on held-out data.

    Without ``data_path`` the model trains on ``n_samples`` simulated records and is
    tested on separate simulated streams (see :func:`simulated_plan`); the metrics then
    also cover the spread over those streams, attack episodes and false alerts per hour
    (see the module docstring). With ``data_path`` the labelled JSONL file is split
    chronologically by ``test_size``; ``n_samples`` and ``attack_ratio`` are then unused.
    Training parameters and a metrics summary are stored in the model metadata.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}")

    params: dict[str, Any] = {
        "window_seconds": float(window_seconds),
        "n_estimators": int(n_estimators),
        "seed": seed,
    }
    test_streams: list[_SimulatedStream] = []
    if data_path is None:
        training, test = simulated_plan(n_samples, seed)
        X_train, y_train, test_streams = _simulated_split(
            training, test, attack_ratio=attack_ratio, window_seconds=window_seconds
        )
        X_test = np.vstack([stream.X for stream in test_streams])
        y_test = [label for stream in test_streams for label in stream.labels]
        params.update(
            source="simulated",
            n_samples=int(n_samples),
            attack_ratio=float(attack_ratio),
            seeds=[stream_seed for stream_seed, _ in training],
            test_seeds=[stream_seed for stream_seed, _ in test],
            episode_warmup_packets=EPISODE_WARMUP_PACKETS,
        )
    else:
        X_train, y_train, X_test, y_test = _file_split(data_path, window_seconds, test_size)
        params.update(source="jsonl", data_path=str(data_path), test_size=float(test_size))
    params.update(train_records=len(y_train), test_records=len(y_test))

    logger.info(
        "training on %d records, evaluating on %d held-out records", len(y_train), len(y_test)
    )
    model = ThreatModel.train(
        X_train,
        y_train,
        window_seconds=window_seconds,
        n_estimators=n_estimators,
        random_state=seed,
        metadata={"training": params},
    )
    metrics = model.evaluate(X_test, y_test)
    if test_streams:
        low, high = ATTACK_FREE_RECORDS
        metrics.update(
            _simulated_metrics(
                model,
                test_streams,
                threshold=metrics["threshold"],
                attack_free_records=min(high, max(low, 2 * len(y_test))),
            )
        )
    model.metadata["metrics"] = {key: metrics.get(key) for key in METRIC_SUMMARY_KEYS}
    logger.info(
        "held-out accuracy %s, detection rate %s, false-positive rate %s, attack episodes "
        "detected %s, false alerts per hour %s",
        _number(metrics.get("accuracy")),
        _percent(metrics.get("detection_rate")),
        _percent(metrics.get("false_positive_rate")),
        _percent(metrics.get("episode_detection_rate")),
        _number(metrics.get("false_alerts_per_hour"), ".1f"),
    )
    return TrainingResult(
        model=model,
        metrics=metrics,
        train_size=len(y_train),
        test_size=len(y_test),
        class_counts=_class_counts(y_train),
        simulated=data_path is None,
    )


def format_report(result: TrainingResult) -> str:
    """A human-readable, multi-line summary of a training run."""
    metrics = result.metrics
    lines = _report_header(result)
    lines += ["", "Class balance (training set):", *_class_balance(result.class_counts)]
    threshold = metrics.get("threshold", 0.5)
    lines += [
        "",
        f"Held-out metrics (threat threshold {_number(threshold, '.2f')}):",
        f"  Accuracy               {_number(metrics.get('accuracy'))}",
        f"  ROC-AUC (threat)       {_number(metrics.get('roc_auc'))}",
        f"  Detection rate         {_percent(metrics.get('detection_rate'))}"
        + _spread(metrics, "detection_rate"),
        f"  False-positive rate    {_percent(metrics.get('false_positive_rate'))}"
        + _spread(metrics, "false_positive_rate"),
        f"  Attack-type accuracy   {_percent(metrics.get('attack_type_accuracy'))}",
        *_alerting_lines(metrics),
    ]
    per_class = metrics.get("per_class") or {}
    if per_class:
        lines += [
            "",
            "Per-class results:",
            *_per_class_table(per_class, metrics.get("episodes") or {}),
        ]
    labels = metrics.get("labels") or []
    matrix = metrics.get("confusion_matrix") or []
    if labels and matrix:
        lines += [
            "",
            "Confusion matrix (rows = true label, columns = predicted label):",
            *_confusion_table(labels, matrix),
        ]
    importances = result.model.feature_importances()[:TOP_FEATURES]
    if importances:
        lines += ["", f"Top {len(importances)} feature importances:"]
        width = max(len(name) for name, _ in importances)
        lines += [
            f"  {rank:>2}. {name:<{width}}  {value:.4f}"
            for rank, (name, value) in enumerate(importances, start=1)
        ]
    if result.simulated:
        lines += ["", SIMULATED_CAVEAT]
    return "\n".join(lines)


def _parse_line(text: str, path: Path, lineno: int, require_label: bool) -> dict[str, Any]:
    where = f"{path}, line {lineno}"
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: invalid JSON ({exc.msg})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(raw).__name__}")
    if raw.get("timestamp") is None:
        raise ValueError(f"{where}: record has no timestamp")
    if require_label and raw.get("label") is None:
        raise ValueError(f"{where}: record has no label")
    try:
        return normalize_record(raw)
    except InvalidRecordError as exc:
        raise InvalidRecordError(f"{where}: {exc}") from exc


def _simulated_split(
    training: Sequence[tuple[int, int]],
    test: Sequence[tuple[int, int]],
    *,
    attack_ratio: float,
    window_seconds: float,
) -> tuple[np.ndarray, list[str], list[_SimulatedStream]]:
    parts = []
    for seed, records in training:
        logger.info("simulating %d training records (seed %d)", records, seed)
        parts.append(
            simulated_dataset(
                records,
                seed=seed,
                attack_ratio=attack_ratio,
                window_seconds=window_seconds,
                skip_episode_starts=EPISODE_WARMUP_PACKETS,
            )
        )
    X_train = np.vstack([X for X, _ in parts])
    y_train = [label for _, labels in parts for label in labels]
    streams = []
    for seed, records in test:
        logger.info("simulating %d held-out test records (seed %d)", records, seed)
        streams.append(
            _simulated_stream(
                records, seed=seed, attack_ratio=attack_ratio, window_seconds=window_seconds
            )
        )
    return X_train, y_train, streams


def _simulated_metrics(
    model: ThreatModel,
    streams: Sequence[_SimulatedStream],
    *,
    threshold: float,
    attack_free_records: int,
) -> dict[str, Any]:
    """Per-stream spread, attack-episode detection and false alerts per hour."""
    per_stream = []
    # Per attack type, for each episode: packets before its first detection (None: none).
    firsts: dict[str, list[int | None]] = {}
    for stream in streams:
        flagged = model.threat_scores(stream.X) >= threshold
        attack = stream.episodes > 0
        per_stream.append(
            {
                "seed": stream.seed,
                "records": len(stream.labels),
                "detection_rate": _rate(flagged[attack]),
                "false_positive_rate": _rate(flagged[~attack]),
            }
        )
        for episode in np.unique(stream.episodes[attack]):
            members = np.flatnonzero(stream.episodes == episode)  # in time order
            hits = np.flatnonzero(flagged[members])
            label = stream.labels[members[0]]
            firsts.setdefault(label, []).append(int(hits[0]) if len(hits) else None)
    detected = [first for values in firsts.values() for first in values if first is not None]
    total = sum(len(values) for values in firsts.values())
    return {
        "test_streams": per_stream,
        "episodes": {
            label: {
                "episodes": len(firsts[label]),
                "detected": sum(first is not None for first in firsts[label]),
            }
            for label in ATTACK_TYPES
            if label in firsts
        },
        "episode_detection_rate": len(detected) / total if total else None,
        "median_packets_before_detection": float(np.median(detected)) if detected else None,
        **_false_alerts(
            model, threshold=threshold, seed=streams[0].seed, records=attack_free_records
        ),
    }


def _false_alerts(
    model: ThreatModel, *, threshold: float, seed: int, records: int
) -> dict[str, Any]:
    """Alerts raised on ``records`` records of attack-free simulated traffic (``seed``,
    attack ratio 0) by the detector with its default alert throttling: false alarms an
    analyst would see."""
    from neuralguard.alerts import AlertThrottler
    from neuralguard.consumer import DetectionService
    from neuralguard.detector import Detector

    detector = Detector(model, threshold=threshold)
    throttler = AlertThrottler.from_settings(Settings())
    service = DetectionService(detector, [], throttler=throttler, model_version=model.version)
    simulator = TrafficSimulator(seed=seed, attack_ratio=0.0, start_time=DEFAULT_START_TIME)
    traffic = list(simulator.records(records))
    for start in range(0, len(traffic), 500):  # batches, like the detection service's polls
        service.emit_alerts(detector.process_many(traffic[start : start + 500]))
    service.flush_alerts()
    minutes = (traffic[-1]["timestamp"] - traffic[0]["timestamp"]) / 60 if traffic else 0.0
    return {
        "false_alerts": service.alerts_emitted,
        "attack_free_minutes": minutes,
        "false_alerts_per_hour": 60 * service.alerts_emitted / minutes if minutes else None,
    }


def _file_split(
    data_path: str | Path, window_seconds: float, test_size: float
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    X, y = build_dataset(load_jsonl(data_path), window_seconds)
    split = round(len(y) * (1.0 - test_size))
    if not 0 < split < len(y):
        raise ValueError(
            f"{data_path} has {len(y)} records - too few to split with test_size={test_size}"
        )
    y_train, y_test = y[:split], y[split:]
    for side, labels in (("training", y_train), ("test", y_test)):
        if NORMAL_LABEL not in labels:
            raise ValueError(
                f"the chronological split of {data_path} left the {side} set without "
                f"{NORMAL_LABEL!r} records; add normal traffic throughout the capture or "
                "change test_size"
            )
    return X[:split], y_train, X[split:], y_test


def _class_counts(labels: Sequence[str]) -> dict[str, int]:
    counts = Counter(labels)
    known = [label for label in LABELS if label in counts]
    others = sorted(set(counts) - set(LABELS))
    return {label: counts[label] for label in (*known, *others)}


def _report_header(result: TrainingResult) -> list[str]:
    training = result.model.metadata.get("training")
    training = training if isinstance(training, dict) else {}
    seeds, test_seeds = training.get("seeds") or [], training.get("test_seeds") or []
    test_streams = ""
    if result.simulated and seeds and test_seeds:
        source = f"simulated traffic, {_streams(seeds)}"
        test_streams = f" from {_streams(test_seeds)}, held out"
    elif result.simulated:
        source = "simulated traffic, tested on separate held-out streams (other seeds)"
    else:
        where = training.get("data_path") or "labelled records"
        source = f"{where} (chronological train/test split)"
    skipped = int(training.get("n_samples") or 0) - result.train_size
    left_out = f" ({skipped:,} first packets of attack episodes left out)" if skipped > 0 else ""
    title = "NeuralGuard model training report"
    lines = [
        title,
        "=" * len(title),
        f"Data:             {source}",
        f"Training set:     {result.train_size:,} records{left_out}",
        f"Test set:         {result.test_size:,} records{test_streams}",
        f"Feature window:   {result.model.window_seconds:g} s",
    ]
    if "n_estimators" in training:
        lines.append(f"Trees:            {training['n_estimators']}")
    if result.model.version != "unsaved":
        lines.append(f"Model version:    {result.model.version}")
    return lines


def _class_balance(class_counts: Mapping[str, int]) -> list[str]:
    if not class_counts:
        return ["  (no class counts recorded)"]
    total = sum(class_counts.values()) or 1
    width = max(len(label) for label in class_counts)
    return [
        f"  {label:<{width}}  {count:>9,}  {count / total:>6.1%}"
        for label, count in class_counts.items()
    ]


def _per_class_table(
    per_class: Mapping[str, Mapping[str, Any]], episodes: Mapping[str, Mapping[str, int]]
) -> list[str]:
    """Precision, recall, f1 and support (packets) per class; with ``episodes`` also the
    attack episodes detected, of all: a flood is hundreds of packets but one attack."""
    width = max(len("class"), *(len(label) for label in per_class))
    header = f"  {'class':<{width}}  {'precision':>9}  {'recall':>7}  {'f1':>7}  {'support':>8}"
    lines = [header + ("  episodes detected" if episodes else "")]
    for label, row in per_class.items():
        line = (
            f"  {label:<{width}}  {_number(row.get('precision')):>9}  "
            f"{_number(row.get('recall')):>7}  {_number(row.get('f1')):>7}  "
            f"{int(row.get('support', 0)):>8,}"
        )
        if episodes:
            counts = episodes.get(label)
            found = f"{counts['detected']} of {counts['episodes']}" if counts else "-"
            line += f"  {found:>17}"
        lines.append(line)
    return lines


def _spread(metrics: Mapping[str, Any], key: str) -> str:
    """``"   (97.41% to 98.79% over 4 streams)"``, or ``""`` for a single stream."""
    values = [row[key] for row in metrics.get("test_streams") or [] if row.get(key) is not None]
    if len(values) < 2:
        return ""
    return f"   ({_percent(min(values))} to {_percent(max(values))} over {len(values)} streams)"


def _alerting_lines(metrics: Mapping[str, Any]) -> list[str]:
    lines = []
    episodes = metrics.get("episodes") or {}
    if episodes:
        total = sum(counts["episodes"] for counts in episodes.values())
        detected = sum(counts["detected"] for counts in episodes.values())
        median = metrics.get("median_packets_before_detection")
        when = f", median {median:g} packets before the first" if median is not None else ""
        lines.append(f"  Attack episodes        {detected} of {total} detected{when}")
    per_hour = metrics.get("false_alerts_per_hour")
    if per_hour is not None:
        lines.append(
            f"  False alerts           {per_hour:.1f} per hour on attack-free traffic "
            f"({metrics.get('false_alerts')} in {metrics.get('attack_free_minutes', 0):.1f} "
            "minutes, through alert throttling)"
        )
    return lines


def _streams(seeds: Sequence[int]) -> str:
    """``"4 independent streams (seeds 42, 142, 242, 342)"``."""
    if len(seeds) == 1:
        return f"1 stream (seed {seeds[0]})"
    return f"{len(seeds)} independent streams (seeds {', '.join(map(str, seeds))})"


def _confusion_table(labels: Sequence[str], matrix: Sequence[Sequence[int]]) -> list[str]:
    row_width = max(len(label) for label in labels)
    biggest = max((int(count) for row in matrix for count in row), default=0)
    col_width = max(max(len(label) for label in labels), len(f"{biggest:,}"))
    header = " " * (row_width + 2) + "  ".join(f"{label:>{col_width}}" for label in labels)
    lines = ["  " + header]
    for label, row in zip(labels, matrix, strict=True):
        cells = "  ".join(f"{int(count):>{col_width},}" for count in row)
        lines.append(f"  {label:<{row_width}}  {cells}")
    return lines


def _rate(flags: np.ndarray) -> float | None:
    return float(np.mean(flags)) if len(flags) else None


def _number(value: Any, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(float(value), spec)


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"
