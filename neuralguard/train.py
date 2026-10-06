"""Training pipeline: build a dataset, fit the model, evaluate it honestly, save it.

Two data sources are supported:

* **Simulated traffic** (the default): the model is trained on one
  :class:`~neuralguard.simulator.TrafficSimulator` stream and evaluated on a *separate*
  stream generated with ``seed + 1`` - a held-out scenario with different hosts, timings
  and attack episodes, not a random split of the same stream.
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

from neuralguard.features import FEATURE_NAMES, FeatureExtractor
from neuralguard.model import ThreatModel
from neuralguard.schema import LABELS, NORMAL_LABEL, InvalidRecordError, normalize_record
from neuralguard.simulator import TrafficSimulator

logger = logging.getLogger(__name__)

DEFAULT_START_TIME = 1_700_000_000.0
METRIC_SUMMARY_KEYS = ("accuracy", "roc_auc", "detection_rate", "false_positive_rate")
TOP_FEATURES = 10
_MIN_SIMULATED_TEST_SAMPLES = 1_000

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


def simulated_dataset(
    n_samples: int,
    *,
    seed: int,
    attack_ratio: float,
    window_seconds: float,
    start_time: float = DEFAULT_START_TIME,
) -> tuple[np.ndarray, list[str]]:
    """``build_dataset`` over ``n_samples`` records of a seeded traffic simulation."""
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    simulator = TrafficSimulator(seed=seed, attack_ratio=attack_ratio, start_time=start_time)
    return build_dataset(simulator.records(n_samples), window_seconds)


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
    n_samples: int = 60_000,
    seed: int = 42,
    attack_ratio: float = 0.3,
    window_seconds: float = 10.0,
    n_estimators: int = 200,
    data_path: str | Path | None = None,
    test_size: float = 0.25,
) -> TrainingResult:
    """Build the dataset, fit a :class:`ThreatModel` and evaluate it on held-out data.

    Without ``data_path`` the model trains on ``n_samples`` simulated records (``seed``)
    and is tested on a separate simulated stream (``seed + 1``, ``n_samples // 4``
    records, at least 1000). With ``data_path`` the labelled JSONL file is split
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
    if data_path is None:
        X_train, y_train, X_test, y_test = _simulated_split(
            n_samples, seed=seed, attack_ratio=attack_ratio, window_seconds=window_seconds
        )
        params.update(
            source="simulated",
            n_samples=int(n_samples),
            attack_ratio=float(attack_ratio),
            test_seed=seed + 1,
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
    model.metadata["metrics"] = {key: metrics.get(key) for key in METRIC_SUMMARY_KEYS}
    logger.info(
        "held-out accuracy %s, detection rate %s, false-positive rate %s",
        _number(metrics.get("accuracy")),
        _percent(metrics.get("detection_rate")),
        _percent(metrics.get("false_positive_rate")),
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
        f"  Detection rate         {_percent(metrics.get('detection_rate'))}",
        f"  False-positive rate    {_percent(metrics.get('false_positive_rate'))}",
        f"  Attack-type accuracy   {_percent(metrics.get('attack_type_accuracy'))}",
    ]
    per_class = metrics.get("per_class") or {}
    if per_class:
        lines += ["", "Per-class results:", *_per_class_table(per_class)]
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
    n_samples: int, *, seed: int, attack_ratio: float, window_seconds: float
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    test_samples = max(n_samples // 4, min(n_samples, _MIN_SIMULATED_TEST_SAMPLES))
    logger.info("simulating %d training records (seed %d)", n_samples, seed)
    X_train, y_train = simulated_dataset(
        n_samples, seed=seed, attack_ratio=attack_ratio, window_seconds=window_seconds
    )
    logger.info("simulating %d held-out test records (seed %d)", test_samples, seed + 1)
    X_test, y_test = simulated_dataset(
        test_samples, seed=seed + 1, attack_ratio=attack_ratio, window_seconds=window_seconds
    )
    return X_train, y_train, X_test, y_test


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
    if result.simulated:
        source = "simulated traffic, tested on a separate held-out stream (different seed)"
    else:
        where = training.get("data_path") or "labelled records"
        source = f"{where} (chronological train/test split)"
    title = "NeuralGuard model training report"
    lines = [
        title,
        "=" * len(title),
        f"Data:             {source}",
        f"Training set:     {result.train_size:,} records",
        f"Test set:         {result.test_size:,} records",
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


def _per_class_table(per_class: Mapping[str, Mapping[str, Any]]) -> list[str]:
    width = max(len("class"), *(len(label) for label in per_class))
    lines = [f"  {'class':<{width}}  {'precision':>9}  {'recall':>7}  {'f1':>7}  {'support':>8}"]
    for label, row in per_class.items():
        lines.append(
            f"  {label:<{width}}  {_number(row.get('precision')):>9}  "
            f"{_number(row.get('recall')):>7}  {_number(row.get('f1')):>7}  "
            f"{int(row.get('support', 0)):>8,}"
        )
    return lines


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


def _number(value: Any, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(float(value), spec)


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"
