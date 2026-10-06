"""Real-time detection logic: record -> features -> model -> :class:`Detection` (no I/O here).

A :class:`Detector` ties together the three pieces that must agree with training:

1. ``schema.normalize_record`` - validates and canonicalises every raw record;
2. a :class:`~neuralguard.features.FeatureExtractor` - by default built with the
   *model's* sliding-window length, so live features are computed exactly like the
   features the model was trained on;
3. the model - anything with ``window_seconds``, ``predict_one(x)`` and ``predict(X)``
   returning :class:`~neuralguard.model.Prediction` objects (normally a
   :class:`~neuralguard.model.ThreatModel`).

A record is a threat when its ``threat_score`` (probability of *any* attack) reaches
the detector's threshold; ``severity_for`` maps the score onto low/medium/high/critical.

:meth:`Detector.process` handles one record and lets ``InvalidRecordError`` propagate.
:meth:`Detector.process_many` handles a batch: invalid records are skipped (logged and
counted), feature extraction stays sequential because the extractor is stateful, and the
model runs once for the whole batch - much faster than one call per packet.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neuralguard.features import FeatureExtractor
from neuralguard.model import ModelError, Prediction, ThreatModel
from neuralguard.schema import InvalidRecordError, normalize_record

logger = logging.getLogger(__name__)

SEVERITY_LEVELS: tuple[tuple[float, str], ...] = (
    (0.9, "critical"),
    (0.75, "high"),
    (0.6, "medium"),
)
LOWEST_SEVERITY = "low"
SEVERITIES = (LOWEST_SEVERITY, *(name for _, name in reversed(SEVERITY_LEVELS)))

_MAX_REASON_LENGTH = 200


def severity_for(score: float) -> str:
    """Severity of a threat score: "critical" >= 0.9, "high" >= 0.75, "medium" >= 0.6,
    otherwise "low"."""
    for minimum, name in SEVERITY_LEVELS:
        if score >= minimum:
            return name
    return LOWEST_SEVERITY


@dataclass(frozen=True)
class Detection:
    """The detector's verdict on one canonical traffic record.

    ``features`` is the vector the model saw (``features.FEATURE_NAMES`` order).
    ``attack_type`` and ``severity`` are ``None`` unless ``is_threat``.
    """

    record: dict[str, Any]
    features: np.ndarray
    threat_score: float
    is_threat: bool
    attack_type: str | None
    severity: str | None  # None when not a threat


@dataclass
class DetectorStats:
    """Running counters: valid records processed, threats among them, invalid records
    rejected, and threats per attack type."""

    processed: int = 0
    threats: int = 0
    invalid: int = 0
    by_attack_type: dict[str, int] = field(default_factory=dict)


class Detector:
    """Turns raw traffic records into :class:`Detection` objects.

    ``threshold`` must be in (0, 1]. When ``extractor`` is ``None`` a fresh
    ``FeatureExtractor(window_seconds=model.window_seconds)`` is used. Records must be
    fed in time order (the extractor keeps sliding-window state). Not thread-safe.
    """

    def __init__(
        self,
        model: ThreatModel,
        *,
        threshold: float = 0.5,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        self.model = model
        self.threshold = float(threshold)
        self.extractor = (
            FeatureExtractor(window_seconds=model.window_seconds)
            if extractor is None
            else extractor
        )
        self.stats = DetectorStats()

    def process(self, raw: Mapping[str, Any]) -> Detection:
        """Detect on one raw record.

        Raises :class:`~neuralguard.schema.InvalidRecordError` for a malformed record
        (after counting it in ``stats.invalid``).
        """
        try:
            record = normalize_record(raw)
        except InvalidRecordError:
            self.stats.invalid += 1
            raise
        features = self.extractor.transform_one(record)
        prediction = self.model.predict_one(features)
        return self._detection(record, features, prediction)

    def process_many(self, raws: Iterable[Mapping[str, Any]]) -> list[Detection]:
        """Detect on a batch of raw records, in input order.

        Invalid records are skipped: logged at WARNING with the reason and counted in
        ``stats.invalid``. The model is called once, with every valid record's features.
        """
        records: list[dict[str, Any]] = []
        rows: list[np.ndarray] = []
        for position, raw in enumerate(raws):
            try:
                record = normalize_record(raw)
            except InvalidRecordError as exc:
                self.stats.invalid += 1
                logger.warning("skipping invalid record #%d: %s", position, _reason(exc))
                continue
            records.append(record)
            rows.append(self.extractor.transform_one(record))
        if not records:
            return []

        predictions = self.model.predict(np.vstack(rows))
        if len(predictions) != len(records):
            raise ModelError(
                f"model returned {len(predictions)} predictions for {len(records)} records"
            )
        return [
            self._detection(record, features, prediction)
            for record, features, prediction in zip(records, rows, predictions, strict=True)
        ]

    def _detection(
        self, record: dict[str, Any], features: np.ndarray, prediction: Prediction
    ) -> Detection:
        score = float(prediction.threat_score)
        is_threat = score >= self.threshold
        attack_type = str(prediction.attack_type) if is_threat else None

        stats = self.stats
        stats.processed += 1
        if attack_type is not None:
            stats.threats += 1
            stats.by_attack_type[attack_type] = stats.by_attack_type.get(attack_type, 0) + 1

        return Detection(
            record=record,
            features=features,
            threat_score=score,
            is_threat=is_threat,
            attack_type=attack_type,
            severity=severity_for(score) if is_threat else None,
        )


def _reason(exc: Exception) -> str:
    """The error message, truncated: records are untrusted and may be arbitrarily large."""
    text = str(exc)
    if len(text) > _MAX_REASON_LENGTH:
        return text[:_MAX_REASON_LENGTH] + "..."
    return text
