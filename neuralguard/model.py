"""The threat-classification model and its on-disk bundle.

:class:`ThreatModel` wraps a multiclass scikit-learn ``RandomForestClassifier`` trained on
``features.FEATURE_NAMES`` with labels from ``schema.LABELS``:

* ``threat_score = 1 - P("normal")`` - how likely the packet is part of *any* attack;
* ``attack_type`` - the most probable non-normal class the model knows;
* ``predicted_label`` - the most probable class overall.

A trained model is saved as a joblib *bundle*: a dict holding the estimator together with
everything needed to use it safely - class order, the feature names it was trained on, the
sliding-window length the features were computed with, and metadata (versions, training
time, training parameters, metrics)::

    {"format": "neuralguard-model", "format_version": 1, "estimator": ...,
     "classes": [...], "feature_names": [...], "window_seconds": 10.0, "metadata": {...}}

Next to it, ``<path>.sha256`` holds the file's SHA-256 in ``sha256sum`` format, so
``sha256sum -c threat_model.joblib.sha256`` works and :meth:`ThreatModel.load` can refuse a
corrupted or tampered file *before* unpickling it.

SECURITY: joblib (pickle) executes code while loading. Only load model files you created
yourself or obtained from a trusted source; the checksum sidecar protects against
accidental corruption and casual tampering, not against an attacker who can rewrite both
files.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import io
import logging
import math
import os
import re
import secrets
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from neuralguard import __version__
from neuralguard.features import FEATURE_NAMES
from neuralguard.schema import LABELS, NORMAL_LABEL

logger = logging.getLogger(__name__)

MODEL_FORMAT = "neuralguard-model"
MODEL_FORMAT_VERSION = 1
CHECKSUM_SUFFIX = ".sha256"
UNSAVED_VERSION = "unsaved"
DEFAULT_THRESHOLD = 0.5

_TRAIN_HINT = "run `neuralguard train` to create a new model"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_VERSION_LENGTH = 12


class ModelError(RuntimeError):
    """Raised when a model cannot be loaded or used."""


@dataclass(frozen=True)
class Prediction:
    """The model's verdict on one feature vector.

    ``probabilities`` maps every class the model knows (in ``ThreatModel.classes`` order)
    to a plain ``float``. ``attack_type`` is always an attack class, even when
    ``threat_score`` is low - callers decide with a threshold whether it is a threat.
    """

    threat_score: float
    predicted_label: str
    attack_type: str
    probabilities: dict[str, float]


class ThreatModel:
    """A fitted classifier plus everything needed to use it consistently.

    ``estimator`` must provide ``predict_proba``. When it has ``classes_`` (every fitted
    scikit-learn classifier does) its columns are mapped onto ``classes`` by name, so the
    two may be in different orders; without ``classes_`` the columns are assumed to be
    in ``classes`` order. ``classes`` must include ``"normal"`` and at least one attack
    type, all from ``schema.LABELS``.
    """

    def __init__(
        self,
        estimator: Any,
        classes: Sequence[str],
        *,
        feature_names: Sequence[str] = FEATURE_NAMES,
        window_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not hasattr(estimator, "predict_proba"):
            raise ValueError("estimator must provide predict_proba()")
        self.estimator = estimator
        self.classes: tuple[str, ...] = _check_classes(classes)
        self.feature_names: tuple[str, ...] = tuple(str(name) for name in feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features is not None and int(n_features) != len(self.feature_names):
            raise ValueError(
                f"estimator expects {n_features} features but {len(self.feature_names)} "
                "feature names were given"
            )
        self.window_seconds = _check_window(window_seconds)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.path: Path | None = None
        self._version: str | None = None
        self._columns = _column_order(estimator, self.classes)
        self._normal_index = self.classes.index(NORMAL_LABEL)
        self._attack_indices = tuple(
            i for i, label in enumerate(self.classes) if label != NORMAL_LABEL
        )

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: Sequence[str],
        *,
        window_seconds: float,
        n_estimators: int = 200,
        random_state: int | None = None,
        n_jobs: int = -1,
        metadata: dict[str, Any] | None = None,
    ) -> ThreatModel:
        """Fit a class-balanced random forest on feature matrix ``X`` and labels ``y``.

        ``window_seconds`` is the feature window ``X`` was computed with; it is stored so
        live detection uses the same window. ``n_jobs`` only applies to fitting: the
        fitted estimator is switched to a single thread, because per-packet predictions
        through a worker pool are around 100x slower.
        """
        matrix = _as_matrix(X, len(FEATURE_NAMES))
        labels = [str(label) for label in y]
        if len(labels) != matrix.shape[0]:
            raise ValueError(f"X has {matrix.shape[0]} rows but y has {len(labels)} labels")
        present = set(labels)
        unknown = sorted(present - set(LABELS))
        if unknown:
            raise ValueError(f"unknown labels {unknown}; expected labels from {LABELS}")
        if NORMAL_LABEL not in present:
            raise ValueError(
                f"training data has no {NORMAL_LABEL!r} samples; the model must see "
                "benign traffic to learn what an attack is not"
            )
        if len(present) < 2:
            raise ValueError(
                "training data needs at least 2 classes (normal plus at least one attack "
                f"type), got only {sorted(present)}"
            )
        window = _check_window(window_seconds)

        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        logger.info(
            "fitting random forest: %d trees, %d samples, %d classes",
            n_estimators,
            matrix.shape[0],
            len(present),
        )
        estimator.fit(matrix, np.asarray(labels))
        _single_threaded(estimator)

        info: dict[str, Any] = {
            "neuralguard_version": __version__,
            "sklearn_version": sklearn.__version__,
            "trained_at": _utc_now(),
            "n_estimators": int(n_estimators),
            "random_state": random_state,
            "train_samples": len(labels),
        }
        info.update(metadata or {})
        classes = [label for label in LABELS if label in present]
        return cls(estimator, classes, window_seconds=window, metadata=info)

    @property
    def version(self) -> str:
        """Short id for alert documents: the first 12 hex chars of the saved file's SHA-256.

        ``"unsaved"`` for a model that has not been saved to or loaded from disk.
        """
        return self._version or UNSAVED_VERSION

    def predict(self, X: np.ndarray) -> list[Prediction]:
        """Predictions for a 2-D batch of feature vectors, in row order."""
        proba = self._probabilities(_as_matrix(X, len(self.feature_names)))
        return [self._prediction(row) for row in proba]

    def predict_one(self, x: np.ndarray) -> Prediction:
        """Prediction for a single 1-D feature vector."""
        vector = np.asarray(x, dtype=np.float64)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError(f"predict_one expects a 1-D feature vector, got shape {vector.shape}")
        return self.predict(vector.reshape(1, -1))[0]

    def evaluate(
        self, X: np.ndarray, y: Sequence[str], *, threshold: float = DEFAULT_THRESHOLD
    ) -> dict[str, Any]:
        """JSON-serialisable metrics of this model on labelled data ``(X, y)``.

        Keys: ``n_samples``, ``threshold``, ``accuracy``, ``roc_auc`` (threat vs normal,
        from ``threat_score``; ``None`` when ``y`` has a single class), ``detection_rate``
        (attacks with ``threat_score >= threshold``), ``false_positive_rate`` (normal
        records flagged), ``attack_type_accuracy`` (attacks whose ``attack_type`` is
        right), ``labels``, ``confusion_matrix`` (rows = true, columns = predicted, in
        ``labels`` order), ``per_class`` (``{label: {precision, recall, f1, support}}``),
        ``macro_avg`` and ``weighted_avg``. Rates are ``None`` when undefined.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        matrix = _as_matrix(X, len(self.feature_names))
        y_true = [str(label) for label in y]
        if len(y_true) != matrix.shape[0]:
            raise ValueError(f"X has {matrix.shape[0]} rows but y has {len(y_true)} labels")
        if not y_true:
            raise ValueError("cannot evaluate on an empty dataset")

        proba = self._probabilities(matrix)
        classes = np.asarray(self.classes, dtype=object)
        y_pred = [str(label) for label in classes[proba.argmax(axis=1)]]
        attack_classes = classes[list(self._attack_indices)]
        attack_cols = proba[:, list(self._attack_indices)]
        attack_pred = attack_classes[attack_cols.argmax(axis=1)]
        scores = np.clip(1.0 - proba[:, self._normal_index], 0.0, 1.0)
        flagged = scores >= threshold
        truth = np.asarray(y_true, dtype=object)
        is_attack = truth != NORMAL_LABEL

        labels = _ordered_labels(set(y_true) | set(y_pred))
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )
        roc_auc = (
            float(roc_auc_score(is_attack.astype(int), scores))
            if 0 < int(is_attack.sum()) < len(y_true)
            else None
        )
        return {
            "n_samples": len(y_true),
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": roc_auc,
            "detection_rate": _mean_or_none(flagged[is_attack]),
            "false_positive_rate": _mean_or_none(flagged[~is_attack]),
            "attack_type_accuracy": _mean_or_none(attack_pred[is_attack] == truth[is_attack]),
            "labels": labels,
            "confusion_matrix": _confusion_matrix(y_true, y_pred, labels),
            "per_class": {label: _report_row(report[label]) for label in labels},
            "macro_avg": _report_row(report["macro avg"]),
            "weighted_avg": _report_row(report["weighted avg"]),
        }

    def feature_importances(self) -> list[tuple[str, float]]:
        """``(feature name, importance)`` pairs, most important first ([] if unavailable)."""
        importances = getattr(self.estimator, "feature_importances_", None)
        if importances is None:
            return []
        pairs = [
            (name, float(value))
            for name, value in zip(self.feature_names, importances, strict=True)
        ]
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)

    def save(self, path: str | Path) -> Path:
        """Write the model bundle to ``path`` (atomically) plus its ``.sha256`` sidecar.

        Parent directories are created. Returns the path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "neuralguard_version": __version__,
            "sklearn_version": sklearn.__version__,
        }
        metadata.setdefault("trained_at", _utc_now())
        bundle = {
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION,
            "estimator": self.estimator,
            "classes": list(self.classes),
            "feature_names": list(self.feature_names),
            "window_seconds": self.window_seconds,
            "metadata": metadata,
        }
        # Write to a unique temporary file and rename it into place, so a crash mid-save
        # never leaves a truncated model where the detector would pick it up.
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            joblib.dump(bundle, tmp, compress=3)
            digest = _sha256_file(tmp)
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
        checksum_path(path).write_text(f"{digest}  {path.name}\n", encoding="ascii")

        self.metadata = metadata
        self.path = path
        self._version = digest[:_VERSION_LENGTH]
        logger.info("saved model %s to %s", self.version, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ThreatModel:
        """Load a bundle written by :meth:`save`.

        SECURITY: unpickling executes code - only load trusted files. When the
        ``.sha256`` sidecar exists the file's digest is verified *before* anything is
        unpickled. Raises :class:`ModelError` when the file is missing, unreadable,
        fails the checksum, is not a NeuralGuard bundle, or was built for a different
        feature set.
        """
        path = Path(path)
        if not path.is_file():
            raise ModelError(f"model file not found: {path}; {_TRAIN_HINT}")
        try:
            # Read once and unpickle the very bytes that were verified (no TOCTOU window).
            data = path.read_bytes()
        except OSError as exc:
            raise ModelError(f"cannot read model file {path}: {exc}") from exc
        digest = hashlib.sha256(data).hexdigest()
        _verify_checksum(path, digest)
        try:
            # Trusted input only (see docstring); integrity checked above when possible.
            bundle = joblib.load(io.BytesIO(data))
        except Exception as exc:  # unpickling can fail in arbitrary ways
            raise ModelError(
                f"{path} is not a readable model file ({type(exc).__name__}: {exc}); {_TRAIN_HINT}"
            ) from exc

        model = cls._from_bundle(bundle, path)
        model.path = path
        model._version = digest[:_VERSION_LENGTH]
        logger.info(
            "loaded model %s from %s (classes=%s, window=%.1fs)",
            model.version,
            path,
            ",".join(model.classes),
            model.window_seconds,
        )
        return model

    def __repr__(self) -> str:
        return (
            f"ThreatModel(version={self.version!r}, classes={list(self.classes)}, "
            f"window_seconds={self.window_seconds})"
        )

    @classmethod
    def _from_bundle(cls, bundle: Any, path: Path) -> ThreatModel:
        if not isinstance(bundle, dict) or bundle.get("format") != MODEL_FORMAT:
            raise ModelError(f"{path} is not a NeuralGuard model bundle; {_TRAIN_HINT}")
        format_version = bundle.get("format_version")
        if format_version != MODEL_FORMAT_VERSION:
            raise ModelError(
                f"{path} uses model format version {format_version!r}, this NeuralGuard "
                f"supports version {MODEL_FORMAT_VERSION}; {_TRAIN_HINT}"
            )
        feature_names = tuple(bundle.get("feature_names") or ())
        if feature_names != FEATURE_NAMES:
            raise ModelError(
                f"{path} was trained on different features ({len(feature_names)}) than this "
                f"NeuralGuard computes ({len(FEATURE_NAMES)}); {_TRAIN_HINT}"
            )
        metadata = bundle.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        trained_with = metadata.get("sklearn_version")
        if trained_with != sklearn.__version__:
            logger.warning(
                "model %s was saved with scikit-learn %s but %s is installed; predictions "
                "may be unreliable - retrain with `neuralguard train`",
                path,
                trained_with or "(unknown version)",
                sklearn.__version__,
            )
        estimator = bundle.get("estimator")
        _single_threaded(estimator)
        try:
            return cls(
                estimator,
                bundle["classes"],
                feature_names=feature_names,
                window_seconds=bundle["window_seconds"],
                metadata=metadata,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"{path} is not a valid model bundle ({exc}); {_TRAIN_HINT}") from exc

    def _probabilities(self, X: np.ndarray) -> np.ndarray:
        """``predict_proba`` with columns rearranged into ``self.classes`` order."""
        if X.shape[0] == 0:
            return np.empty((0, len(self.classes)), dtype=np.float64)
        proba = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        if proba.shape != (X.shape[0], len(self.classes)):
            raise ModelError(
                f"estimator returned probabilities of shape {proba.shape} for {X.shape[0]} "
                f"rows and {len(self.classes)} classes"
            )
        return proba[:, list(self._columns)]

    def _prediction(self, row: np.ndarray) -> Prediction:
        values = [float(value) for value in row]
        best = max(range(len(values)), key=values.__getitem__)
        attack = max(self._attack_indices, key=values.__getitem__)
        return Prediction(
            threat_score=min(1.0, max(0.0, 1.0 - values[self._normal_index])),
            predicted_label=self.classes[best],
            attack_type=self.classes[attack],
            probabilities=dict(zip(self.classes, values, strict=True)),
        )


def checksum_path(path: str | Path) -> Path:
    """The ``sha256sum``-format sidecar of a model file: ``<path>.sha256``."""
    path = Path(path)
    return path.with_name(path.name + CHECKSUM_SUFFIX)


def _verify_checksum(path: Path, digest: str) -> None:
    sidecar = checksum_path(path)
    if not sidecar.exists():
        logger.warning(
            "no checksum file %s; loading %s without integrity verification", sidecar, path
        )
        return
    try:
        fields = sidecar.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelError(f"cannot read checksum file {sidecar}: {exc}") from exc
    expected = fields[0].lower() if fields else ""
    if not _SHA256_HEX.fullmatch(expected):
        raise ModelError(f"malformed checksum file {sidecar}; {_TRAIN_HINT}")
    if not hmac.compare_digest(expected, digest):
        raise ModelError(
            f"checksum mismatch for {path}: it does not match {sidecar}, so it may be "
            f"corrupted or tampered with and was not loaded; {_TRAIN_HINT}"
        )


def _check_classes(classes: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(label) for label in classes)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate classes in {list(result)}")
    unknown = [label for label in result if label not in LABELS]
    if unknown:
        raise ValueError(f"unknown classes {unknown}; expected labels from {LABELS}")
    if NORMAL_LABEL not in result:
        raise ValueError(f"classes must include {NORMAL_LABEL!r}, got {list(result)}")
    if len(result) < 2:
        raise ValueError(f"classes must include at least one attack type, got {list(result)}")
    return result


def _column_order(estimator: Any, classes: tuple[str, ...]) -> tuple[int, ...]:
    """Index of each of ``classes`` among the estimator's ``predict_proba`` columns."""
    estimator_classes = getattr(estimator, "classes_", None)
    if estimator_classes is None:
        return tuple(range(len(classes)))
    known = [str(label) for label in estimator_classes]
    if sorted(known) != sorted(classes):
        raise ValueError(f"classes {list(classes)} do not match the estimator's classes {known}")
    return tuple(known.index(label) for label in classes)


def _check_window(window_seconds: float) -> float:
    window = float(window_seconds)
    if not math.isfinite(window) or window <= 0:
        raise ValueError(f"window_seconds must be a positive number, got {window_seconds}")
    return window


def _as_matrix(X: Any, n_features: int) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {matrix.shape}")
    if matrix.shape[1] != n_features:
        raise ValueError(f"expected {n_features} features per row, got {matrix.shape[1]}")
    return matrix


def _single_threaded(estimator: Any) -> None:
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1


def _ordered_labels(labels: set[str]) -> list[str]:
    """Labels in ``schema.LABELS`` order (anything unexpected sorted at the end)."""
    known = [label for label in LABELS if label in labels]
    return known + sorted(labels - set(LABELS))


def _confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """Counts of (true, predicted) pairs: rows = true label, columns = predicted label."""
    pairs = Counter(zip(y_true, y_pred, strict=True))
    return [[pairs[(true, predicted)] for predicted in labels] for true in labels]


def _report_row(row: dict[str, Any]) -> dict[str, float | int]:
    return {
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1-score"]),
        "support": int(row["support"]),
    }


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
