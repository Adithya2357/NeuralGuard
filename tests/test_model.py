import hashlib
import json
import logging
import re
from datetime import datetime, timezone

import joblib
import numpy as np
import pytest
import sklearn

from neuralguard import __version__
from neuralguard.features import FEATURE_NAMES
from neuralguard.model import (
    MODEL_FORMAT,
    MODEL_FORMAT_VERSION,
    ModelError,
    Prediction,
    ThreatModel,
    checksum_path,
)
from neuralguard.schema import ATTACK_TYPES

N_FEATURES = len(FEATURE_NAMES)
# Alphabetical order (what scikit-learn uses for classes_) differs from LABELS order here.
TRAIN_LABELS = ("normal", "port_scan", "icmp_flood")


def separable(labels=TRAIN_LABELS, per_class=60, seed=0):
    """Well-separated Gaussian blobs, one per label: class i is centred at 4 * i."""
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [rng.normal(4.0 * i, 0.5, size=(per_class, N_FEATURES)) for i in range(len(labels))]
    )
    y = [label for label in labels for _ in range(per_class)]
    return X, y


@pytest.fixture(scope="module")
def dataset():
    return separable()


@pytest.fixture(scope="module")
def trained(dataset):
    X, y = dataset
    return ThreatModel.train(
        X, y, window_seconds=5.0, n_estimators=15, random_state=7, metadata={"note": "unit"}
    )


class FakeEstimator:
    """predict_proba returns fixed rows; columns follow ``classes_`` (not LABELS)."""

    def __init__(self, classes, rows, n_jobs=4):
        self.classes_ = np.asarray(classes)
        self.rows = np.asarray(rows, dtype=float)
        self.n_features_in_ = N_FEATURES
        self.n_jobs = n_jobs

    def predict_proba(self, X):
        return np.vstack([self.rows[i % len(self.rows)] for i in range(len(X))])


def fake_model(rows, classes=("udp_flood", "normal", "port_scan"), order=None):
    estimator = FakeEstimator(classes, rows)
    return ThreatModel(estimator, order or ["normal", "port_scan", "udp_flood"], window_seconds=10)


# --------------------------------------------------------------------------- training


def test_train_learns_and_uses_labels_order(trained, dataset):
    X, y = dataset
    assert list(trained.estimator.classes_) == sorted(TRAIN_LABELS)
    assert trained.classes == ("normal", "port_scan", "icmp_flood")
    assert trained.feature_names == FEATURE_NAMES
    assert trained.window_seconds == 5.0
    predictions = trained.predict(X)
    assert [p.predicted_label for p in predictions] == y


def test_train_sets_single_threaded_estimator_and_forest_params(dataset):
    X, y = dataset
    model = ThreatModel.train(X, y, window_seconds=10, n_estimators=5, random_state=0, n_jobs=-1)
    params = model.estimator.get_params()
    assert model.estimator.n_jobs == 1
    assert params["class_weight"] == "balanced_subsample"
    assert params["min_samples_leaf"] == 2
    assert params["n_estimators"] == 5


def test_train_metadata(trained):
    meta = trained.metadata
    assert meta["note"] == "unit"
    assert meta["neuralguard_version"] == __version__
    assert meta["sklearn_version"] == sklearn.__version__
    trained_at = datetime.fromisoformat(meta["trained_at"])
    assert trained_at.utcoffset() == timezone.utc.utcoffset(None)
    json.dumps(meta)


def test_train_is_deterministic_with_random_state(dataset):
    X, y = dataset
    a = ThreatModel.train(X, y, window_seconds=10, n_estimators=5, random_state=3)
    b = ThreatModel.train(X, y, window_seconds=10, n_estimators=5, random_state=3)
    noisy = X + np.random.default_rng(9).normal(0, 2.0, size=X.shape)
    assert a.predict(noisy) == b.predict(noisy)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (["port_scan", "syn_flood"], "normal"),
        (["normal", "normal"], "at least 2 classes"),
        (["normal", "bogus"], "unknown labels"),
    ],
)
def test_train_rejects_bad_labels(labels, message):
    X, y = separable(labels, per_class=5)
    with pytest.raises(ValueError, match=message):
        ThreatModel.train(X, y, window_seconds=10, n_estimators=2)


def test_train_rejects_bad_shapes():
    X, y = separable(per_class=5)
    with pytest.raises(ValueError, match="features per row"):
        ThreatModel.train(X[:, :-1], y, window_seconds=10, n_estimators=2)
    with pytest.raises(ValueError, match="2-D"):
        ThreatModel.train(X[0], y[:1], window_seconds=10, n_estimators=2)
    with pytest.raises(ValueError, match="labels"):
        ThreatModel.train(X, y[:-1], window_seconds=10, n_estimators=2)
    with pytest.raises(ValueError, match="window_seconds"):
        ThreatModel.train(X, y, window_seconds=0, n_estimators=2)


# --------------------------------------------------------------------------- constructor


def test_constructor_validates_classes():
    estimator = FakeEstimator(["normal", "port_scan"], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="do not match"):
        ThreatModel(estimator, ["normal", "syn_flood"], window_seconds=10)
    with pytest.raises(ValueError, match="normal"):
        ThreatModel(
            FakeEstimator(["port_scan", "syn_flood"], [[1, 0]]),
            ["port_scan", "syn_flood"],
            window_seconds=10,
        )
    with pytest.raises(ValueError, match="unknown"):
        ThreatModel(estimator, ["normal", "port_scan", "nope"], window_seconds=10)
    with pytest.raises(ValueError, match="duplicate"):
        ThreatModel(estimator, ["normal", "normal"], window_seconds=10)
    with pytest.raises(ValueError, match="attack type"):
        ThreatModel(FakeEstimator(["normal"], [[1.0]]), ["normal"], window_seconds=10)
    with pytest.raises(ValueError, match="window_seconds"):
        ThreatModel(estimator, ["normal", "port_scan"], window_seconds=-1)
    with pytest.raises(ValueError, match="predict_proba"):
        ThreatModel(object(), ["normal", "port_scan"], window_seconds=10)


def test_constructor_rejects_feature_count_mismatch():
    estimator = FakeEstimator(["normal", "port_scan"], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="features"):
        ThreatModel(estimator, ["normal", "port_scan"], feature_names=["a", "b"], window_seconds=1)


def test_estimator_without_classes_attribute_uses_given_order():
    class Bare:
        def predict_proba(self, X):
            return np.tile([0.2, 0.8], (len(X), 1))

    model = ThreatModel(Bare(), ["normal", "syn_flood"], window_seconds=1)
    prediction = model.predict_one(np.zeros(N_FEATURES))
    assert prediction.probabilities == {"normal": 0.2, "syn_flood": 0.8}
    assert prediction.attack_type == "syn_flood"


# --------------------------------------------------------------------------- prediction


def test_probabilities_are_mapped_by_class_name_not_column_position():
    # estimator columns: udp_flood, normal, port_scan
    model = fake_model([[0.1, 0.7, 0.2]])
    prediction = model.predict_one(np.zeros(N_FEATURES))
    assert prediction == Prediction(
        threat_score=pytest.approx(0.3),
        predicted_label="normal",
        attack_type="port_scan",
        probabilities={"normal": 0.7, "port_scan": 0.2, "udp_flood": 0.1},
    )
    assert list(prediction.probabilities) == ["normal", "port_scan", "udp_flood"]


def test_prediction_values_are_plain_python_types(trained, dataset):
    X, _ = dataset
    prediction = trained.predict_one(X[-1])
    assert type(prediction.threat_score) is float
    assert all(type(value) is float for value in prediction.probabilities.values())
    assert set(prediction.probabilities) == set(trained.classes)
    json.dumps(prediction.probabilities)


def test_only_known_classes_reported_and_attack_type_is_best_attack(trained, dataset):
    X, y = dataset
    for prediction, label in zip(trained.predict(X), y, strict=True):
        assert set(prediction.probabilities) == set(TRAIN_LABELS)
        assert prediction.attack_type in ATTACK_TYPES
        attacks = {k: v for k, v in prediction.probabilities.items() if k != "normal"}
        assert prediction.probabilities[prediction.attack_type] == max(attacks.values())
        assert prediction.threat_score == pytest.approx(1 - prediction.probabilities["normal"])
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
        if label != "normal":
            assert prediction.attack_type == label


def test_attack_type_tie_breaks_in_labels_order():
    model = fake_model([[0.25, 0.5, 0.25]])
    assert model.predict_one(np.zeros(N_FEATURES)).attack_type == "port_scan"


def test_threat_score_is_clamped_to_unit_interval():
    model = fake_model([[0.0, 1.0000000002, 0.0]])
    assert model.predict_one(np.zeros(N_FEATURES)).threat_score == 0.0


def test_predict_batch_matches_predict_one(trained, dataset):
    X, _ = dataset
    batch = trained.predict(X[::17])
    assert batch == [trained.predict_one(row) for row in X[::17]]


def test_predict_edge_cases(trained):
    assert trained.predict(np.empty((0, N_FEATURES))) == []
    with pytest.raises(ValueError, match="features per row"):
        trained.predict(np.zeros((2, N_FEATURES + 1)))
    with pytest.raises(ValueError, match="2-D"):
        trained.predict(np.zeros(N_FEATURES))
    with pytest.raises(ValueError, match="1-D"):
        trained.predict_one(np.zeros((2, N_FEATURES)))
    # a single-row matrix is accepted by predict_one
    assert trained.predict_one(np.zeros((1, N_FEATURES))) == trained.predict_one(
        np.zeros(N_FEATURES)
    )


def test_wrong_probability_shape_from_estimator_raises_model_error():
    class Broken(FakeEstimator):
        def predict_proba(self, X):
            return np.ones((len(X), 2))

    model = ThreatModel(
        Broken(["normal", "port_scan", "udp_flood"], [[1, 0, 0]]),
        ["normal", "port_scan", "udp_flood"],
        window_seconds=1,
    )
    with pytest.raises(ModelError, match="shape"):
        model.predict_one(np.zeros(N_FEATURES))


# --------------------------------------------------------------------------- evaluation


def test_evaluate_metrics_with_exact_values():
    # Row 0 -> normal (score 0.2), row 1 -> port_scan (score 0.9), row 2 -> udp_flood (0.6)
    rows = [[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.5, 0.4, 0.1]]
    model = fake_model(rows)
    X = np.zeros((6, N_FEATURES))
    y = ["normal", "port_scan", "udp_flood", "normal", "normal", "udp_flood"]
    # predictions: normal, port_scan, udp_flood, normal, port_scan, udp_flood
    metrics = model.evaluate(X, y)
    json.dumps(metrics)
    assert metrics["n_samples"] == 6
    assert metrics["threshold"] == 0.5
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["labels"] == ["normal", "port_scan", "udp_flood"]
    assert metrics["confusion_matrix"] == [[2, 1, 0], [0, 1, 0], [0, 0, 2]]
    assert metrics["detection_rate"] == 1.0
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["attack_type_accuracy"] == 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["per_class"]["normal"] == {
        "precision": 1.0,
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(0.8),
        "support": 3,
    }
    assert metrics["per_class"]["port_scan"]["precision"] == 0.5
    assert set(metrics["macro_avg"]) == {"precision", "recall", "f1", "support"}
    assert metrics["weighted_avg"]["support"] == 6


def test_evaluate_on_trained_model_is_json_serialisable(trained, dataset):
    X, y = dataset
    metrics = trained.evaluate(X, y)
    decoded = json.loads(json.dumps(metrics))
    assert decoded["accuracy"] == 1.0
    assert decoded["roc_auc"] == 1.0
    assert decoded["labels"] == list(TRAIN_LABELS)
    assert len(decoded["confusion_matrix"]) == len(TRAIN_LABELS)
    expected = {
        "n_samples",
        "threshold",
        "accuracy",
        "roc_auc",
        "detection_rate",
        "false_positive_rate",
        "attack_type_accuracy",
        "labels",
        "confusion_matrix",
        "per_class",
        "macro_avg",
        "weighted_avg",
    }
    assert expected <= set(metrics)


def test_evaluate_single_class_gives_none_for_undefined_metrics(trained, dataset):
    X, y = dataset
    normal_only = [i for i, label in enumerate(y) if label == "normal"]
    metrics = trained.evaluate(X[normal_only], [y[i] for i in normal_only])
    assert metrics["roc_auc"] is None
    assert metrics["detection_rate"] is None
    assert metrics["attack_type_accuracy"] is None
    assert metrics["false_positive_rate"] == 0.0
    json.dumps(metrics)


def test_evaluate_includes_labels_unknown_to_the_model(trained, dataset):
    X, y = dataset
    y = list(y)
    y[-1] = "syn_flood"  # the model never saw syn_flood
    metrics = trained.evaluate(X, y)
    assert metrics["labels"] == ["normal", "port_scan", "syn_flood", "icmp_flood"]
    assert metrics["per_class"]["syn_flood"]["recall"] == 0.0


def test_evaluate_validates_input(trained):
    with pytest.raises(ValueError, match="empty"):
        trained.evaluate(np.empty((0, N_FEATURES)), [])
    with pytest.raises(ValueError, match="labels"):
        trained.evaluate(np.zeros((2, N_FEATURES)), ["normal"])
    with pytest.raises(ValueError, match="threshold"):
        trained.evaluate(np.zeros((1, N_FEATURES)), ["normal"], threshold=0)


def test_feature_importances(trained):
    importances = trained.feature_importances()
    assert {name for name, _ in importances} == set(FEATURE_NAMES)
    values = [value for _, value in importances]
    assert values == sorted(values, reverse=True)
    assert all(type(value) is float for value in values)
    assert sum(values) == pytest.approx(1.0)
    assert fake_model([[0.1, 0.8, 0.1]]).feature_importances() == []


# --------------------------------------------------------------------------- save / load


def test_version_is_unsaved_until_saved(trained):
    assert ThreatModel(trained.estimator, trained.classes, window_seconds=5).version == "unsaved"


def test_save_writes_bundle_and_sha256sum_sidecar(trained, tmp_path):
    path = trained.save(tmp_path / "nested" / "dir" / "model.joblib")
    assert path == tmp_path / "nested" / "dir" / "model.joblib"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = checksum_path(path)
    assert sidecar.name == "model.joblib.sha256"
    assert sidecar.read_text() == f"{digest}  model.joblib\n"
    assert re.fullmatch(r"[0-9a-f]{12}", trained.version)
    assert trained.version == digest[:12]
    assert sorted(p.name for p in path.parent.iterdir()) == ["model.joblib", "model.joblib.sha256"]

    bundle = joblib.load(path)
    assert bundle["format"] == MODEL_FORMAT == "neuralguard-model"
    assert bundle["format_version"] == MODEL_FORMAT_VERSION == 1
    assert bundle["classes"] == list(trained.classes)
    assert bundle["feature_names"] == list(FEATURE_NAMES)
    assert bundle["window_seconds"] == 5.0
    assert {"neuralguard_version", "sklearn_version", "trained_at", "note"} <= set(
        bundle["metadata"]
    )


def test_save_into_a_directory_names_the_directory(trained, tmp_path):
    target = tmp_path / "models"
    target.mkdir()
    version = trained.version
    with pytest.raises(IsADirectoryError) as excinfo:
        trained.save(target)
    assert excinfo.value.filename == str(target)
    assert list(target.iterdir()) == []  # no temporary file left behind
    assert trained.version == version


def test_save_load_round_trip(trained, dataset, tmp_path):
    X, _ = dataset
    path = trained.save(tmp_path / "model.joblib")
    loaded = ThreatModel.load(str(path))
    assert loaded.version == trained.version
    assert loaded.path == path
    assert loaded.classes == trained.classes
    assert loaded.window_seconds == 5.0
    assert loaded.metadata["note"] == "unit"
    assert loaded.estimator.n_jobs == 1
    assert loaded.predict(X) == trained.predict(X)


def test_load_sets_estimator_single_threaded(trained, tmp_path):
    estimator = FakeEstimator(["normal", "port_scan"], [[0.5, 0.5]], n_jobs=-1)
    path = ThreatModel(estimator, ["normal", "port_scan"], window_seconds=1).save(tmp_path / "m")
    assert ThreatModel.load(path).estimator.n_jobs == 1


def test_load_missing_file_hints_at_training(tmp_path):
    with pytest.raises(ModelError, match="neuralguard train") as excinfo:
        ThreatModel.load(tmp_path / "absent.joblib")
    assert "not found" in str(excinfo.value)
    with pytest.raises(ModelError, match="neuralguard train"):
        ThreatModel.load(tmp_path)  # a directory is not a model file


def test_checksum_mismatch_is_detected_before_unpickling(trained, tmp_path, monkeypatch):
    path = trained.save(tmp_path / "model.joblib")
    with path.open("ab") as handle:
        handle.write(b"tampered")

    def must_not_unpickle(*args, **kwargs):
        raise AssertionError("joblib.load called on a file that failed verification")

    monkeypatch.setattr(joblib, "load", must_not_unpickle)
    with pytest.raises(ModelError, match=r"checksum mismatch.*neuralguard train"):
        ThreatModel.load(path)


def test_malformed_sidecar_is_rejected(trained, tmp_path):
    path = trained.save(tmp_path / "model.joblib")
    checksum_path(path).write_text("not-a-digest  model.joblib\n")
    with pytest.raises(ModelError, match="malformed checksum"):
        ThreatModel.load(path)
    checksum_path(path).write_text("")
    with pytest.raises(ModelError, match="malformed checksum"):
        ThreatModel.load(path)


def test_uppercase_or_binary_mode_sidecar_is_accepted(trained, tmp_path):
    path = trained.save(tmp_path / "model.joblib")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path(path).write_text(f"{digest.upper()} *model.joblib\n")
    assert ThreatModel.load(path).version == digest[:12]


def test_missing_sidecar_loads_with_warning(trained, tmp_path, caplog):
    path = trained.save(tmp_path / "model.joblib")
    checksum_path(path).unlink()
    with caplog.at_level(logging.WARNING, logger="neuralguard.model"):
        loaded = ThreatModel.load(path)
    assert loaded.version == trained.version
    assert "without integrity verification" in caplog.text


def write_bundle(path, **changes):
    bundle = {
        "format": MODEL_FORMAT,
        "format_version": MODEL_FORMAT_VERSION,
        "estimator": FakeEstimator(["normal", "port_scan"], [[0.9, 0.1]]),
        "classes": ["normal", "port_scan"],
        "feature_names": list(FEATURE_NAMES),
        "window_seconds": 10.0,
        "metadata": {"sklearn_version": sklearn.__version__},
    }
    bundle.update(changes)
    joblib.dump(bundle, path)
    return path


def test_hand_written_bundle_without_sidecar_loads(tmp_path):
    model = ThreatModel.load(write_bundle(tmp_path / "ok.joblib"))
    assert model.predict_one(np.zeros(N_FEATURES)).threat_score == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([1, 2, 3], "not a NeuralGuard model bundle"),
        ({"format": "something-else"}, "not a NeuralGuard model bundle"),
    ],
)
def test_load_rejects_foreign_pickles(tmp_path, payload, message):
    path = tmp_path / "foreign.joblib"
    joblib.dump(payload, path)
    with pytest.raises(ModelError, match=f"{message}.*neuralguard train"):
        ThreatModel.load(path)


def test_load_rejects_garbage_file(tmp_path):
    path = tmp_path / "garbage.joblib"
    path.write_bytes(b"\x00 definitely not a pickle")
    with pytest.raises(ModelError, match=r"not a readable model file.*neuralguard train"):
        ThreatModel.load(path)


def test_load_rejects_feature_name_mismatch(tmp_path):
    path = write_bundle(tmp_path / "old.joblib", feature_names=list(reversed(FEATURE_NAMES)))
    with pytest.raises(ModelError, match=r"different features.*neuralguard train"):
        ThreatModel.load(path)
    path = write_bundle(tmp_path / "older.joblib", feature_names=list(FEATURE_NAMES[:-1]))
    with pytest.raises(ModelError, match="different features"):
        ThreatModel.load(path)


def test_load_rejects_unsupported_format_version(tmp_path):
    path = write_bundle(tmp_path / "future.joblib", format_version=99)
    with pytest.raises(ModelError, match="format version 99"):
        ThreatModel.load(path)


@pytest.mark.parametrize(
    "changes",
    [
        {"classes": ["normal", "syn_flood"]},
        {"classes": ["port_scan"]},
        {"window_seconds": -5},
        {"estimator": None},
    ],
)
def test_load_rejects_inconsistent_bundles(tmp_path, changes):
    path = write_bundle(tmp_path / "bad.joblib", **changes)
    with pytest.raises(ModelError, match="not a valid model bundle"):
        ThreatModel.load(path)


def test_load_warns_on_sklearn_version_mismatch(tmp_path, caplog):
    path = write_bundle(tmp_path / "v.joblib", metadata={"sklearn_version": "0.0.1"})
    with caplog.at_level(logging.WARNING, logger="neuralguard.model"):
        ThreatModel.load(path)
    assert "scikit-learn 0.0.1" in caplog.text
    assert sklearn.__version__ in caplog.text


def test_resaving_keeps_metadata_and_updates_version(trained, tmp_path):
    first = trained.save(tmp_path / "a.joblib")
    loaded = ThreatModel.load(first)
    loaded.metadata["extra"] = True
    loaded.save(tmp_path / "b.joblib")
    reloaded = ThreatModel.load(tmp_path / "b.joblib")
    assert reloaded.metadata["extra"] is True
    assert reloaded.metadata["trained_at"] == trained.metadata["trained_at"]
    assert reloaded.version == hashlib.sha256((tmp_path / "b.joblib").read_bytes()).hexdigest()[:12]
