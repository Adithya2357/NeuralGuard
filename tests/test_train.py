import json
import random
import re
from collections import Counter

import numpy as np
import pytest

from neuralguard import train
from neuralguard.features import FEATURE_NAMES, FeatureExtractor
from neuralguard.model import ThreatModel
from neuralguard.schema import LABELS, InvalidRecordError, normalize_record
from neuralguard.train import (
    METRIC_SUMMARY_KEYS,
    SIMULATED_CAVEAT,
    TrainingResult,
    build_dataset,
    format_report,
    load_jsonl,
    simulated_dataset,
    train_model,
)

N_FEATURES = len(FEATURE_NAMES)


def rec(ts, label="normal", *, src="192.168.1.10", dst="10.0.0.5", dport=443, flags="A", **extra):
    record = {
        "timestamp": ts,
        "source_ip": src,
        "destination_ip": dst,
        "protocol": "TCP",
        "source_port": 40000,
        "destination_port": dport,
        "length": 60,
        "ttl": 64,
        "tcp_flags": flags,
        "label": label,
    }
    record.update(extra)
    return record


def labelled_stream(n=400, seed=0, start=1_000.0, attack_blocks=lambda block: block % 2 == 1):
    """Hand-made traffic: normal LAN chatter, with port-scan bursts in alternating
    blocks of 40 packets (every other packet of an attack block is a scan probe)."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        ts = start + 0.05 * i
        if attack_blocks(i // 40) and i % 2 == 0:
            records.append(
                rec(
                    ts,
                    "port_scan",
                    src="203.0.113.7",
                    dst="192.168.1.20",
                    dport=rng.randint(1, 1024),
                    flags="S",
                    length=44,
                )
            )
        else:
            records.append(
                rec(
                    ts,
                    "normal",
                    src=f"192.168.1.{rng.randint(2, 20)}",
                    dst=f"10.0.0.{rng.randint(1, 5)}",
                    dport=rng.choice([80, 443, 22]),
                    flags=rng.choice(["S", "SA", "A", "PA", "FA"]),
                    length=rng.randint(60, 1500),
                )
            )
    return records


def write_jsonl(path, records, *, shuffle_seed=None, blank_lines=False):
    records = list(records)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(records)
    lines = []
    for record in records:
        lines.append(json.dumps(record))
        if blank_lines:
            lines.append("   ")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- build_dataset


def test_build_dataset_matches_feature_extractor():
    records = labelled_stream(60)
    X, y = build_dataset(records, window_seconds=3.0)
    expected = FeatureExtractor(window_seconds=3.0).transform(
        [normalize_record(r) for r in records]
    )
    assert X.shape == (60, N_FEATURES)
    np.testing.assert_array_equal(X, expected)
    assert y == [r["label"] for r in records]


def test_build_dataset_uses_a_fresh_extractor_and_accepts_iterators():
    records = labelled_stream(30)
    first, _ = build_dataset(iter(records), window_seconds=5.0)
    second, _ = build_dataset((r for r in records), window_seconds=5.0)
    np.testing.assert_array_equal(first, second)


def test_build_dataset_window_changes_features():
    records = labelled_stream(80)
    narrow, _ = build_dataset(records, window_seconds=0.1)
    wide, _ = build_dataset(records, window_seconds=60.0)
    column = FEATURE_NAMES.index("src_packet_count")
    assert wide[:, column].sum() > narrow[:, column].sum()


def test_build_dataset_empty():
    X, y = build_dataset([], window_seconds=10.0)
    assert X.shape == (0, N_FEATURES)
    assert y == []


def test_build_dataset_requires_labels():
    records = labelled_stream(3)
    del records[1]["label"]
    with pytest.raises(ValueError, match="record 1 has no label"):
        build_dataset(records, window_seconds=10.0)


def test_build_dataset_reports_invalid_records():
    records = labelled_stream(3)
    records[2]["destination_port"] = 70000
    with pytest.raises(InvalidRecordError, match="record 2: destination_port"):
        build_dataset(records, window_seconds=10.0)
    records = labelled_stream(3)
    records[0]["label"] = "teardrop"
    with pytest.raises(ValueError, match="record 0: label"):
        build_dataset(records, window_seconds=10.0)


def test_build_dataset_validates_window():
    with pytest.raises(ValueError, match="window_seconds"):
        build_dataset(labelled_stream(3), window_seconds=0)


# --------------------------------------------------------------------------- load_jsonl


def test_load_jsonl_sorts_by_timestamp_and_skips_blank_lines(tmp_path):
    records = labelled_stream(50)
    path = write_jsonl(tmp_path / "data.jsonl", records, shuffle_seed=3, blank_lines=True)
    loaded = load_jsonl(path)
    assert loaded == [normalize_record(r) for r in records]
    assert [r["timestamp"] for r in loaded] == sorted(r["timestamp"] for r in loaded)


def test_load_jsonl_keeps_file_order_for_equal_timestamps(tmp_path):
    records = [
        rec(5.0, dport=1),
        rec(1.0, dport=2),
        rec(5.0, dport=3),
        rec(5.0, "port_scan", dport=4, flags="S"),
    ]
    loaded = load_jsonl(write_jsonl(tmp_path / "ties.jsonl", records))
    assert [r["destination_port"] for r in loaded] == [2, 1, 3, 4]


def test_load_jsonl_canonicalises_records(tmp_path):
    path = write_jsonl(tmp_path / "raw.jsonl", [rec(1.0, flags="as", protocol="tcp", extra=1)])
    (record,) = load_jsonl(path)
    assert record["tcp_flags"] == "SA"
    assert record["protocol"] == "TCP"
    assert "extra" not in record


def test_load_jsonl_bad_json_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(rec(1.0)) + "\n\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 3: invalid JSON"):
        load_jsonl(path)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("[1, 2]", r"line 2: expected a JSON object, got list"),
        ('"text"', r"line 2: expected a JSON object, got str"),
        (json.dumps({**rec(2.0), "timestamp": None}), r"line 2: record has no timestamp"),
        (json.dumps({k: v for k, v in rec(2.0).items() if k != "label"}), r"line 2: .*no label"),
        (json.dumps(rec(2.0, ttl=999)), r"line 2: ttl"),
    ],
)
def test_load_jsonl_rejects_bad_records(tmp_path, line, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(rec(1.0)) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_jsonl(path)


def test_load_jsonl_unlabelled_records_allowed_on_request(tmp_path):
    unlabelled = {k: v for k, v in rec(1.0).items() if k != "label"}
    path = write_jsonl(tmp_path / "capture.jsonl", [unlabelled])
    assert "label" not in load_jsonl(path, require_label=False)[0]


def test_load_jsonl_unreadable_files(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_jsonl(tmp_path / "missing.jsonl")
    binary = tmp_path / "binary.jsonl"
    binary.write_bytes(b"\xff\xfe\x00garbage")
    with pytest.raises(ValueError, match="UTF-8"):
        load_jsonl(binary)


# --------------------------------------------------------------------------- train_model on a file


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "captures.jsonl"
    return write_jsonl(path, labelled_stream(400), shuffle_seed=1)


@pytest.fixture(scope="module")
def file_result(data_file):
    return train_model(data_path=data_file, window_seconds=2.0, n_estimators=12, seed=1)


def test_train_model_from_file_splits_chronologically(data_file, monkeypatch):
    seen = {}
    original_train = ThreatModel.train.__func__
    original_evaluate = ThreatModel.evaluate

    def spy_train(cls, X, y, **kwargs):
        seen["train"] = (X, list(y))
        return original_train(cls, X, y, **kwargs)

    def spy_evaluate(self, X, y, **kwargs):
        seen["test"] = (X, list(y))
        return original_evaluate(self, X, y, **kwargs)

    monkeypatch.setattr(ThreatModel, "train", classmethod(spy_train))
    monkeypatch.setattr(ThreatModel, "evaluate", spy_evaluate)
    result = train_model(data_path=data_file, window_seconds=2.0, n_estimators=5, seed=0)

    # Features are computed over the whole time-ordered stream, then split by time.
    X_all, y_all = build_dataset(sorted(labelled_stream(400), key=lambda r: r["timestamp"]), 2.0)
    np.testing.assert_array_equal(seen["train"][0], X_all[:300])
    np.testing.assert_array_equal(seen["test"][0], X_all[300:])
    assert seen["train"][1] == y_all[:300]
    assert seen["test"][1] == y_all[300:]
    assert (result.train_size, result.test_size) == (300, 100)


def test_train_model_from_file_result(file_result, data_file):
    result = file_result
    assert isinstance(result, TrainingResult)
    assert result.simulated is False
    assert (result.train_size, result.test_size) == (300, 100)
    first_300 = [r["label"] for r in labelled_stream(400)[:300]]
    assert result.class_counts == dict(Counter(first_300))
    assert list(result.class_counts) == ["normal", "port_scan"]  # LABELS order
    assert result.model.window_seconds == 2.0
    assert result.model.classes == ("normal", "port_scan")
    assert result.metrics["n_samples"] == 100
    assert result.metrics["accuracy"] > 0.9
    assert result.metrics["detection_rate"] > 0.8
    json.dumps(result.metrics)


def test_train_model_stores_params_and_metrics_in_metadata(file_result, data_file, tmp_path):
    metadata = file_result.model.metadata
    training = metadata["training"]
    assert training["source"] == "jsonl"
    assert training["data_path"] == str(data_file)
    assert training["window_seconds"] == 2.0
    assert training["n_estimators"] == 12
    assert training["test_size"] == 0.25
    assert (training["train_records"], training["test_records"]) == (300, 100)
    assert set(metadata["metrics"]) == set(METRIC_SUMMARY_KEYS)
    for key in METRIC_SUMMARY_KEYS:
        assert metadata["metrics"][key] == file_result.metrics[key]
    json.dumps(metadata)
    # and it survives a save/load round trip
    reloaded = ThreatModel.load(file_result.model.save(tmp_path / "m.joblib"))
    assert reloaded.metadata["metrics"] == metadata["metrics"]
    assert reloaded.metadata["training"] == training


def test_train_model_custom_test_size(data_file):
    result = train_model(data_path=data_file, test_size=0.5, n_estimators=3)
    assert (result.train_size, result.test_size) == (200, 200)


def test_train_model_rejects_split_without_normal_in_test_set(tmp_path):
    records = labelled_stream(300)
    tail = [rec(1_000 + 20 + 0.01 * i, "port_scan", dport=i + 1, flags="S") for i in range(100)]
    path = write_jsonl(tmp_path / "attack_tail.jsonl", records + tail)
    with pytest.raises(ValueError, match="test set without 'normal'"):
        train_model(data_path=path, n_estimators=3)


def test_train_model_rejects_split_without_normal_in_training_set(tmp_path):
    head = [rec(0.01 * i, "port_scan", dport=i + 1, flags="S") for i in range(300)]
    path = write_jsonl(tmp_path / "attack_head.jsonl", head + labelled_stream(100, start=10.0))
    with pytest.raises(ValueError, match="training set without 'normal'"):
        train_model(data_path=path, n_estimators=3)


def test_train_model_rejects_too_small_file(tmp_path):
    path = write_jsonl(tmp_path / "one.jsonl", [rec(1.0)])
    with pytest.raises(ValueError, match="too few"):
        train_model(data_path=path, n_estimators=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"test_size": 0.0}, "test_size"),
        ({"test_size": 1.0}, "test_size"),
        ({"n_estimators": 0}, "n_estimators"),
        ({"window_seconds": 0}, "window_seconds"),
    ],
)
def test_train_model_validates_parameters(data_file, kwargs, message):
    with pytest.raises(ValueError, match=message):
        train_model(data_path=data_file, **kwargs)


# --------------------------------------------------------------------------- simulated data


def fake_simulated_dataset(calls):
    """Stand-in for simulated_dataset: synthetic separable arrays, records the calls."""

    def fake(n_samples, *, seed, attack_ratio, window_seconds, start_time=1_700_000_000.0):
        calls.append(
            {"n": n_samples, "seed": seed, "ratio": attack_ratio, "window": window_seconds}
        )
        rng = np.random.default_rng(seed)
        y = ["normal" if i % 3 else "syn_flood" for i in range(n_samples)]
        X = rng.normal(0, 0.3, size=(n_samples, N_FEATURES))
        X[:, 0] += [5.0 if label == "syn_flood" else 0.0 for label in y]
        return X, y

    return fake


@pytest.mark.parametrize(("n_samples", "test_n"), [(800, 800), (4000, 1000), (8000, 2000)])
def test_simulated_training_evaluates_on_a_separate_seed(monkeypatch, n_samples, test_n):
    calls = []
    monkeypatch.setattr(train, "simulated_dataset", fake_simulated_dataset(calls))
    result = train_model(
        n_samples=n_samples, seed=5, attack_ratio=0.4, window_seconds=4.0, n_estimators=4
    )
    assert calls == [
        {"n": n_samples, "seed": 5, "ratio": 0.4, "window": 4.0},
        {"n": test_n, "seed": 6, "ratio": 0.4, "window": 4.0},
    ]
    assert result.simulated is True
    assert (result.train_size, result.test_size) == (n_samples, test_n)
    assert result.model.window_seconds == 4.0
    training = result.model.metadata["training"]
    assert training["source"] == "simulated"
    assert (training["seed"], training["test_seed"]) == (5, 6)
    assert training["attack_ratio"] == 0.4
    assert training["n_samples"] == n_samples
    assert result.metrics["accuracy"] == 1.0


def test_simulated_dataset_drives_the_simulator(monkeypatch):
    created = []

    class FakeSimulator:
        def __init__(self, seed=None, attack_ratio=0.2, start_time=None, **kwargs):
            self.args = {"seed": seed, "attack_ratio": attack_ratio, "start_time": start_time}
            created.append(self)

        def records(self, count=None):
            self.count = count
            stream = labelled_stream(count, seed=self.args["seed"], start=self.args["start_time"])
            return iter(stream)

    monkeypatch.setattr(train, "TrafficSimulator", FakeSimulator)
    X, y = simulated_dataset(120, seed=3, attack_ratio=0.25, window_seconds=5.0)
    (simulator,) = created
    assert simulator.args == {"seed": 3, "attack_ratio": 0.25, "start_time": 1_700_000_000.0}
    assert simulator.count == 120
    assert X.shape == (120, N_FEATURES)
    expected_X, expected_y = build_dataset(labelled_stream(120, seed=3, start=1_700_000_000.0), 5.0)
    np.testing.assert_array_equal(X, expected_X)
    assert y == expected_y

    simulated_dataset(10, seed=1, attack_ratio=0.5, window_seconds=1.0, start_time=42.0)
    assert created[-1].args["start_time"] == 42.0


def test_simulated_dataset_rejects_empty_request():
    with pytest.raises(ValueError, match="n_samples"):
        simulated_dataset(0, seed=1, attack_ratio=0.2, window_seconds=10.0)


def test_simulated_training_end_to_end():
    """Uses the real TrafficSimulator (small sizes, so it stays fast)."""
    X, y = simulated_dataset(500, seed=11, attack_ratio=0.3, window_seconds=10.0)
    assert X.shape == (500, N_FEATURES)
    assert set(y) <= set(LABELS)
    again, _ = simulated_dataset(500, seed=11, attack_ratio=0.3, window_seconds=10.0)
    np.testing.assert_array_equal(X, again)

    result = train_model(n_samples=3000, seed=11, attack_ratio=0.3, n_estimators=10)
    assert result.simulated is True
    assert (result.train_size, result.test_size) == (3000, 1000)
    assert sum(result.class_counts.values()) == 3000
    assert "normal" in result.class_counts
    assert len(result.class_counts) >= 2
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    json.dumps(result.metrics)
    assert SIMULATED_CAVEAT in format_report(result)


# --------------------------------------------------------------------------- format_report


def test_format_report_sections(file_result, data_file):
    report = format_report(file_result)
    assert report.startswith("NeuralGuard model training report")
    assert str(data_file) in report
    assert "chronological" in report
    assert re.search(r"Training set:\s+300 records", report)
    assert re.search(r"Test set:\s+100 records", report)
    assert re.search(r"normal\s+[\d,]+\s+\d+\.\d%", report)  # class balance line
    for heading in (
        "Accuracy",
        "ROC-AUC",
        "Detection rate",
        "False-positive rate",
        "Per-class results",
        "precision",
        "recall",
        "Confusion matrix",
        "Top 10 feature importances",
    ):
        assert heading in report
    ranked = re.findall(r"^\s+(\d+)\. (\w+)\s+\d\.\d{4}$", report, flags=re.MULTILINE)
    assert [int(rank) for rank, _ in ranked] == list(range(1, 11))
    assert {name for _, name in ranked} <= set(FEATURE_NAMES)
    assert SIMULATED_CAVEAT not in report


def test_format_report_confusion_matrix_rows(file_result):
    report = format_report(file_result).splitlines()
    start = next(i for i, line in enumerate(report) if line.startswith("Confusion matrix"))
    header, *rows = report[start + 1 : start + 4]
    assert header.split() == file_result.metrics["labels"]
    for line, label, counts in zip(
        rows, file_result.metrics["labels"], file_result.metrics["confusion_matrix"], strict=True
    ):
        assert line.split() == [label, *(f"{count:,}" for count in counts)]


def test_format_report_caveat_for_simulated_and_missing_values(file_result):
    result = TrainingResult(
        model=file_result.model,
        metrics={"accuracy": 1.0, "roc_auc": None, "detection_rate": None},
        train_size=10,
        test_size=5,
        class_counts={"normal": 10},
        simulated=True,
    )
    report = format_report(result)
    assert report.count(SIMULATED_CAVEAT) == 1
    assert "simulated traffic" in report
    assert re.search(r"ROC-AUC \(threat\)\s+n/a", report)
    assert re.search(r"Detection rate\s+n/a", report)
    assert "Confusion matrix" not in report  # no matrix in these metrics


def test_format_report_shows_version_once_saved(file_result, tmp_path):
    assert "Model version" not in format_report(file_result) or file_result.model.version != (
        "unsaved"
    )
    model = ThreatModel.load(file_result.model.save(tmp_path / "v.joblib"))
    result = TrainingResult(model, file_result.metrics, 300, 100, file_result.class_counts, False)
    assert f"Model version:    {model.version}" in format_report(result)
