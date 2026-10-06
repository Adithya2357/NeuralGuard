import logging
import math

import numpy as np
import pytest

from neuralguard.detector import (
    SEVERITIES,
    Detection,
    Detector,
    DetectorStats,
    severity_for,
)
from neuralguard.features import FEATURE_NAMES, FeatureExtractor, features_as_dict
from neuralguard.model import ModelError, Prediction, ThreatModel
from neuralguard.schema import InvalidRecordError, normalize_record

DPORT = FEATURE_NAMES.index("destination_port")


class FakeModel:
    """Stands in for ThreatModel: threat_score = destination_port / 1000.

    So a record's destination port chooses its score (port 970 -> 0.97). Records sent to
    port 53 are reported as ``udp_flood``, everything else as ``port_scan``.
    """

    def __init__(self, window_seconds=7.5, version="fake0001"):
        self.window_seconds = window_seconds
        self.version = version
        self.predict_calls = []
        self.predict_one_calls = 0

    def predict_one(self, x):
        self.predict_one_calls += 1
        return self._prediction(np.asarray(x))

    def predict(self, X):
        X = np.asarray(X)
        self.predict_calls.append(X.copy())
        return [self._prediction(row) for row in X]

    @staticmethod
    def _prediction(row):
        score = min(1.0, float(row[DPORT]) / 1000.0)
        attack = "udp_flood" if row[DPORT] == 53 else "port_scan"
        return Prediction(
            threat_score=score,
            predicted_label=attack if score >= 0.5 else "normal",
            attack_type=attack,
            probabilities={"normal": 1.0 - score, attack: score},
        )


def raw(ts=0.0, src="10.0.0.1", dst="10.0.0.2", dport=80, flags="S", protocol="TCP", **extra):
    return {
        "timestamp": ts,
        "source_ip": src,
        "destination_ip": dst,
        "protocol": protocol,
        "source_port": 40000,
        "destination_port": dport,
        "length": 60,
        "ttl": 64,
        "tcp_flags": flags,
        **extra,
    }


# --- severity -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.0, "critical"),
        (0.9, "critical"),
        (0.8999, "high"),
        (0.75, "high"),
        (0.7499, "medium"),
        (0.6, "medium"),
        (0.5999, "low"),
        (0.0, "low"),
    ],
)
def test_severity_thresholds(score, expected):
    assert severity_for(score) == expected


def test_severities_are_ordered_low_to_critical():
    assert SEVERITIES == ("low", "medium", "high", "critical")


# --- construction ---------------------------------------------------------------------


def test_default_extractor_uses_model_window():
    model = FakeModel(window_seconds=3.0)
    detector = Detector(model)
    assert isinstance(detector.extractor, FeatureExtractor)
    assert detector.extractor.window_seconds == 3.0

    # Behaviourally: packets 4 s apart are outside a 3 s window (a 10 s default would
    # count both).
    detector.process(raw(ts=0.0))
    second = detector.process(raw(ts=4.0))
    assert features_as_dict(second.features)["src_packet_count"] == 1.0


def test_injected_extractor_is_used():
    extractor = FeatureExtractor(window_seconds=60.0)
    detector = Detector(FakeModel(window_seconds=3.0), extractor=extractor)
    assert detector.extractor is extractor
    detector.process(raw(ts=0.0))
    assert extractor.tracked_hosts == 2


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, math.nan])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        Detector(FakeModel(), threshold=threshold)


def test_threshold_one_is_allowed():
    assert Detector(FakeModel(), threshold=1.0).threshold == 1.0


# --- process --------------------------------------------------------------------------


def test_process_threat():
    model = FakeModel()
    detector = Detector(model, threshold=0.5)
    detection = detector.process(raw(dport=970, protocol="tcp", flags="s"))

    assert isinstance(detection, Detection)
    assert detection.is_threat
    assert detection.threat_score == pytest.approx(0.97)
    assert type(detection.threat_score) is float
    assert detection.attack_type == "port_scan"
    assert detection.severity == "critical"
    # The record is the canonical (normalised) form.
    assert detection.record["protocol"] == "TCP"
    assert detection.record["tcp_flags"] == "S"
    assert detection.features.shape == (len(FEATURE_NAMES),)
    assert detection.features[DPORT] == 970.0
    assert model.predict_one_calls == 1


def test_process_not_a_threat():
    detection = Detector(FakeModel()).process(raw(dport=443))
    assert not detection.is_threat
    assert detection.threat_score == pytest.approx(0.443)
    assert detection.attack_type is None
    assert detection.severity is None


def test_score_equal_to_threshold_is_a_threat():
    detection = Detector(FakeModel(), threshold=0.8).process(raw(dport=800))
    assert detection.is_threat
    assert detection.severity == "high"


def test_process_invalid_record_propagates_and_is_counted():
    detector = Detector(FakeModel())
    with pytest.raises(InvalidRecordError):
        detector.process(raw(source_ip="not-an-ip"))
    assert detector.stats == DetectorStats(processed=0, threats=0, invalid=1)
    assert detector.extractor.tracked_hosts == 0


def test_stats_track_threats_by_attack_type():
    detector = Detector(FakeModel(), threshold=0.5)
    for dport in (970, 980, 100, 600):
        detector.process(raw(dport=dport))
    # Port 53 maps to udp_flood but scores 0.053 - not a threat - so is not counted.
    detector.process(raw(dport=53, protocol="UDP", flags=""))
    assert detector.stats.processed == 5
    assert detector.stats.threats == 3
    assert detector.stats.invalid == 0
    assert detector.stats.by_attack_type == {"port_scan": 3}


def test_stats_count_several_attack_types():
    detector = Detector(FakeModel(), threshold=0.05)
    detector.process(raw(dport=53, protocol="UDP", flags=""))
    detector.process(raw(dport=900))
    assert detector.stats.by_attack_type == {"udp_flood": 1, "port_scan": 1}


# --- process_many ---------------------------------------------------------------------


def test_process_many_uses_one_batched_predict_call():
    model = FakeModel()
    detector = Detector(model)
    raws = [raw(ts=i * 0.1, dport=port) for i, port in enumerate((970, 100, 650, 800))]

    detections = detector.process_many(raws)

    assert len(model.predict_calls) == 1
    assert model.predict_calls[0].shape == (4, len(FEATURE_NAMES))
    assert model.predict_one_calls == 0
    assert [d.record["destination_port"] for d in detections] == [970, 100, 650, 800]
    assert [d.is_threat for d in detections] == [True, False, True, True]
    assert [d.severity for d in detections] == ["critical", None, "medium", "high"]
    assert detector.stats.processed == 4
    assert detector.stats.threats == 3


def test_process_many_matches_sequential_process():
    raws = [raw(ts=i * 0.01, dport=1 + i * 37, src=f"10.0.0.{i % 3 + 1}") for i in range(30)]
    one_by_one = Detector(FakeModel())
    batched = Detector(FakeModel())

    expected = [one_by_one.process(r) for r in raws]
    actual = batched.process_many(raws)

    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.features, e.features)
        assert a.record == e.record
        assert a.threat_score == e.threat_score
        assert (a.is_threat, a.attack_type, a.severity) == (e.is_threat, e.attack_type, e.severity)
    assert batched.stats == one_by_one.stats
    # Window features really are stateful: the last packet sees earlier ones.
    assert features_as_dict(actual[-1].features)["src_packet_count"] == 10.0


def test_process_many_skips_invalid_records(caplog):
    model = FakeModel()
    detector = Detector(model)
    raws = [
        raw(ts=0.0, dport=900),
        raw(ts=0.1, destination_port=70000),
        "not a dict",
        raw(ts=0.2, dport=100),
    ]
    with caplog.at_level(logging.WARNING, logger="neuralguard.detector"):
        detections = detector.process_many(raws)

    assert [d.record["destination_port"] for d in detections] == [900, 100]
    assert detector.stats.invalid == 2
    assert detector.stats.processed == 2
    assert model.predict_calls[0].shape[0] == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "destination_port must be between 0 and 65535" in warnings[0].getMessage()
    assert "record must be a JSON object" in warnings[1].getMessage()


def test_process_many_truncates_long_reasons(caplog):
    detector = Detector(FakeModel())
    with caplog.at_level(logging.WARNING, logger="neuralguard.detector"):
        detector.process_many([raw(source_ip="x" * 5000)])
    message = caplog.records[0].getMessage()
    assert len(message) < 400
    assert message.endswith("...")


def test_process_many_empty_input_does_not_call_model():
    model = FakeModel()
    detector = Detector(model)
    assert detector.process_many([]) == []
    assert detector.process_many(iter(())) == []
    assert model.predict_calls == []
    assert detector.stats == DetectorStats()


def test_process_many_all_invalid_does_not_call_model():
    model = FakeModel()
    detector = Detector(model)
    assert detector.process_many([{"timestamp": "yesterday"}, 42]) == []
    assert model.predict_calls == []
    assert detector.stats.invalid == 2


def test_process_many_accepts_a_generator():
    detector = Detector(FakeModel())
    detections = detector.process_many(raw(ts=i, dport=950) for i in range(3))
    assert len(detections) == 3
    assert all(d.is_threat for d in detections)


def test_process_many_rejects_wrong_prediction_count():
    class ShortModel(FakeModel):
        def predict(self, X):
            return super().predict(X)[:-1]

    with pytest.raises(ModelError, match="2 predictions for 3 records"):
        Detector(ShortModel()).process_many([raw(ts=i) for i in range(3)])


# --- with the real model --------------------------------------------------------------


def _scan(start, src, count):
    return [
        normalize_record(raw(ts=start + i * 0.01, src=src, dst="192.168.1.20", dport=1 + i))
        for i in range(count)
    ]


def _normal(start, count):
    return [
        normalize_record(
            raw(
                ts=start + i * 0.5,
                src=f"192.168.1.{10 + i % 5}",
                dst="93.184.216.34",
                dport=443,
                flags="PA",
            )
        )
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def real_model():
    records = _normal(0.0, 120) + _scan(100.0, "203.0.113.5", 120)
    labels = ["normal"] * 120 + ["port_scan"] * 120
    X = FeatureExtractor(window_seconds=5.0).transform(records)
    return ThreatModel.train(
        X, labels, window_seconds=5.0, n_estimators=10, random_state=0, n_jobs=1
    )


def test_detector_with_real_threat_model(real_model):
    detector = Detector(real_model, threshold=0.5)
    assert detector.extractor.window_seconds == 5.0
    records = _normal(1000.0, 20) + _scan(1100.0, "198.51.100.7", 60)

    detections = detector.process_many(records)

    expected = real_model.predict(FeatureExtractor(window_seconds=5.0).transform(records))
    assert [d.threat_score for d in detections] == [p.threat_score for p in expected]
    assert not any(d.is_threat for d in detections[:20])
    assert all(d.is_threat and d.attack_type == "port_scan" for d in detections[-30:])
    assert detector.stats.by_attack_type["port_scan"] == detector.stats.threats
