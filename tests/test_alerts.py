import dataclasses
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from neuralguard import alerts
from neuralguard.alerts import AlertThrottler, build_alert_document
from neuralguard.detector import Detection
from neuralguard.features import FEATURE_NAMES, FeatureExtractor
from neuralguard.schema import ATTACK_TYPES, normalize_record
from neuralguard.sinks import INDEX_MAPPINGS

PACKET_TIME = 1730647800.123  # 2024-11-03T15:30:00.123Z
NOW = 1730647805.5  # 2024-11-03T15:30:05.500Z

PLAIN_TYPES = (str, int, float, bool, type(None))


def detection(
    *,
    ts=PACKET_TIME,
    dst="192.168.1.20",
    attack_type="port_scan",
    score=0.97,
    severity="critical",
    is_threat=True,
    **fields,
):
    record = normalize_record(
        {
            "timestamp": ts,
            "source_ip": "203.0.113.5",
            "destination_ip": dst,
            "protocol": "TCP",
            "source_port": 40000,
            "destination_port": 22,
            "length": 60,
            "ttl": 64,
            "tcp_flags": "S",
            **fields,
        }
    )
    features = FeatureExtractor().transform_one(record)
    return Detection(
        record=record,
        features=features,
        threat_score=score,
        is_threat=is_threat,
        attack_type=attack_type if is_threat else None,
        severity=severity if is_threat else None,
    )


def walk(value):
    """Every leaf value and every key of a nested document."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk(item)
    else:
        yield value


def parse_utc(text):
    # datetime.fromisoformat only accepts a "Z" suffix from Python 3.11 on.
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


# --- build_alert_document -------------------------------------------------------------


def test_alert_document_fields():
    doc = build_alert_document(
        detection(), model_version="3f2a9c1b7d4e", suppressed_count=12, now=NOW
    )
    features = doc.pop("features")
    assert doc == {
        "@timestamp": "2024-11-03T15:30:00.123Z",
        "detected_at": "2024-11-03T15:30:05.500Z",
        "source_ip": "203.0.113.5",
        "destination_ip": "192.168.1.20",
        "protocol": "TCP",
        "source_port": 40000,
        "destination_port": 22,
        "length": 60,
        "ttl": 64,
        "tcp_flags": "S",
        "threat_score": 0.97,
        "attack_type": "port_scan",
        "severity": "critical",
        "suppressed_count": 12,
        "model_version": "3f2a9c1b7d4e",
    }
    assert list(features) == list(FEATURE_NAMES)
    assert features["is_tcp"] == 1.0
    assert features["destination_port"] == 22.0


def test_timestamps_are_utc_iso8601():
    doc = build_alert_document(detection(), model_version="v", now=NOW)
    packet_time = parse_utc(doc["@timestamp"])
    detected_at = parse_utc(doc["detected_at"])
    assert packet_time.utcoffset().total_seconds() == 0
    assert packet_time.timestamp() == pytest.approx(PACKET_TIME, abs=1e-3)
    assert detected_at.timestamp() == pytest.approx(NOW, abs=1e-3)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")
def test_timestamps_do_not_depend_on_local_timezone(monkeypatch):
    expected = build_alert_document(detection(), model_version="v", now=NOW)
    try:
        monkeypatch.setenv("TZ", "America/New_York")
        time.tzset()
        doc = build_alert_document(detection(), model_version="v", now=NOW)
    finally:
        monkeypatch.undo()
        time.tzset()
    assert doc["@timestamp"] == expected["@timestamp"] == "2024-11-03T15:30:00.123Z"
    assert doc["detected_at"] == expected["detected_at"]


def test_detected_at_defaults_to_current_time():
    before = time.time()
    doc = build_alert_document(detection(), model_version="v")
    after = time.time()
    assert before - 0.001 <= parse_utc(doc["detected_at"]).timestamp() <= after + 0.001


def test_unrepresentable_packet_time_falls_back_to_detection_time():
    # normalize_record rejects such times, but a Detection assembled some other way must
    # still produce a document instead of crashing the detector loop.
    det = detection()
    det = dataclasses.replace(det, record={**det.record, "timestamp": 1e20})
    doc = build_alert_document(det, model_version="v", now=NOW)
    assert doc["@timestamp"] == doc["detected_at"] == "2024-11-03T15:30:05.500Z"


def test_document_contains_only_plain_python_types():
    det = detection()
    numpy_det = Detection(
        record={
            **det.record,
            "timestamp": np.float64(PACKET_TIME),
            "source_port": np.int64(40000),
            "length": np.int32(60),
            "ttl": np.uint8(64),
        },
        features=det.features.astype(np.float32),
        threat_score=np.float32(0.97123456),
        is_threat=True,
        attack_type="port_scan",
        severity="critical",
    )
    doc = build_alert_document(
        numpy_det, model_version="v", suppressed_count=np.int64(3), now=np.float64(NOW)
    )

    for value in walk(doc):
        assert type(value) in PLAIN_TYPES, (value, type(value))
        assert not isinstance(value, np.generic)
    assert type(doc["threat_score"]) is float
    assert doc["threat_score"] == 0.9712
    assert type(doc["source_port"]) is int
    assert type(doc["suppressed_count"]) is int
    assert all(type(v) is float for v in doc["features"].values())
    json.loads(json.dumps(doc, allow_nan=False))


def test_threat_score_rounded_to_four_places():
    doc = build_alert_document(detection(score=0.876543), model_version="v", now=NOW)
    assert doc["threat_score"] == 0.8765


def test_simulated_label_only_when_record_has_label():
    without = build_alert_document(detection(), model_version="v", now=NOW)
    labelled = build_alert_document(detection(label="port_scan"), model_version="v", now=NOW)
    assert "simulated_label" not in without
    assert labelled["simulated_label"] == "port_scan"


def test_unknown_ips_are_null():
    doc = build_alert_document(detection(source_ip=None), model_version="v", now=NOW)
    assert doc["source_ip"] is None
    json.dumps(doc)


def test_every_document_field_has_an_index_mapping():
    doc = build_alert_document(detection(label="normal"), model_version="v", now=NOW)
    assert set(doc) == set(INDEX_MAPPINGS["properties"])


def test_rejects_non_threat():
    with pytest.raises(ValueError, match="not a threat"):
        build_alert_document(detection(is_threat=False), model_version="v")


def test_rejects_negative_suppressed_count():
    with pytest.raises(ValueError, match="suppressed_count"):
        build_alert_document(detection(), model_version="v", suppressed_count=-1)


# --- AlertThrottler -------------------------------------------------------------------


def test_first_alert_emitted_then_suppressed_within_cooldown():
    throttler = AlertThrottler(cooldown_seconds=5.0)
    assert throttler.check(detection(ts=100.0)) == 0
    assert throttler.check(detection(ts=101.0)) is None
    assert throttler.check(detection(ts=104.9)) is None
    assert throttler.suppressed_total == 2


def test_next_alert_reports_suppressed_count():
    throttler = AlertThrottler(cooldown_seconds=5.0)
    assert throttler.check(detection(ts=100.0)) == 0
    for i in range(12):
        assert throttler.check(detection(ts=100.1 + i * 0.1)) is None
    assert throttler.check(detection(ts=105.0)) == 12  # exactly one cooldown later
    assert throttler.check(detection(ts=106.0)) is None
    assert throttler.check(detection(ts=110.0)) == 1
    assert throttler.check(detection(ts=200.0)) == 0
    assert throttler.suppressed_total == 13


def test_keys_are_attack_type_and_destination():
    throttler = AlertThrottler(cooldown_seconds=60.0)
    assert throttler.check(detection(ts=0.0, dst="10.0.0.1")) == 0
    assert throttler.check(detection(ts=0.1, dst="10.0.0.2")) == 0
    assert throttler.check(detection(ts=0.2, dst="10.0.0.1", attack_type="syn_flood")) == 0
    # Same attack type and target, different source: still the same alert.
    assert throttler.check(detection(ts=0.3, dst="10.0.0.1", source_ip="198.51.100.9")) is None
    assert throttler.tracked_keys == 3


def test_throttling_uses_packet_time_not_wall_clock():
    throttler = AlertThrottler(cooldown_seconds=5.0)
    # Replayed instantly, but 10 s apart in packet time: every one is emitted.
    results = [throttler.check(detection(ts=1000.0 + 10 * i)) for i in range(5)]
    assert results == [0, 0, 0, 0, 0]


def test_out_of_order_packet_is_suppressed():
    throttler = AlertThrottler(cooldown_seconds=5.0)
    assert throttler.check(detection(ts=100.0)) == 0
    assert throttler.check(detection(ts=90.0)) is None


def test_zero_cooldown_disables_throttling():
    throttler = AlertThrottler(cooldown_seconds=0)
    assert [throttler.check(detection(ts=100.0)) for _ in range(5)] == [0] * 5
    assert throttler.suppressed_total == 0
    assert throttler.tracked_keys == 0


def test_non_threat_is_never_alerted_nor_counted():
    throttler = AlertThrottler(cooldown_seconds=5.0)
    assert throttler.check(detection(is_threat=False)) is None
    assert AlertThrottler(cooldown_seconds=0).check(detection(is_threat=False)) is None
    assert throttler.suppressed_total == 0
    assert throttler.check(detection()) == 0


def test_flush_reports_the_held_back_tail_once_the_cooldown_is_over():
    throttler = AlertThrottler(cooldown_seconds=5.0)
    assert throttler.check(detection(ts=100.0)) == 0
    tail = [detection(ts=100.0 + i, score=0.5 + i / 10) for i in (1, 2, 3)]
    assert all(throttler.check(d) is None for d in tail)
    assert throttler.flush(now=104.9) == []  # the flood may still go on
    ((latest, suppressed),) = throttler.flush(now=105.0)
    assert latest is tail[-1] and suppressed == 2  # the latest one, plus 2 before it
    assert throttler.flush(now=106.0) == []  # reported once
    # The summary counts as an alert at its own packet time (103).
    assert throttler.check(detection(ts=107.0)) is None
    assert throttler.check(detection(ts=108.0)) == 1


def test_flush_everything_at_shutdown_and_forget_quiet_keys():
    throttler = AlertThrottler(cooldown_seconds=60.0)
    assert throttler.check(detection(ts=0.0, dst="10.0.0.1")) == 0
    assert throttler.check(detection(ts=1.0, dst="10.0.0.1")) is None
    assert throttler.check(detection(ts=2.0, dst="10.0.0.2")) == 0
    assert throttler.flush(now=3.0) == []
    summaries = throttler.flush(everything=True)
    assert [(d.record["destination_ip"], n) for d, n in summaries] == [("10.0.0.1", 0)]
    assert throttler.flush(now=1000.0) == []
    assert throttler.tracked_keys == 0  # expired, nothing held back: nothing to remember


def test_flush_accounts_for_every_threat_exactly_once():
    import random

    rng = random.Random(3)
    throttler = AlertThrottler(cooldown_seconds=5.0)
    threats = accounted = alerts = 0
    ts = 0.0
    for _ in range(3000):
        ts += rng.expovariate(20.0)
        det = detection(ts=ts, dst=f"10.0.0.{rng.randint(1, 6)}", attack_type="syn_flood")
        threats += 1
        results = [throttler.check(det)]
        if rng.random() < 0.05:  # the service flushes after every batch
            results += [n for _, n in throttler.flush(now=ts)]
        for result in results:
            if result is not None:
                alerts += 1
                accounted += 1 + result
    for _, n in throttler.flush(everything=True):
        alerts += 1
        accounted += 1 + n
    assert accounted == threats
    assert alerts + throttler.suppressed_total == threats
    assert throttler.suppressed_total > 0


def test_flush_finds_expired_keys_wherever_they_sit():
    throttler = AlertThrottler(cooldown_seconds=5.0)
    assert throttler.check(detection(ts=0.0, dst="10.0.0.1")) == 0
    assert throttler.check(detection(ts=0.5, dst="10.0.0.1")) is None
    assert throttler.check(detection(ts=1.0, dst="10.0.0.2")) == 0
    (summary,) = throttler.flush(now=5.2)  # 10.0.0.1 summarised at 0.5, moved behind .2
    assert summary[0].record["timestamp"] == 0.5
    assert throttler.check(detection(ts=5.3, dst="10.0.0.1")) is None  # within 0.5 + 5
    # 10.0.0.1 is now after 10.0.0.2 in the table, but expires first.
    ((latest, suppressed),) = throttler.flush(now=5.8)
    assert latest.record["timestamp"] == 5.3 and suppressed == 0


def test_flush_with_throttling_disabled():
    throttler = AlertThrottler(cooldown_seconds=0)
    throttler.check(detection())
    assert throttler.flush(now=1e9) == [] and throttler.flush(everything=True) == []


def test_max_keys_drops_oldest_when_none_expired():
    throttler = AlertThrottler(cooldown_seconds=5.0, max_keys=3)
    for i in range(4):
        assert throttler.check(detection(ts=float(i), dst=f"10.0.0.{i + 1}")) == 0
    assert throttler.tracked_keys == 3
    # 10.0.0.1 (the oldest) was forgotten, so it alerts again; 10.0.0.2 is still throttled.
    assert throttler.check(detection(ts=3.5, dst="10.0.0.2")) is None
    assert throttler.check(detection(ts=3.5, dst="10.0.0.1")) == 0
    assert throttler.tracked_keys == 3


def test_max_keys_drops_expired_keys_first():
    throttler = AlertThrottler(cooldown_seconds=5.0, max_keys=3)
    assert throttler.check(detection(ts=0.0, dst="10.0.0.1")) == 0
    assert throttler.check(detection(ts=1.0, dst="10.0.0.2")) == 0
    assert throttler.check(detection(ts=6.0, dst="10.0.0.3")) == 0
    assert throttler.check(detection(ts=6.5, dst="10.0.0.4")) == 0
    # Both expired keys (t=0 and t=1, cooldown over at t=6.5) went; the active ones stay.
    assert throttler.tracked_keys == 2
    assert throttler.check(detection(ts=7.0, dst="10.0.0.3")) is None
    assert throttler.check(detection(ts=7.0, dst="10.0.0.4")) is None


def test_max_keys_bounds_memory_under_scan_of_many_targets():
    throttler = AlertThrottler(cooldown_seconds=30.0, max_keys=100)
    for i in range(5000):
        throttler.check(detection(ts=i * 0.001, dst=f"10.{i // 65536}.{i // 256 % 256}.{i % 256}"))
    assert throttler.tracked_keys == 100


@pytest.mark.parametrize("cooldown", [-1.0, math.nan, math.inf])
def test_invalid_cooldown(cooldown):
    with pytest.raises(ValueError, match="cooldown"):
        AlertThrottler(cooldown)


def test_invalid_max_keys():
    with pytest.raises(ValueError, match="max_keys"):
        AlertThrottler(5.0, max_keys=0)


def test_throttle_key():
    for attack_type in ATTACK_TYPES:
        key = alerts.throttle_key(detection(dst="10.9.8.7", attack_type=attack_type))
        assert key == (attack_type, "10.9.8.7")


def test_iso_utc_whole_second_has_milliseconds():
    expected = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert alerts._iso_utc(expected.timestamp()) == "2024-01-02T03:04:05.000Z"
