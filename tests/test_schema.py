import math

import pytest

from neuralguard.schema import InvalidRecordError, normalize_record


def test_full_record_round_trips():
    raw = {
        "timestamp": 1_700_000_000.5,
        "source_ip": "192.168.1.10",
        "destination_ip": "10.0.0.3",
        "protocol": "tcp",
        "source_port": 51514,
        "destination_port": 443,
        "length": 60,
        "ttl": 64,
        "tcp_flags": "as",
        "label": "normal",
    }
    record = normalize_record(raw)
    assert record["protocol"] == "TCP"
    assert record["tcp_flags"] == "SA"
    assert record["label"] == "normal"
    assert normalize_record(record) == record


def test_defaults_for_missing_fields():
    record = normalize_record({"protocol": "UDP"})
    assert record["source_ip"] is None
    assert record["destination_port"] == 0
    assert record["tcp_flags"] == ""
    assert math.isfinite(record["timestamp"])
    assert "label" not in record


@pytest.mark.parametrize("value", ["Unknown", "", "none", None])
def test_unknown_ips_become_none(value):
    assert normalize_record({"source_ip": value})["source_ip"] is None


def test_ipv6_is_accepted_and_compressed():
    record = normalize_record({"source_ip": "2001:0db8:0000:0000:0000:0000:0000:0001"})
    assert record["source_ip"] == "2001:db8::1"


def test_unknown_protocol_maps_to_other_and_drops_flags():
    record = normalize_record({"protocol": "Ether / IP / TCP", "tcp_flags": "S"})
    assert record["protocol"] == "OTHER"
    assert record["tcp_flags"] == ""


@pytest.mark.parametrize(
    "raw",
    [
        {"source_ip": "999.1.1.1"},
        {"source_ip": 12345},
        {"destination_port": 70000},
        {"destination_port": -1},
        {"length": "big"},
        {"length": 1.5},
        {"ttl": True},
        {"timestamp": float("nan")},
        {"timestamp": "2024-01-01"},
        {"protocol": "TCP", "tcp_flags": "SXZ"},
        {"label": "evil"},
    ],
)
def test_invalid_records_raise(raw):
    with pytest.raises(InvalidRecordError):
        normalize_record(raw)


def test_non_mapping_raises():
    with pytest.raises(InvalidRecordError):
        normalize_record(["not", "a", "dict"])


def test_integral_floats_are_accepted():
    assert normalize_record({"length": 60.0})["length"] == 60
