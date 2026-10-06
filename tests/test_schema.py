import math

import pytest

from neuralguard.schema import MAX_TIMESTAMP, InvalidRecordError, normalize_record


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


@pytest.mark.parametrize(
    "value",
    [
        "fe80::1%eth0",
        "fe80::bad%\n2026-10-06 12:00:00,000 INFO    neuralguard.consumer: stats\x1b]0;x\x07",
        "fe80::1%" + "A" * 100_000,
    ],
    ids=["interface", "log-injection", "100-kB"],
)
def test_ipv6_zone_ids_are_rejected(value):
    # No captured packet carries one, but a zone id may contain anything: forged log
    # lines, terminal escapes, or megabytes that every window would keep as a host key.
    for field in ("source_ip", "destination_ip"):
        with pytest.raises(InvalidRecordError, match="zone id") as excinfo:
            normalize_record({field: value})
        assert len(str(excinfo.value)) < 200


@pytest.mark.parametrize(
    "value",
    [
        "10.0.0.1",
        " 192.168.1.10 ",
        "2001:0db8:0000:0000:0000:0000:0000:0001",
        "FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF",
        "::ffff:255.255.255.255",
        "fe80::1",
    ],
)
def test_accepted_ip_addresses_are_short_and_canonical(value):
    canonical = normalize_record({"source_ip": value})["source_ip"]
    assert len(canonical) <= 45 and "%" not in canonical
    assert canonical == canonical.strip().lower()


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


@pytest.mark.parametrize("timestamp", [1e20, 1e300, MAX_TIMESTAMP + 1, 1_700_000_000_000])
def test_implausible_timestamps_are_rejected(timestamp):
    # Past the year 3000 a timestamp is corrupt or mis-scaled; it would otherwise jump the
    # feature extractor's forward-only clock far into the future for good.
    with pytest.raises(InvalidRecordError, match="not a plausible capture time"):
        normalize_record({"timestamp": timestamp})


@pytest.mark.parametrize(
    "timestamp", [10**400, -(10**400), 10**309], ids=["1e400", "-1e400", "1e309"]
)
def test_integers_too_large_for_a_float_are_invalid_records(timestamp):
    # json.loads turns a 400-digit number into a Python int; float() of it overflows.
    with pytest.raises(InvalidRecordError, match="not a plausible capture time") as excinfo:
        normalize_record({"timestamp": timestamp})
    assert "0000000000" not in str(excinfo.value)  # the huge value is not echoed


def test_millisecond_timestamps_get_a_hint():
    with pytest.raises(InvalidRecordError, match="milliseconds instead of seconds"):
        normalize_record({"timestamp": 1_700_000_000_123})
    with pytest.raises(InvalidRecordError) as excinfo:
        normalize_record({"timestamp": 1e20})
    assert "milliseconds" not in str(excinfo.value)


def test_timestamp_bounds_are_inclusive():
    assert normalize_record({"timestamp": 0})["timestamp"] == 0.0
    assert normalize_record({"timestamp": MAX_TIMESTAMP})["timestamp"] == MAX_TIMESTAMP
