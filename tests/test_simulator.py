"""Tests for the traffic simulator (seeded, so every run sees the same streams)."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import pytest

from neuralguard.schema import ATTACK_TYPES, LABELS, NORMAL_LABEL, RECORD_FIELDS, normalize_record
from neuralguard.simulator import TrafficSimulator, generate_records

START = 1_700_000_000.0


@pytest.fixture(scope="module")
def stream_60k():
    """The stream ``neuralguard train`` uses by default: seed 42, attack ratio 0.3."""
    return generate_records(60_000, seed=42, attack_ratio=0.3, start_time=START)


def labels(records):
    return Counter(record["label"] for record in records)


def attack_fraction(records):
    return 1 - labels(records)[NORMAL_LABEL] / len(records)


# ------------------------------------------------------------------ stream properties


def test_same_seed_same_stream():
    first = generate_records(3000, seed=7, attack_ratio=0.3, start_time=START)
    second = generate_records(3000, seed=7, attack_ratio=0.3, start_time=START)
    assert first == second


def test_different_seeds_differ():
    first = generate_records(500, seed=1, start_time=START)
    second = generate_records(500, seed=2, start_time=START)
    assert first != second


def test_records_are_time_ordered_and_start_at_start_time(stream_60k):
    timestamps = [record["timestamp"] for record in stream_60k]
    assert all(a <= b for a, b in itertools.pairwise(timestamps))
    assert timestamps[0] >= START
    assert timestamps[-1] - START < 3600  # minutes of traffic, not days


def test_records_are_canonical_and_labelled(stream_60k):
    for record in stream_60k:
        assert normalize_record(record) == record
        assert set(record) == {*RECORD_FIELDS, "label"}
        assert record["label"] in LABELS


def test_every_attack_type_appears_at_the_default_training_ratio(stream_60k):
    counts = labels(stream_60k)
    for attack in ATTACK_TYPES:
        assert counts[attack] > 0, attack
    assert 0.2 < attack_fraction(stream_60k) < 0.4


def test_normal_traffic_is_varied(stream_60k):
    normal = [record for record in stream_60k if record["label"] == NORMAL_LABEL]
    protocols = Counter(record["protocol"] for record in normal)
    assert {"TCP", "UDP", "ICMP", "ARP"} <= set(protocols)
    # Legitimate traffic has pure SYNs and IPv6 too, so neither gives an attack away.
    assert any(record["tcp_flags"] == "S" for record in normal)
    assert any(":" in (record["source_ip"] or "") for record in normal)


def test_attack_ratio_zero_has_no_attacks():
    records = generate_records(5000, seed=3, attack_ratio=0.0, start_time=START)
    assert set(labels(records)) == {NORMAL_LABEL}


def test_attack_ratio_one_has_only_attacks():
    records = generate_records(5000, seed=3, attack_ratio=1.0, start_time=START)
    counts = labels(records)
    assert NORMAL_LABEL not in counts
    assert set(counts) <= set(ATTACK_TYPES)
    assert all(normalize_record(record) == record for record in records)


@pytest.mark.parametrize("ratio", [0.5, 0.6, 0.8, 0.9])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_high_attack_ratios_are_reached(ratio, seed):
    # Above ~0.4 normal traffic is thinned, harder while the attack share lags behind
    # (slow scans can hold every attack slot for a long time).
    records = generate_records(20_000, seed=seed, attack_ratio=ratio, start_time=START)
    assert ratio - 0.08 < attack_fraction(records) < ratio + 0.05


def test_attack_types_subset():
    simulator = TrafficSimulator(
        seed=5, attack_ratio=0.5, start_time=START, attack_types=["udp_flood", "port_scan"]
    )
    counts = labels(simulator.records(10_000))
    assert set(counts) == {NORMAL_LABEL, "udp_flood", "port_scan"}


def test_attack_types_accepts_a_single_name_and_deduplicates():
    simulator = TrafficSimulator(seed=5, attack_types="icmp_flood", start_time=START)
    assert simulator.attack_types == ("icmp_flood",)
    simulator = TrafficSimulator(seed=5, attack_types=["syn_flood", "syn_flood"])
    assert simulator.attack_types == ("syn_flood",)


def test_empty_attack_types_are_fine_without_attacks():
    records = list(TrafficSimulator(seed=1, attack_ratio=0, attack_types=()).records(500))
    assert set(labels(records)) == {NORMAL_LABEL}


def test_default_start_time_is_now():
    import time

    before = time.time()
    (record,) = TrafficSimulator(seed=1).records(1)
    assert before - 1 <= record["timestamp"] <= time.time() + 5


# ------------------------------------------------------------------ records() / API


def test_count_none_is_an_infinite_stream():
    simulator = TrafficSimulator(seed=11, start_time=START)
    records = list(itertools.islice(simulator.records(None), 4000))
    assert len(records) == 4000
    assert records == generate_records(4000, seed=11, start_time=START)


def test_records_calls_continue_one_stream():
    simulator = TrafficSimulator(seed=9, start_time=START)
    first = list(simulator.records(100))
    second = list(simulator.records(100))
    assert first + second == generate_records(200, seed=9, start_time=START)


def test_records_zero():
    assert list(TrafficSimulator(seed=1, start_time=START).records(0)) == []


def test_generate_records_returns_a_list_of_count_records():
    records = generate_records(250, seed=4, attack_ratio=0.2, start_time=START)
    assert isinstance(records, list) and len(records) == 250


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize("ratio", [-0.1, 1.5, math.nan, "0.3", True, None])
def test_invalid_attack_ratio(ratio):
    with pytest.raises(ValueError, match="attack_ratio"):
        TrafficSimulator(seed=1, attack_ratio=ratio)


def test_unknown_attack_type():
    with pytest.raises(ValueError, match="unknown attack types"):
        TrafficSimulator(seed=1, attack_types=["port_scan", "teardrop"])


def test_empty_attack_types_with_attacks():
    with pytest.raises(ValueError, match="attack_types"):
        TrafficSimulator(seed=1, attack_ratio=0.3, attack_types=[])


@pytest.mark.parametrize("start_time", [-1.0, math.inf, math.nan, "now", True, 1e20])
def test_invalid_start_time(start_time):
    with pytest.raises(ValueError, match="start_time"):
        TrafficSimulator(seed=1, start_time=start_time)


@pytest.mark.parametrize("count", [-1, 2.5, True, "10"])
def test_invalid_count(count):
    with pytest.raises(ValueError, match="count"):
        TrafficSimulator(seed=1, start_time=START).records(count)
