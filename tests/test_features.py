import numpy as np
import pytest

from neuralguard.features import (
    DISCONTINUITY_RECORDS,
    FEATURE_NAMES,
    FeatureExtractor,
    features_as_dict,
    is_pure_syn,
    packet_features,
)
from neuralguard.schema import normalize_record


def pkt(ts, src="10.0.0.1", dst="10.0.0.2", dport=80, flags="S", protocol="TCP", **extra):
    return normalize_record(
        {
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
    )


def named(extractor, record):
    return features_as_dict(extractor.transform_one(record))


def test_feature_vector_shape_and_names():
    vector = FeatureExtractor().transform_one(pkt(0.0))
    assert vector.shape == (len(FEATURE_NAMES),)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_packet_features_flags_and_protocol():
    values = dict(zip(FEATURE_NAMES, packet_features(pkt(0.0, flags="FPU")), strict=False))
    assert values["is_tcp"] == 1.0
    assert values["flag_fin"] == values["flag_psh"] == values["flag_urg"] == 1.0
    assert values["flag_syn"] == 0.0
    assert values["dst_port_well_known"] == 1.0


def test_pure_syn():
    assert is_pure_syn(pkt(0, flags="S"))
    assert not is_pure_syn(pkt(0, flags="SA"))
    assert not is_pure_syn(pkt(0, protocol="UDP", flags=""))


def test_port_scan_raises_unique_port_count():
    extractor = FeatureExtractor(window_seconds=10)
    for i in range(50):
        features = named(extractor, pkt(i * 0.01, dport=1 + i))
    assert features["src_packet_count"] == 50
    assert features["src_unique_dst_ports"] == 50
    assert features["src_unique_dst_ips"] == 1
    assert features["src_syn_ratio"] == 1.0


def test_spoofed_flood_raises_unique_sources_on_destination():
    extractor = FeatureExtractor()
    for i in range(30):
        features = named(extractor, pkt(i * 0.001, src=f"203.0.113.{i}"))
    assert features["dst_packet_count"] == 30
    assert features["dst_unique_src_ips"] == 30
    assert features["src_packet_count"] == 1


def test_ports_of_the_packets_destination_are_counted_per_source():
    # One host hammering many ports of one target (a single-source UDP flood) is unlike
    # a fan-out to many hosts, which spreads its ports over the destinations.
    extractor = FeatureExtractor(window_seconds=1.0)
    for i in range(40):
        flood = named(extractor, pkt(i * 0.01, dport=1000 + i, protocol="UDP", flags=""))
    assert flood["pair_unique_dst_ports"] == 40
    other = named(extractor, pkt(0.5, dst="10.0.0.3", dport=53, protocol="UDP", flags=""))
    assert other["pair_unique_dst_ports"] == 1
    assert other["src_unique_dst_ports"] == 41
    later = named(extractor, pkt(2.0, dport=1000, protocol="UDP", flags=""))
    assert later["pair_unique_dst_ports"] == 1  # the window has moved on


def test_window_eviction():
    extractor = FeatureExtractor(window_seconds=1.0)
    for i in range(5):
        named(extractor, pkt(i * 0.1, dport=1000 + i))
    features = named(extractor, pkt(5.0, dport=22, flags="SA"))
    assert features["src_packet_count"] == 1
    assert features["src_unique_dst_ports"] == 1
    assert features["src_syn_ratio"] == 0.0


def test_out_of_order_records_do_not_break_eviction():
    extractor = FeatureExtractor(window_seconds=1.0)
    named(extractor, pkt(10.0))
    named(extractor, pkt(5.0))  # late: treated as t=10
    features = named(extractor, pkt(10.5))
    assert features["src_packet_count"] == 3


def stream(n=400, start=1_700_000_000.0, step=0.3, seed=0):
    """Synthetic time-ordered traffic between a few hosts, ``n * step`` seconds long."""
    import random

    rng = random.Random(seed)
    return [
        pkt(
            start + i * step,
            src=f"10.0.0.{rng.randint(1, 5)}",
            dst=f"10.0.1.{rng.randint(1, 3)}",
            dport=rng.choice((22, 80, 443, rng.randint(1, 65535))),
            flags=rng.choice(("S", "SA", "A", "PA")),
        )
        for i in range(n)
    ]


def retained_events(extractor):
    tables = (extractor._sources, extractor._destinations)
    return sum(len(window.events) for table in tables for window in table.values())


def test_a_stray_old_record_is_treated_as_out_of_order():
    extractor = FeatureExtractor(window_seconds=10.0)
    t0 = 1_700_000_000.0
    for i in range(3):
        named(extractor, pkt(t0 + i))
    named(extractor, pkt(t0 - 3600, src="10.9.9.9"))  # one record an hour old
    assert named(extractor, pkt(t0 + 3))["src_packet_count"] == 4  # the windows were kept


def test_windows_recover_from_a_record_dated_far_in_the_future():
    records = stream()  # 120 s
    bad = 100
    outlier = {**records[bad], "timestamp": records[bad]["timestamp"] + 86_400}
    poisoned = FeatureExtractor()
    X = poisoned.transform([*records[:bad], outlier, *records[bad:]])[bad + 1 :]
    expected = FeatureExtractor().transform(records[bad:])  # never saw the bad record
    # The windows restart DISCONTINUITY_RECORDS records later; one window after that the
    # features are exactly those of an extractor that never saw the bad record.
    restart = records[bad + DISCONTINUITY_RECORDS - 1]["timestamp"]
    settled = [i for i, record in enumerate(records[bad:]) if record["timestamp"] > restart + 10.0]
    assert len(settled) > 200
    np.testing.assert_array_equal(X[settled], expected[settled])
    fresh = FeatureExtractor()
    fresh.transform(records[bad:])
    assert retained_events(poisoned) == retained_events(fresh)  # bounded again


def test_replaying_a_capture_gives_the_same_features_again():
    records = stream()
    extractor = FeatureExtractor()
    first = extractor.transform(records)
    second = extractor.transform(records)  # the same 120 s again: time goes back
    restart = records[DISCONTINUITY_RECORDS - 1]["timestamp"]
    settled = [i for i, record in enumerate(records) if record["timestamp"] > restart + 10.0]
    assert len(settled) > 300
    np.testing.assert_array_equal(second[settled], first[settled])


def test_hosts_whose_window_expired_are_forgotten():
    # Spoofed sources that each send a burst and go quiet used to keep their whole last
    # window until seen again: the full history, up to max_tracked_hosts sources.
    extractor = FeatureExtractor(window_seconds=10.0)
    t0 = 1_700_000_000.0
    for i in range(1000):
        source = f"10.{i // 256}.{i % 256}.1"
        for j in range(5):
            extractor.transform_one(pkt(t0 + i * 0.001 + j * 1e-5, src=source))
    assert extractor.tracked_hosts == 1001  # 1000 sources and their one target
    extractor.transform_one(pkt(t0 + 11.0, src="198.51.100.1", dst="198.51.100.2"))
    assert extractor.tracked_hosts == 2
    assert retained_events(extractor) == 2


def test_forgetting_idle_hosts_does_not_change_any_feature(monkeypatch):
    import neuralguard.features
    from neuralguard.simulator import generate_records

    records = [
        normalize_record(r)
        for r in generate_records(20_000, seed=3, attack_ratio=0.3, start_time=1_700_000_000.0)
    ]
    forgetting = FeatureExtractor()
    actual = forgetting.transform(records)
    monkeypatch.setattr(neuralguard.features, "_forget_idle", lambda table, cutoff: None)
    remembering = FeatureExtractor()
    np.testing.assert_array_equal(actual, remembering.transform(records))
    assert forgetting.tracked_hosts < remembering.tracked_hosts / 2


def test_a_hosts_window_holds_at_most_max_window_events(monkeypatch):
    import neuralguard.features

    monkeypatch.setattr(neuralguard.features, "MAX_WINDOW_EVENTS", 50)
    extractor = FeatureExtractor(window_seconds=10.0)
    for i in range(200):  # one host far faster than any attack the model knows
        features = named(extractor, pkt(1_700_000_000.0 + i * 1e-4, dport=1 + i))
    assert features["src_packet_count"] == 50 and features["dst_packet_count"] == 50
    assert features["src_unique_dst_ports"] == 50  # the counters follow the evictions
    assert retained_events(extractor) == 100


def test_tracked_hosts_are_bounded():
    extractor = FeatureExtractor(max_tracked_hosts=10)
    for i in range(100):
        extractor.transform_one(pkt(i, src=f"198.51.100.{i}", dst=f"192.0.2.{i}"))
    assert extractor.tracked_hosts <= 20


def test_transform_matrix_and_reset_are_deterministic():
    records = [pkt(i * 0.5, dport=20 + i % 3) for i in range(20)]
    extractor = FeatureExtractor()
    first = extractor.transform(records)
    extractor.reset()
    second = extractor.transform(records)
    assert first.shape == (20, len(FEATURE_NAMES))
    np.testing.assert_array_equal(first, second)
    assert FeatureExtractor().transform([]).shape == (0, len(FEATURE_NAMES))


@pytest.mark.parametrize("kwargs", [{"window_seconds": 0}, {"max_tracked_hosts": 0}])
def test_invalid_extractor_args(kwargs):
    with pytest.raises(ValueError):
        FeatureExtractor(**kwargs)
