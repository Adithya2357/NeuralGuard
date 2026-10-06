import numpy as np
import pytest

from neuralguard.features import (
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
