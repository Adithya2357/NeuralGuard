"""Model-quality regression tests: what an operator sees from a model trained the way
`neuralguard train` trains it (default simulated data; fewer trees, to stay fast).

They pin down the fixes for three problems: a model that alarmed on ordinary traffic
(about 1000 false alerts an hour, a third of them "SYN floods" of the web server), one
blind to single-source UDP floods (too few episodes of them in its training data), and
held-out metrics measured on a stream with one or two episodes per attack type.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from neuralguard.alerts import AlertThrottler
from neuralguard.config import Settings
from neuralguard.consumer import DetectionService
from neuralguard.detector import Detector
from neuralguard.features import FeatureExtractor
from neuralguard.schema import ATTACK_TYPES
from neuralguard.simulator import TrafficSimulator
from neuralguard.train import DEFAULT_SAMPLES, format_report, simulated_plan, train_model

pytestmark = pytest.mark.slow

START = 1_700_000_000.0
WEB_SERVER = "192.168.1.10"  # the simulated LAN's web server, reachable from the internet


@pytest.fixture(scope="module")
def trained():
    return train_model(n_estimators=30)  # everything else as `neuralguard train` does


class Recorder:
    def __init__(self):
        self.docs = []

    def emit(self, doc):
        self.docs.append(doc)

    def flush(self):
        pass

    def flush_if_due(self, now=None):
        pass

    def close(self):
        pass


def episodes(items):
    """Record indices of each attack episode of ``records_with_episodes`` output."""
    groups = defaultdict(list)
    for index, (_, episode, _) in enumerate(items):
        if episode:
            groups[episode].append(index)
    return list(groups.values())


def test_ordinary_traffic_raises_few_alerts(trained):
    # About 13 minutes of attack-free office traffic, through the detector's own alert
    # path. The first packets of every flood were labelled attacks, and class weights
    # amplified them ~9x: a new visitor's first SYN to the web server looked like one.
    model = trained.model
    sink = Recorder()
    detector = Detector(model)
    service = DetectionService(
        detector,
        [sink],
        throttler=AlertThrottler.from_settings(Settings()),
        model_version=model.version,
    )
    records = list(TrafficSimulator(seed=99, attack_ratio=0.0, start_time=START).records(60_000))
    for start in range(0, len(records), 500):
        service.emit_alerts(detector.process_many(records[start : start + 500]))
    service.flush_alerts()
    hours = (records[-1]["timestamp"] - records[0]["timestamp"]) / 3600
    by_target = Counter((doc["attack_type"], doc["destination_ip"]) for doc in sink.docs)
    assert len(sink.docs) / hours < 400  # was about 1000 an hour
    assert by_target["syn_flood", WEB_SERVER] / hours < 150  # was about 400 an hour
    assert trained.metrics["false_alerts_per_hour"] < 600  # the report's own measurement


def test_every_held_out_attack_episode_is_detected(trained):
    assert trained.metrics["episode_detection_rate"] == 1.0
    assert trained.metrics["detection_rate"] > 0.95


def test_single_source_udp_floods_are_detected(trained):
    # One host flooding hundreds of ports of one target: the old model, which had seen
    # one or two such episodes, caught 30% of their packets and missed some entirely.
    recalls = []
    for seed in range(2000, 2010):
        simulator = TrafficSimulator(
            seed=seed, attack_ratio=0.3, start_time=START, attack_types=("udp_flood",)
        )
        items = list(simulator.records_with_episodes(10_000))
        records = [record for record, _, _ in items]
        flagged = trained.model.threat_scores(FeatureExtractor().transform(records)) >= 0.5
        for members in episodes(items):
            if len({records[i]["source_ip"] for i in members}) == 1:
                recalls.append(float(flagged[members].mean()))
    assert len(recalls) >= 8
    assert np.mean(recalls) >= 0.9
    assert min(recalls) >= 0.5


def test_the_default_test_set_holds_several_episodes_of_every_attack_type():
    # It used to be one 47-second stream: one syn_flood episode, one icmp_flood.
    _, test = simulated_plan(DEFAULT_SAMPLES, seed=42)
    found = Counter()
    for seed, records in test:
        simulator = TrafficSimulator(seed=seed, attack_ratio=0.3, start_time=START)
        items = list(simulator.records_with_episodes(records))
        found.update(items[members[0]][0]["label"] for members in episodes(items))
    assert set(found) == set(ATTACK_TYPES)
    assert min(found.values()) >= 5, found


def test_the_report_counts_episodes_and_false_alerts(trained):
    report = format_report(trained)
    assert "episodes detected" in report
    for label in ATTACK_TYPES:
        counts = trained.metrics["episodes"][label]
        assert f"{counts['detected']} of {counts['episodes']}" in report
    assert "per hour on attack-free traffic" in report
    assert "over 4 streams" in report  # the spread over the held-out streams
