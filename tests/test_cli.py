"""Tests for the command-line interface.

Unit tests drive ``main()`` with fakes for Kafka, Elasticsearch, capture and the model.
The end-to-end tests at the bottom run ``train``, ``demo``, ``detect`` and ``produce``
for real on tiny inputs (no network).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pytest

from neuralguard import __version__, cli
from neuralguard.cli import DemoSummary, build_parser, main
from neuralguard.features import FEATURE_NAMES
from neuralguard.schema import ATTACK_TYPES, LABELS, normalize_record

SSH = 22
DST_PORT = FEATURE_NAMES.index("destination_port")


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """No NEURALGUARD_* variables from the outer environment; leave logging and the
    process's signal handlers alone."""
    import neuralguard.consumer

    for name in list(os.environ):
        if name.startswith("NEURALGUARD_"):
            monkeypatch.delenv(name)
    calls = {"logging": [], "signals": []}
    monkeypatch.setattr(
        cli, "configure_logging", lambda level, fmt: calls["logging"].append((level, fmt))
    )
    monkeypatch.setattr(
        neuralguard.consumer, "install_signal_handlers", lambda ev: calls["signals"].append(ev)
    )
    return calls


def run(capsys, *argv):
    """``main(argv)`` -> (exit status, stdout, stderr)."""
    status = main(list(argv))
    out, err = capsys.readouterr()
    return status, out, err


def assert_one_line_error(err, *fragments):
    lines = err.strip().splitlines()
    assert len(lines) == 1, err
    assert lines[0].startswith("error: ")
    assert "Traceback" not in err
    for fragment in fragments:
        assert fragment in lines[0]


# ----------------------------------------------------------------------------- parser


def test_parser_defaults():
    parser = build_parser()
    train = parser.parse_args(["train"])
    assert (train.samples, train.seed, train.attack_ratio, train.trees) == (240_000, 42, 0.3, 200)
    assert train.window is None and train.output is None and train.data is None
    assert train.report_json is None

    produce = parser.parse_args(["produce"])
    assert produce.source == "simulate" and produce.count == 0
    assert produce.attack_ratio == 0.2 and produce.no_pace is False and produce.output is None

    detect = parser.parse_args(["detect"])
    assert detect.max_messages == 0 and detect.no_elasticsearch is False
    assert detect.threshold is None and detect.model is None

    demo = parser.parse_args(["demo"])
    assert (demo.count, demo.seed, demo.attack_ratio) == (6000, 7, 0.3)
    assert (demo.train_samples, demo.train_trees) == (240_000, 100)


def test_training_defaults_agree_with_the_trainer():
    from neuralguard import train

    default = build_parser().parse_args(["train"]).samples
    assert cli.TRAIN_SAMPLES == train.DEFAULT_SAMPLES == default
    assert cli.DEMO_TRAIN_SAMPLES == cli.TRAIN_SAMPLES  # the demo fallback sees as much


def test_default_demo_stream_shows_every_attack_type():
    """A first-time `neuralguard demo` must showcase all attack types, not just one."""
    from collections import Counter

    from neuralguard.simulator import TrafficSimulator

    demo = build_parser().parse_args(["demo"])
    simulator = TrafficSimulator(seed=demo.seed, attack_ratio=demo.attack_ratio)
    counts = Counter(record["label"] for record in simulator.records(demo.count))
    for attack in ATTACK_TYPES:
        assert counts[attack] >= 100, (attack, counts)


def test_parser_global_options_and_flags():
    args = build_parser().parse_args(
        [
            "--log-level", "debug", "--log-format", "json",
            "detect", "--threshold", "0.8", "--cooldown", "0", "--no-elasticsearch",
            "--alerts-file", "alerts.jsonl", "--max-messages", "10",
            "--bootstrap-servers", "k1:9092,k2:9092", "--topic", "t", "--es-hosts", "http://e:9200",
        ]
    )  # fmt: skip
    assert args.log_level == "DEBUG" and args.log_format == "json"
    assert args.command == "detect"
    assert args.threshold == 0.8 and args.cooldown == 0.0 and args.no_elasticsearch
    assert str(args.alerts_file) == "alerts.jsonl" and args.max_messages == 10
    assert args.bootstrap_servers == "k1:9092,k2:9092" and args.es_hosts == "http://e:9200"


def test_demo_internal_options_are_hidden(capsys):
    assert main(["demo", "--help"]) == 0
    out = capsys.readouterr().out
    assert "--count" in out and "--train-samples" not in out


def test_version(capsys):
    status, out, _ = run(capsys, "--version")
    assert status == 0
    assert out.strip() == f"neuralguard {__version__}"


def test_help(capsys):
    status, out, _ = run(capsys, "--help")
    assert status == 0
    for command in ("train", "produce", "detect", "demo"):
        assert command in out


def test_missing_command_is_a_usage_error(capsys):
    status, _, err = run(capsys)
    assert status == 2
    assert "usage:" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--samples", "0"],
        ["train", "--trees", "-3"],
        ["train", "--attack-ratio", "1.5"],
        ["train", "--window", "inf"],
        ["detect", "--threshold", "abc"],
        ["detect", "--threshold", "nan"],
        ["detect", "--max-messages", "-1"],
        ["detect", "--min-hits", "0"],
        ["detect", "--min-hits", "two"],
        ["detect", "--corroboration-window", "0"],
        ["demo", "--corroboration-window", "-5"],
        ["produce", "--source", "carrier-pigeon"],
        ["produce", "--count", "many"],
        ["demo", "--count", "0"],
        ["--log-level", "chatty", "demo"],
        ["--log-format", "xml", "demo"],
        ["no-such-command"],
    ],
)
def test_invalid_flags_exit_2(capsys, argv):
    status, _, err = run(capsys, *argv)
    assert status == 2
    assert "error:" in err


def test_cli_import_is_lightweight():
    code = (
        "import sys, neuralguard.cli; "
        "print(','.join(m for m in ('sklearn', 'scapy', 'kafka', 'elasticsearch', 'numpy') "
        "if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == ""


def test_python_dash_m_entry_point():
    result = subprocess.run(
        [sys.executable, "-m", "neuralguard", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"neuralguard {__version__}"


# ---------------------------------------------------------------- errors and settings


def test_invalid_threshold_value_is_a_one_line_error(capsys):
    status, _, err = run(capsys, "detect", "--threshold", "1.5")
    assert status == 2
    assert_one_line_error(err, "threat_threshold")


def test_detect_with_missing_model_file(capsys, tmp_path):
    status, out, err = run(capsys, "detect", "--model", str(tmp_path / "nope.joblib"))
    assert status == 2
    assert out == ""
    assert_one_line_error(err, "model file not found", "nope.joblib")


def test_invalid_environment_variable(capsys, monkeypatch):
    monkeypatch.setenv("NEURALGUARD_THREAT_THRESHOLD", "very")
    status, _, err = run(capsys, "detect")
    assert status == 2
    assert_one_line_error(err, "NEURALGUARD_THREAT_THRESHOLD")


def test_logging_settings_flag_beats_environment(capsys, monkeypatch, hermetic):
    monkeypatch.setenv("NEURALGUARD_LOG_LEVEL", "warning")
    monkeypatch.setenv("NEURALGUARD_LOG_FORMAT", "json")
    monkeypatch.setenv("NEURALGUARD_MODEL_PATH", "/nonexistent/model.joblib")
    run(capsys, "detect")
    run(capsys, "--log-level", "DEBUG", "--log-format", "text", "detect")
    assert hermetic["logging"] == [("WARNING", "json"), ("DEBUG", "text")]


def test_keyboard_interrupt_exits_130(capsys, monkeypatch):
    from neuralguard.model import ThreatModel

    def interrupted(path):
        raise KeyboardInterrupt

    monkeypatch.setattr(ThreatModel, "load", interrupted)
    status, _, _ = run(capsys, "detect")
    assert status == 130


def test_os_errors_are_one_line_errors(capsys, monkeypatch):
    from neuralguard.model import ThreatModel

    def unreadable(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(ThreatModel, "load", unreadable)
    status, _, err = run(capsys, "detect", "--model", "secret.joblib")
    assert status == 1
    assert_one_line_error(err, "Permission denied", "secret.joblib")


def test_unexpected_errors_are_not_swallowed(capsys, monkeypatch):
    from neuralguard.model import ThreatModel

    def bug(path):
        raise RuntimeError("a genuine bug")

    monkeypatch.setattr(ThreatModel, "load", bug)
    with pytest.raises(RuntimeError, match="genuine bug"):
        main(["detect"])


# ---------------------------------------------------------------------- detect (fakes)


class FakeModel:
    """Port 22 is a port scan, everything else is normal."""

    window_seconds = 10.0
    version = "fakemodel001"

    def predict(self, X):
        return [self.predict_one(row) for row in np.asarray(X)]

    def predict_one(self, x):
        from neuralguard.model import Prediction

        score = 0.97 if int(np.asarray(x)[DST_PORT]) == SSH else 0.02
        return Prediction(score, "port_scan" if score > 0.5 else "normal", "port_scan", {})


@dataclass
class FakeMessage:
    value: bytes


class FakeConsumer:
    def __init__(self, values):
        self.values = list(values)
        self.closed = False

    def poll(self, timeout_ms=0, max_records=None):
        if not self.values:
            return {}
        count = len(self.values) if max_records is None else max_records
        batch, self.values = self.values[:count], self.values[count:]
        return {("network-traffic", 0): [FakeMessage(v) for v in batch]}

    def close(self):
        self.closed = True


class FakeSink:
    def __init__(self):
        self.docs = []
        self.closed = False

    def emit(self, doc):
        self.docs.append(doc)

    def flush(self):
        pass

    def flush_if_due(self, now=None):
        pass

    def close(self):
        self.closed = True


def traffic(n=4, dport=SSH, t0=1_700_000_000.0):
    return [
        json.dumps(
            {
                "timestamp": t0 + i * 10.0,
                "source_ip": "203.0.113.5",
                "destination_ip": "192.168.1.20",
                "protocol": "TCP",
                "source_port": 40000 + i,
                "destination_port": dport,
                "length": 60,
                "ttl": 64,
                "tcp_flags": "S",
            }
        ).encode()
        for i in range(n)
    ]


@pytest.fixture
def detect_env(monkeypatch):
    """Fakes for everything ``detect`` talks to; records what it was given."""
    import neuralguard.consumer
    import neuralguard.sinks
    from neuralguard.model import ThreatModel

    seen = {"es": [], "kafka": [], "services": [], "consumer": FakeConsumer(traffic())}

    def load(path):
        seen["model_path"] = path
        return FakeModel()

    def es_from_settings(settings, **kwargs):
        sink = FakeSink()
        seen["es"].append((settings, sink))
        return sink

    def kafka_consumer(settings, **kwargs):
        seen["kafka"].append(settings)
        return seen["consumer"]

    class RecordingService(neuralguard.consumer.DetectionService):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen["services"].append(self)

    monkeypatch.setattr(ThreatModel, "load", load)
    monkeypatch.setattr(neuralguard.sinks.ElasticsearchSink, "from_settings", es_from_settings)
    monkeypatch.setattr(neuralguard.consumer, "create_kafka_consumer", kafka_consumer)
    monkeypatch.setattr(neuralguard.consumer, "DetectionService", RecordingService)
    return seen


def test_detect_runs_the_service_with_every_sink(capsys, detect_env, hermetic, tmp_path):
    alerts = tmp_path / "out" / "alerts.jsonl"
    # --min-hits 1: every detection may alert at once, so each sink sees all four.
    status, _, err = run(
        capsys, "detect", "--alerts-file", str(alerts), "--max-messages", "4", "--min-hits", "1"
    )
    assert status == 0, err
    (service,) = detect_env["services"]
    assert [type(s).__name__ for s in service.sinks] == ["ConsoleSink", "FakeSink", "JsonlSink"]
    assert service.model_version == "fakemodel001"
    assert service.detector.stats.processed == 4
    assert service.alerts_emitted == 4  # 10 s apart: longer than the 5 s cooldown
    lines = [json.loads(line) for line in alerts.read_text().splitlines()]
    assert len(lines) == service.alerts_emitted
    assert lines[0]["attack_type"] == "port_scan"
    assert detect_env["consumer"].closed
    ((_, es_sink),) = detect_env["es"]
    assert es_sink.closed and len(es_sink.docs) == service.alerts_emitted
    assert len(hermetic["signals"]) == 1


def test_detect_no_elasticsearch(capsys, detect_env):
    status, _, _ = run(capsys, "detect", "--no-elasticsearch", "--max-messages", "2")
    assert status == 0
    (service,) = detect_env["services"]
    assert [type(s).__name__ for s in service.sinks] == ["ConsoleSink"]
    assert detect_env["es"] == []
    assert (service.throttler.min_hits, service.throttler.corroboration_seconds) == (3, 30.0)


def test_detect_settings_environment_and_flag_precedence(capsys, monkeypatch, detect_env):
    monkeypatch.setenv("NEURALGUARD_THREAT_THRESHOLD", "0.9")
    monkeypatch.setenv("NEURALGUARD_ALERT_COOLDOWN_SECONDS", "30")
    monkeypatch.setenv("NEURALGUARD_ALERT_MIN_HITS", "4")
    monkeypatch.setenv("NEURALGUARD_KAFKA_TOPIC", "env-topic")
    monkeypatch.setenv("NEURALGUARD_ES_HOSTS", "http://env-es:9200")
    monkeypatch.setenv("NEURALGUARD_MODEL_PATH", "env/model.joblib")

    assert run(capsys, "detect", "--max-messages", "1")[0] == 0
    service = detect_env["services"][-1]
    assert service.detector.threshold == 0.9
    assert service.throttler.cooldown_seconds == 30.0
    assert service.throttler.min_hits == 4
    assert detect_env["kafka"][-1].kafka_topic == "env-topic"
    assert detect_env["es"][-1][0].es_hosts == ("http://env-es:9200",)
    assert str(detect_env["model_path"]) == "env/model.joblib"

    detect_env["consumer"] = FakeConsumer(traffic())
    argv = [
        "detect", "--max-messages", "1", "--threshold", "0.7", "--cooldown", "0",
        "--min-hits", "2", "--corroboration-window", "9",
        "--topic", "flag-topic", "--bootstrap-servers", "k1:9092, k2:9093",
        "--es-hosts", "https://a:9200,https://b:9200", "--model", "flag/model.joblib",
    ]  # fmt: skip
    assert run(capsys, *argv)[0] == 0
    service = detect_env["services"][-1]
    assert service.detector.threshold == 0.7
    assert service.throttler.cooldown_seconds == 0.0
    assert (service.throttler.min_hits, service.throttler.corroboration_seconds) == (2, 9.0)
    kafka_settings = detect_env["kafka"][-1]
    assert kafka_settings.kafka_topic == "flag-topic"
    assert kafka_settings.kafka_bootstrap_servers == ("k1:9092", "k2:9093")
    assert detect_env["es"][-1][0].es_hosts == ("https://a:9200", "https://b:9200")
    assert str(detect_env["model_path"]) == "flag/model.joblib"


def test_detect_logs_a_startup_line(capsys, caplog, detect_env):
    with caplog.at_level(logging.INFO, logger="neuralguard.cli"):
        run(capsys, "detect", "--no-elasticsearch", "--max-messages", "1", "--threshold", "0.6")
    startup = [r.getMessage() for r in caplog.records if "detector starting" in r.getMessage()]
    assert len(startup) == 1
    for fragment in ("fakemodel001", "threshold 0.60", "window 10.0s"):
        assert fragment in startup[0]


def test_detect_kafka_unavailable(capsys, monkeypatch, detect_env, tmp_path):
    import kafka.errors

    import neuralguard.consumer

    def unavailable(settings, **kwargs):
        raise kafka.errors.KafkaTimeoutError("Unable to bootstrap from ['localhost:9092']")

    monkeypatch.setattr(neuralguard.consumer, "create_kafka_consumer", unavailable)
    status, _, err = run(capsys, "detect", "--alerts-file", str(tmp_path / "a.jsonl"))
    assert status == 1
    assert_one_line_error(err, "Unable to bootstrap")
    ((_, es_sink),) = detect_env["es"]
    assert es_sink.closed  # sinks built before the failure are closed


def test_detect_listens_for_signals_before_waiting_for_kafka(
    capsys, monkeypatch, detect_env, hermetic
):
    # As PID 1 in a container, a process without a SIGTERM handler ignores docker stop:
    # it used to be installed only once Kafka was reachable, up to a minute later.
    import neuralguard.consumer

    def stopped_while_connecting(settings, *, stop_event=None, **kwargs):
        assert hermetic["signals"] == [stop_event]  # installed before this
        stop_event.set()
        return None  # what create_kafka_consumer returns once stop_event is set

    monkeypatch.setattr(neuralguard.consumer, "create_kafka_consumer", stopped_while_connecting)
    status, _, err = run(capsys, "detect")
    assert status == 0, err
    ((_, es_sink),) = detect_env["es"]
    assert es_sink.closed
    assert detect_env["services"] == []  # never started


def test_produce_listens_for_signals_before_waiting_for_kafka(capsys, monkeypatch, hermetic):
    import neuralguard.producer

    def stopped_while_connecting(settings, *, stop_event=None, **kwargs):
        assert hermetic["signals"] == [stop_event]
        return None

    monkeypatch.setattr(neuralguard.producer, "create_kafka_producer", stopped_while_connecting)
    status, _, err = run(capsys, "produce", "--count", "5")
    assert status == 0, err


def test_a_second_ctrl_c_while_flushing_to_kafka_aborts_at_once(capsys, monkeypatch):
    # The first Ctrl+C stops publishing, which then waits for Kafka to take the buffered
    # records; the second must end it. close() used to flush again and kafka-python's
    # close() once more: four Ctrl+C in all.
    import neuralguard.producer

    class StuckProducer:
        def __init__(self):
            self.flushes = 0
            self.closes = []

        def send(self, topic, value=None, key=None):
            return type("Future", (), {"add_errback": lambda self, callback: self})()

        def flush(self):
            self.flushes += 1
            raise KeyboardInterrupt  # the second Ctrl+C, while waiting for the broker

        def close(self, timeout=None):
            self.closes.append(timeout)

    producer = StuckProducer()
    monkeypatch.setattr(
        neuralguard.producer, "create_kafka_producer", lambda settings, **kwargs: producer
    )
    status, _, _ = run(capsys, "produce", "--count", "3", "--no-pace")
    assert status == 130
    assert producer.flushes == 1  # not flushed again ...
    assert producer.closes == [0]  # ... and closed without waiting


def test_detect_empty_bootstrap_servers_is_an_error(capsys, detect_env):
    status, _, err = run(capsys, "detect", "--bootstrap-servers", " , ")
    assert status == 2
    assert_one_line_error(err, "kafka_bootstrap_servers")


# --------------------------------------------------------------------- produce (fakes)


class FakeRecordSink:
    instances: ClassVar[list] = []

    def __init__(self, *args):
        self.args = args
        self.records = []
        self.closed = False
        FakeRecordSink.instances.append(self)

    def send(self, record):
        self.records.append(record)

    def flush(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def produce_env(monkeypatch):
    import neuralguard.capture
    import neuralguard.producer

    seen = {"publish": [], "producers": [], "live": [], "pcap": []}
    FakeRecordSink.instances = []

    def publish(records, sink, *, pace=False, stop_event=None, **kwargs):
        seen["publish"].append({"pace": pace, "stop_event": stop_event})
        count = 0
        for record in records:
            sink.send(record)
            count += 1
        return count

    def create_producer(settings, **kwargs):
        seen["producers"].append(settings)
        return "producer"

    def live_records(
        interface=None,
        bpf_filter=None,
        count=0,
        stop_event=None,
        *,
        exclude_tcp_endpoints=(),
        exclude_tcp_ports=(),
    ):
        seen["live"].append(
            {
                "interface": interface,
                "bpf_filter": bpf_filter,
                "count": count,
                "stop": stop_event,
                "endpoints": tuple(exclude_tcp_endpoints),
                "ports": tuple(exclude_tcp_ports),
            }
        )
        yield from fake_records(count or 3)

    def pcap_records(path):
        seen["pcap"].append(path)
        yield from fake_records(10)

    monkeypatch.setattr(neuralguard.producer, "publish", publish)
    monkeypatch.setattr(neuralguard.producer, "create_kafka_producer", create_producer)
    monkeypatch.setattr(neuralguard.producer, "KafkaRecordSink", FakeRecordSink)
    monkeypatch.setattr(neuralguard.producer, "JsonlRecordSink", FakeRecordSink)
    monkeypatch.setattr(neuralguard.capture, "live_records", live_records)
    monkeypatch.setattr(neuralguard.capture, "pcap_records", pcap_records)
    monkeypatch.setattr(
        neuralguard.capture,
        "default_exclusions",
        lambda settings: neuralguard.capture.Exclusions(settings.kafka_endpoints, (9200,)),
    )
    return seen


def fake_records(n):
    return [
        {"timestamp": 1_700_000_000.0 + i, "source_ip": "10.0.0.1", "protocol": "UDP"}
        for i in range(n)
    ]


def test_produce_simulated_to_kafka(capsys, monkeypatch, produce_env, hermetic):
    monkeypatch.setenv("NEURALGUARD_KAFKA_TOPIC", "env-topic")
    status, out, err = run(
        capsys, "produce", "--count", "25", "--seed", "3", "--bootstrap-servers", "k:1234"
    )
    assert status == 0, err
    assert out == ""
    (settings,) = produce_env["producers"]
    assert settings.kafka_bootstrap_servers == ("k:1234",)
    (sink,) = FakeRecordSink.instances
    assert sink.args == ("producer", "env-topic")
    assert len(sink.records) == 25 and sink.closed
    (call,) = produce_env["publish"]
    assert call["pace"] is True  # simulated traffic is paced by default
    assert call["stop_event"] is hermetic["signals"][0]


def test_produce_no_pace_and_topic_flag(capsys, produce_env):
    status, _, _ = run(capsys, "produce", "--count", "5", "--no-pace", "--topic", "flag-topic")
    assert status == 0
    assert produce_env["publish"][0]["pace"] is False
    assert FakeRecordSink.instances[0].args == ("producer", "flag-topic")


def test_produce_to_a_jsonl_file(capsys, produce_env, tmp_path):
    target = str(tmp_path / "records.jsonl")
    status, _, _ = run(capsys, "produce", "--count", "7", "--no-pace", "--output", target)
    assert status == 0
    assert produce_env["producers"] == []  # no Kafka at all
    (sink,) = FakeRecordSink.instances
    assert sink.args == (target,)
    assert len(sink.records) == 7


def test_produce_live_uses_the_default_bpf_filter(capsys, produce_env):
    argv = ["produce", "--source", "live", "--count", "4", "--interface", "eth1", "--output", "-"]
    status, _, _ = run(capsys, *argv, "--bootstrap-servers", "k:9999")
    assert status == 0
    (call,) = produce_env["live"]
    assert call["interface"] == "eth1" and call["count"] == 4
    assert call["bpf_filter"] is None  # capture builds the filter (or filters in Python)
    assert call["endpoints"] == (("k", 9999),)  # NeuralGuard's own Kafka traffic stays out
    assert call["ports"] == (9200,)
    assert call["stop"] is not None
    assert produce_env["publish"][0]["pace"] is False
    assert len(FakeRecordSink.instances[0].records) == 4


@pytest.mark.parametrize(("flag", "expected"), [("udp", "udp"), ("", None)])
def test_produce_live_with_a_custom_filter(capsys, produce_env, flag, expected):
    status, _, _ = run(
        capsys, "produce", "--source", "live", "--bpf-filter", flag, "--output", "-"
    )
    assert status == 0
    (call,) = produce_env["live"]
    assert call["bpf_filter"] == expected  # '' captures everything
    assert call["endpoints"] == call["ports"] == ()  # the user's filter replaces the default


def test_produce_pcap_honours_count(capsys, produce_env, tmp_path):
    pcap = tmp_path / "capture.pcap"
    status, _, _ = run(
        capsys, "produce", "--source", "pcap", "--pcap", str(pcap), "--count", "4", "--output", "-"
    )
    assert status == 0
    assert produce_env["pcap"] == [pcap]
    assert len(FakeRecordSink.instances[0].records) == 4


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["--source", "pcap"], "--pcap"),
        (["--pcap", "x.pcap"], "--pcap does not apply"),
        (["--source", "live", "--seed", "1"], "--seed does not apply"),
        (["--interface", "eth0"], "--interface does not apply"),
        (["--source", "pcap", "--pcap", "x", "--bpf-filter", "tcp"], "--bpf-filter"),
    ],
)
def test_produce_rejects_options_for_another_source(capsys, produce_env, argv, fragment):
    status, _, err = run(capsys, "produce", "--output", "-", *argv)
    assert status == 2
    assert_one_line_error(err, fragment)


def test_produce_with_a_missing_pcap_leaves_the_output_file_alone(capsys, tmp_path):
    output = tmp_path / "records.jsonl"
    output.write_text('{"keep": "me"}\n')
    status, _, err = run(
        capsys, "produce", "--source", "pcap", "--pcap", str(tmp_path / "missing.pcap"),
        "--output", str(output),
    )  # fmt: skip
    assert status == 2
    assert_one_line_error(err, "pcap file not found")
    assert output.read_text() == '{"keep": "me"}\n'


def test_produce_live_fails_when_the_capture_stops(capsys, monkeypatch, tmp_path):
    """A link that goes down ends scapy's sniffer thread with only a warning. produce must
    then fail with status 1, or restart-on-failure supervisors never restart the sensor."""
    import scapy.sendrecv
    from scapy.layers.inet import IP, TCP
    from scapy.layers.l2 import Ether

    class DyingSniffer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.running = False
            self.exception = None  # scapy stores nothing: the socket error is only logged
            self.thread = None

        def start(self):
            self.running = True
            self.kwargs.get("started_callback", lambda: None)()
            for port in (80, 443):
                ether = Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb")  # no ARP
                packet = ether / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(dport=port)
                packet.time = 1_700_000_000.0
                self.kwargs["prn"](packet)
            self.running = False  # "Network is down ... It was closed."
            self.thread = threading.Thread(target=print)  # never started: not alive

        def stop(self, join=True):
            raise AssertionError("stop() on a sniffer that is not running")

    monkeypatch.setattr(scapy.sendrecv, "AsyncSniffer", DyingSniffer)
    target = tmp_path / "records.jsonl"
    status, _, err = run(
        capsys, "produce", "--source", "live", "--interface", "eth0", "--output", str(target)
    )
    assert status == 1
    assert_one_line_error(err, "capture on eth0 stopped unexpectedly")
    lines = target.read_text().splitlines()
    assert [json.loads(line)["destination_port"] for line in lines] == [80, 443]


def test_produce_live_leaves_out_only_neuralguards_own_connections(capsys, monkeypatch, tmp_path):
    """A scan sent from source port 9092 (nmap -g 9092), or a flood of a web server on the
    port Elasticsearch happens to use, must not vanish from the capture."""
    import scapy.sendrecv
    from scapy.layers.inet import IP, TCP
    from scapy.layers.l2 import Ether

    broker, es, web, sensor = "192.0.2.50", "192.0.2.60", "192.0.2.10", "192.168.1.5"
    flows = [
        ("203.0.113.5", 9092, web, 22),
        (sensor, 40000, broker, 9092),
        (broker, 9092, sensor, 40000),
        ("203.0.113.5", 40000, web, 443),
        (sensor, 40001, es, 443),
    ]

    class Sniffer:
        def __init__(self, **kwargs):
            self.kwargs, self.running, self.exception = kwargs, False, None
            self.thread = threading.Thread(target=print)  # never started: ends after this

        def start(self):
            self.kwargs.get("started_callback", lambda: None)()
            for src, sport, dst, dport in flows:
                ether = Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb")  # no ARP
                packet = ether / IP(src=src, dst=dst) / TCP(sport=sport, dport=dport)
                packet.time = 1_700_000_000.0
                self.kwargs["prn"](packet)

    monkeypatch.setattr(scapy.sendrecv, "AsyncSniffer", Sniffer)
    monkeypatch.setenv("NEURALGUARD_KAFKA_BOOTSTRAP_SERVERS", f"{broker}:9092")
    monkeypatch.setenv("NEURALGUARD_ES_HOSTS", f"https://{es}")  # port 443
    target = tmp_path / "records.jsonl"
    status, _, err = run(
        capsys, "produce", "--source", "live", "--interface", "eth0", "--output", str(target)
    )
    assert status == 1, err  # the fake sniffer stops after its packets
    kept = [json.loads(line) for line in target.read_text().splitlines()]
    assert [(r["source_port"], r["destination_port"]) for r in kept] == [(9092, 22), (40000, 443)]


def test_produce_capture_error_is_a_one_line_error(capsys, monkeypatch, produce_env):
    import neuralguard.capture

    def denied(**kwargs):
        raise neuralguard.capture.CaptureError("live capture needs root (try sudo)")
        yield  # pragma: no cover

    monkeypatch.setattr(neuralguard.capture, "live_records", denied)
    status, _, err = run(capsys, "produce", "--source", "live", "--output", "-")
    assert status == 2
    assert_one_line_error(err, "needs root")
    assert FakeRecordSink.instances[0].closed


# ------------------------------------------------------------------------- DemoSummary


@dataclass
class FakeDetection:
    record: dict
    is_threat: bool
    attack_type: str | None


def test_demo_summary_table():
    summary = DemoSummary()
    summary.add(
        [FakeDetection({"label": "normal"}, False, None)] * 98
        + [FakeDetection({"label": "normal"}, True, "syn_flood")] * 2
        + [FakeDetection({"label": "port_scan"}, True, "port_scan")] * 45
        + [FakeDetection({"label": "port_scan"}, True, "stealth_scan")] * 5
        + [FakeDetection({"label": "port_scan"}, False, None)] * 50
        + [FakeDetection({}, False, None)]
    )
    summary.alerts_emitted, summary.alerts_suppressed = 4, 48
    text = summary.format()
    assert text.isascii()
    lines = text.splitlines()
    normal = next(line for line in lines if line.startswith("normal"))
    assert normal.split() == ["normal", "100", "2", "(2.0%)", "-"]
    scan = next(line for line in lines if line.startswith("port_scan"))
    assert scan.split() == ["port_scan", "100", "50", "(50.0%)", "45", "(90.0%)"]
    assert any(line.startswith("unlabelled") for line in lines)
    assert "50.0% (50 of 100)" in text  # detection rate
    assert "2.0% (2 of 100)" in text  # false-positive rate
    assert summary.total == 201
    assert any("Alerts emitted" in line and line.endswith(" 4") for line in lines)
    assert any("suppressed" in line and line.endswith(" 48") for line in lines)


def test_demo_summary_without_attacks_reports_n_a():
    summary = DemoSummary()
    summary.add([FakeDetection({"label": "normal"}, False, None)] * 3)
    text = summary.format()
    assert "n/a (no packets)" in text
    assert "0.0% (0 of 3)" in text
    assert "Not in this run" not in text  # no attacks at all (e.g. --attack-ratio 0)


def test_demo_summary_names_attack_types_missing_from_the_run():
    summary = DemoSummary()
    summary.add(
        [FakeDetection({"label": "normal"}, False, None)] * 3
        + [FakeDetection({"label": label}, True, label) for label in ATTACK_TYPES[:3]]
    )
    (hint,) = [line for line in summary.format().splitlines() if "Not in this run" in line]
    assert ", ".join(ATTACK_TYPES[3:]) in hint and "--count" in hint
    summary.add([FakeDetection({"label": label}, True, label) for label in ATTACK_TYPES[3:]])
    assert "Not in this run" not in summary.format()


# ------------------------------------------------------------------- end-to-end tests


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory):
    from neuralguard.train import train_model

    path = tmp_path_factory.mktemp("model") / "tiny.joblib"
    train_model(n_samples=3000, n_estimators=10, seed=1).model.save(path)
    return path


def test_train_end_to_end(capsys, tmp_path):
    model_path = tmp_path / "models" / "threat.joblib"
    report_path = tmp_path / "reports" / "metrics.json"
    status, out, err = run(
        capsys,
        "train", "--samples", "3000", "--trees", "10", "--seed", "5", "--window", "8",
        "--output", str(model_path), "--report-json", str(report_path),
    )  # fmt: skip
    assert status == 0, err
    assert "Held-out metrics" in out
    assert f"saved to {model_path}" in out
    assert f"Metrics written to {report_path}" in out
    assert model_path.is_file()
    assert model_path.with_name(model_path.name + ".sha256").is_file()

    from neuralguard.model import ThreatModel

    model = ThreatModel.load(model_path)
    assert model.window_seconds == 8.0
    report = json.loads(report_path.read_text())
    assert report["model_version"] == model.version
    assert report["model_path"] == str(model_path)
    assert report["simulated"] is True
    assert 2900 < report["train_size"] < 3000  # the first packets of each episode left out
    assert 0.0 <= report["metrics"]["accuracy"] <= 1.0
    assert report["training"]["n_estimators"] == 10


def test_train_defaults_to_the_model_path_setting(capsys, monkeypatch, tmp_path):
    target = tmp_path / "from-env.joblib"
    monkeypatch.setenv("NEURALGUARD_MODEL_PATH", str(target))
    status, out, _ = run(capsys, "train", "--samples", "3000", "--trees", "5")
    assert status == 0
    assert target.is_file()
    assert str(target) in out


def test_train_with_missing_data_file(capsys, tmp_path):
    status, _, err = run(capsys, "train", "--data", str(tmp_path / "missing.jsonl"))
    assert status == 2
    assert_one_line_error(err, "missing.jsonl")


def test_train_data_with_a_huge_integer_timestamp_is_a_one_line_error(capsys, tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"timestamp": 1' + "0" * 400 + ', "label": "normal"}\n')
    status, _, err = run(
        capsys, "train", "--data", str(data), "--output", str(tmp_path / "m.joblib")
    )
    assert status == 2
    assert_one_line_error(err, "line 1", "timestamp")


def test_train_output_that_cannot_be_written(capsys, tmp_path, monkeypatch):
    import neuralguard.train

    def must_not_train(**kwargs):
        raise AssertionError("training started although the output cannot be written")

    monkeypatch.setattr(neuralguard.train, "train_model", must_not_train)
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    for flags, fragment in (
        (["--output", str(blocker / "sub" / "m.joblib")], f"Not a directory: {blocker}"),
        (["--output", str(tmp_path)], f"Is a directory: {tmp_path}"),
        (["--report-json", str(tmp_path)], f"Is a directory: {tmp_path}"),
    ):
        status, out, err = run(capsys, "train", "--samples", "3000", *flags)
        assert status == 1
        assert out == ""  # failed before training, not after
        assert_one_line_error(err, fragment)


def test_train_save_failure_after_training_is_a_one_line_error(capsys, tmp_path, monkeypatch):
    from neuralguard.model import ThreatModel

    def disk_full(self, path):
        raise OSError(28, "No space left on device", str(path))

    monkeypatch.setattr(ThreatModel, "save", disk_full)
    target = tmp_path / "m.joblib"
    status, out, err = run(
        capsys, "train", "--samples", "3000", "--trees", "5", "--output", str(target)
    )
    assert status == 1
    assert "Held-out metrics" in out  # the report is still printed before saving
    assert_one_line_error(err, "No space left on device", str(target))


@pytest.mark.parametrize("make_target", ["under_a_file", "a_directory"])
def test_detect_alerts_file_that_cannot_be_opened(capsys, detect_env, tmp_path, make_target):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    if make_target == "under_a_file":
        target, fragment = blocker / "alerts.jsonl", f"Not a directory: {blocker}"
    else:
        target, fragment = tmp_path, f"Is a directory: {tmp_path}"
    status, _, err = run(capsys, "detect", "--no-elasticsearch", "--alerts-file", str(target))
    assert status == 1
    assert_one_line_error(err, fragment)
    assert detect_env["kafka"] == []  # failed before connecting to Kafka


def test_produce_output_that_cannot_be_opened(capsys, tmp_path):
    status, _, err = run(capsys, "produce", "--count", "3", "--no-pace", "--output", str(tmp_path))
    assert status == 1
    assert_one_line_error(err, f"Is a directory: {tmp_path}")


def test_demo_trains_an_in_memory_model_when_the_file_is_missing(capsys, caplog, tmp_path):
    missing = tmp_path / "missing.joblib"
    with caplog.at_level(logging.INFO, logger="neuralguard.cli"):
        status, out, err = run(
            capsys,
            "demo", "--count", "1500", "--model", str(missing),
            "--train-samples", "3000", "--train-trees", "10",
        )  # fmt: skip
    assert status == 0, err
    assert not missing.exists()  # the in-memory model is never saved
    assert any("training a small in-memory model" in r.getMessage() for r in caplog.records)
    assert "1500 simulated packets" in out
    assert "trained in memory" in out
    for heading in ("True label", "Packets", "Flagged as threat", "Correct attack type"):
        assert heading in out
    assert any(line.startswith("normal ") for line in out.splitlines())
    for fragment in ("Detection rate", "False-positive rate", "Alerts emitted", "suppressed"):
        assert fragment in out
    packets = [
        int(line.split()[1])
        for line in out.splitlines()
        if line.split() and line.split()[0] in LABELS
    ]
    assert sum(packets) == 1500


def test_demo_uses_an_existing_model(capsys, tiny_model_path):
    from neuralguard.model import ThreatModel

    status, out, err = run(
        capsys, "demo", "--count", "500", "--model", str(tiny_model_path), "--threshold", "0.6"
    )
    assert status == 0, err
    assert f"loaded from {tiny_model_path}" in out
    assert ThreatModel.load(tiny_model_path).version in out
    assert "threshold 0.60" in out


def test_demo_warns_when_it_replays_the_models_training_traffic(capsys, caplog, tiny_model_path):
    # tiny_model_path was trained on seed 1 at attack ratio 0.3: the very same stream.
    argv = ["demo", "--count", "300", "--model", str(tiny_model_path)]
    with caplog.at_level(logging.WARNING, logger="neuralguard.cli"):
        assert run(capsys, *argv, "--seed", "1")[0] == 0
    assert "the model's own training traffic" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="neuralguard.cli"):
        assert run(capsys, *argv, "--seed", "2")[0] == 0
        assert run(capsys, *argv, "--seed", "1", "--attack-ratio", "0.5")[0] == 0
    assert "training traffic" not in caplog.text


def test_the_demo_fallback_never_trains_on_the_demo_stream(capsys, caplog, monkeypatch, tmp_path):
    # `demo --seed 42` used to score the fallback model on its own training stream
    # (seed 42 is the default training seed): 100% detected, 0.0% false positives.
    import neuralguard.train

    seen = []
    real = neuralguard.train.train_model

    def recording(**kwargs):
        seen.append(kwargs["seed"])
        return real(**kwargs)

    monkeypatch.setattr(neuralguard.train, "train_model", recording)
    argv = ["demo", "--count", "300", "--model", str(tmp_path / "missing.joblib")]
    tiny = ["--train-samples", "3000", "--train-trees", "5"]
    with caplog.at_level(logging.WARNING, logger="neuralguard.cli"):
        assert run(capsys, *argv, *tiny, "--seed", "42")[0] == 0
        assert run(capsys, *argv, *tiny, "--seed", "7")[0] == 0
    assert seen == [43, cli.DEMO_TRAIN_SEED]
    assert "training traffic" not in caplog.text


def test_demo_with_a_corrupt_model_does_not_fall_back(capsys, tmp_path):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"definitely not a model")
    status, out, err = run(capsys, "demo", "--count", "100", "--model", str(corrupt))
    assert status == 2
    assert out == ""
    assert_one_line_error(err, "corrupt.joblib")


def test_detect_end_to_end_with_a_real_model(capsys, monkeypatch, tmp_path, tiny_model_path):
    import neuralguard.consumer
    from neuralguard.model import ThreatModel
    from neuralguard.simulator import generate_records

    # Traffic the tiny model is known to flag: its own (simulated) training stream.
    records = generate_records(1000, seed=1, attack_ratio=0.3, start_time=1_700_000_000.0)
    attacks = sum(record["label"] != "normal" for record in records)
    assert attacks > 100
    values = [json.dumps(record).encode() for record in records]
    consumer = FakeConsumer([*values[:500], b"not json", *values[500:]])
    monkeypatch.setattr(neuralguard.consumer, "create_kafka_consumer", lambda s, **kw: consumer)
    alerts = tmp_path / "alerts.jsonl"
    argv = [
        "detect", "--model", str(tiny_model_path), "--no-elasticsearch",
        "--alerts-file", str(alerts), "--max-messages", "1001", "--threshold", "0.5",
    ]  # fmt: skip
    status, _, err = run(capsys, *argv)
    assert status == 0, err
    assert consumer.closed
    docs = [json.loads(line) for line in alerts.read_text().splitlines()]
    assert docs, "the real model raised no alert at all"
    version = ThreatModel.load(tiny_model_path).version
    for doc in docs:
        assert doc["attack_type"] in ATTACK_TYPES
        assert doc["threat_score"] >= 0.5
        assert doc["model_version"] == version
        assert doc["simulated_label"] in LABELS
    assert sum(doc["simulated_label"] != "normal" for doc in docs) >= len(docs) / 2
    assert sum(1 + doc["suppressed_count"] for doc in docs) <= attacks + 50  # mostly real


def test_produce_no_pace_with_a_count_ends_now(capsys, tmp_path):
    # Generated far faster than real time, a stream starting now ended minutes in the
    # future, and a detector rejects records dated too far ahead (or, before that check,
    # froze its sliding windows on them).
    from neuralguard.simulator import generate_records

    target = tmp_path / "records.jsonl"
    before = time.time()
    status, _, err = run(
        capsys, "produce", "--count", "3000", "--seed", "1", "--no-pace", "--output", str(target)
    )
    assert status == 0, err
    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(records) == 3000
    assert before - 5 < records[-1]["timestamp"] <= time.time()  # ends now, never later
    assert records[0]["timestamp"] < before - 5  # starts as long ago as the stream lasts
    expected = generate_records(3000, seed=1, attack_ratio=0.2, start_time=0.0)
    strip = [{k: v for k, v in r.items() if k != "timestamp"} for r in records]
    assert strip == [{k: v for k, v in r.items() if k != "timestamp"} for r in expected]


def test_unpaced_simulation_without_a_count_warns_that_it_runs_ahead(caplog):
    with caplog.at_level(logging.WARNING, logger="neuralguard.cli"):
        assert cli._unpaced_start(7, 0.2, None) == (7, None)
    assert "runs ahead of the clock" in caplog.text


def test_detect_max_clock_skew(capsys, monkeypatch, detect_env):
    argv = ["detect", "--no-elasticsearch", "--max-messages", "1"]
    assert run(capsys, *argv)[0] == 0
    assert detect_env["services"][-1].detector.max_future_skew == 300.0  # the default
    detect_env["consumer"] = FakeConsumer(traffic())
    monkeypatch.setenv("NEURALGUARD_MAX_CLOCK_SKEW_SECONDS", "60")
    assert run(capsys, *argv)[0] == 0
    assert detect_env["services"][-1].detector.max_future_skew == 60.0
    detect_env["consumer"] = FakeConsumer(traffic())
    assert run(capsys, *argv, "--max-clock-skew", "0")[0] == 0
    assert detect_env["services"][-1].detector.max_future_skew is None  # 0: no check
    status, _, err = run(capsys, *argv, "--max-clock-skew", "-1")
    assert status == 2
    assert_one_line_error(err, "max_clock_skew_seconds")


def test_produce_end_to_end_to_a_jsonl_file(capsys, tmp_path):
    """Needs the real producer (``JsonlRecordSink`` + ``publish``) and simulator."""
    target = tmp_path / "records.jsonl"
    status, _, err = run(
        capsys, "produce", "--count", "50", "--seed", "1", "--no-pace", "--output", str(target)
    )
    assert status == 0, err
    records = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(records) == 50
    for record in records:
        assert normalize_record(record) == record


def test_corroboration_options():
    for command in ("detect", "demo"):
        args = build_parser().parse_args(
            [command, "--min-hits", "5", "--corroboration-window", "12.5"]
        )
        assert (args.min_hits, args.corroboration_window) == (5, 12.5)
        defaults = build_parser().parse_args([command])
        assert (defaults.min_hits, defaults.corroboration_window) == (None, None)  # settings


def test_demo_summary_reports_uncorroborated_detections():
    summary = DemoSummary()
    summary.add([FakeDetection({"label": "normal"}, True, "syn_flood")] * 7)
    summary.alerts_uncorroborated = 7
    lines = summary.format().splitlines()
    assert any(
        line.strip().startswith("Lone detections not corroborated") and line.endswith(" 7")
        for line in lines
    )
