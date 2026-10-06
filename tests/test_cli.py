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
    assert (train.samples, train.seed, train.attack_ratio, train.trees) == (60_000, 42, 0.3, 200)
    assert train.window is None and train.output is None and train.data is None
    assert train.report_json is None

    produce = parser.parse_args(["produce"])
    assert produce.source == "simulate" and produce.count == 0
    assert produce.attack_ratio == 0.2 and produce.no_pace is False and produce.output is None

    detect = parser.parse_args(["detect"])
    assert detect.max_messages == 0 and detect.no_elasticsearch is False
    assert detect.threshold is None and detect.model is None

    demo = parser.parse_args(["demo"])
    assert (demo.count, demo.seed, demo.attack_ratio) == (3000, 7, 0.3)
    assert (demo.train_samples, demo.train_trees) == (20_000, 100)


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
    status, _, err = run(capsys, "detect", "--alerts-file", str(alerts), "--max-messages", "4")
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


def test_detect_settings_environment_and_flag_precedence(capsys, monkeypatch, detect_env):
    monkeypatch.setenv("NEURALGUARD_THREAT_THRESHOLD", "0.9")
    monkeypatch.setenv("NEURALGUARD_ALERT_COOLDOWN_SECONDS", "30")
    monkeypatch.setenv("NEURALGUARD_KAFKA_TOPIC", "env-topic")
    monkeypatch.setenv("NEURALGUARD_ES_HOSTS", "http://env-es:9200")
    monkeypatch.setenv("NEURALGUARD_MODEL_PATH", "env/model.joblib")

    assert run(capsys, "detect", "--max-messages", "1")[0] == 0
    service = detect_env["services"][-1]
    assert service.detector.threshold == 0.9
    assert service.throttler.cooldown_seconds == 30.0
    assert detect_env["kafka"][-1].kafka_topic == "env-topic"
    assert detect_env["es"][-1][0].es_hosts == ("http://env-es:9200",)
    assert str(detect_env["model_path"]) == "env/model.joblib"

    detect_env["consumer"] = FakeConsumer(traffic())
    argv = [
        "detect", "--max-messages", "1", "--threshold", "0.7", "--cooldown", "0",
        "--topic", "flag-topic", "--bootstrap-servers", "k1:9092, k2:9093",
        "--es-hosts", "https://a:9200,https://b:9200", "--model", "flag/model.joblib",
    ]  # fmt: skip
    assert run(capsys, *argv)[0] == 0
    service = detect_env["services"][-1]
    assert service.detector.threshold == 0.7
    assert service.throttler.cooldown_seconds == 0.0
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

    def live_records(interface=None, bpf_filter=None, count=0, stop_event=None):
        seen["live"].append(
            {"interface": interface, "bpf_filter": bpf_filter, "count": count, "stop": stop_event}
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
        "default_bpf_filter",
        lambda settings: f"not (tcp port {settings.kafka_ports[0]})",
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
    assert call["bpf_filter"] == "not (tcp port 9999)"
    assert call["stop"] is not None
    assert produce_env["publish"][0]["pace"] is False
    assert len(FakeRecordSink.instances[0].records) == 4


def test_produce_live_with_a_custom_filter(capsys, produce_env):
    status, _, _ = run(
        capsys, "produce", "--source", "live", "--bpf-filter", "udp", "--output", "-"
    )
    assert status == 0
    assert produce_env["live"][0]["bpf_filter"] == "udp"


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
    assert report["train_size"] == 3000
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


def test_train_output_that_cannot_be_written(capsys, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    status, out, err = run(
        capsys, "train", "--samples", "3000", "--trees", "5", "--output", str(blocker / "m.joblib")
    )
    assert status == 1
    assert "Held-out metrics" in out  # the report is printed before saving
    assert_one_line_error(err)


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


def test_demo_with_a_corrupt_model_does_not_fall_back(capsys, tmp_path):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"definitely not a model")
    status, out, err = run(capsys, "demo", "--count", "100", "--model", str(corrupt))
    assert status == 2
    assert out == ""
    assert_one_line_error(err, "corrupt.joblib")


def test_detect_end_to_end_with_a_real_model(capsys, monkeypatch, tmp_path, tiny_model_path):
    import neuralguard.consumer

    consumer = FakeConsumer([*traffic(20, dport=SSH), b"not json", *traffic(20, dport=443)])
    monkeypatch.setattr(neuralguard.consumer, "create_kafka_consumer", lambda s, **kw: consumer)
    alerts = tmp_path / "alerts.jsonl"
    argv = [
        "detect", "--model", str(tiny_model_path), "--no-elasticsearch",
        "--alerts-file", str(alerts), "--max-messages", "41", "--threshold", "0.5",
    ]  # fmt: skip
    status, _, err = run(capsys, *argv)
    assert status == 0, err
    assert consumer.closed
    for line in alerts.read_text().splitlines():
        doc = json.loads(line)
        assert doc["attack_type"] in ATTACK_TYPES
        assert doc["threat_score"] >= 0.5


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
