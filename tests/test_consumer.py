"""Tests for the detection service (fakes only: no Kafka, no Elasticsearch)."""

from __future__ import annotations

import json
import logging
import signal
import threading
from dataclasses import dataclass
from typing import ClassVar

import kafka.errors
import numpy as np
import pytest
from kafka.structs import TopicPartition

from neuralguard.alerts import AlertThrottler
from neuralguard.config import Settings
from neuralguard.consumer import (
    DetectionService,
    create_kafka_consumer,
    decode_message,
    install_signal_handlers,
)
from neuralguard.detector import Detector, DetectorStats
from neuralguard.features import FEATURE_NAMES
from neuralguard.kafkautil import BOOTSTRAP_TIMEOUT_MS
from neuralguard.model import Prediction

DST_PORT = FEATURE_NAMES.index("destination_port")
SSH = 22


# ----------------------------------------------------------------------------- fakes


class FakeModel:
    """Anything sent to port 22 is a port scan; everything else is normal."""

    window_seconds = 10.0
    version = "fake-v1"

    def predict(self, X):
        return [self._prediction(row) for row in np.asarray(X)]

    def predict_one(self, x):
        return self._prediction(np.asarray(x))

    @staticmethod
    def _prediction(row):
        score = 0.95 if int(row[DST_PORT]) == SSH else 0.05
        return Prediction(
            threat_score=score,
            predicted_label="port_scan" if score > 0.5 else "normal",
            attack_type="port_scan",
            probabilities={"normal": 1.0 - score, "port_scan": score},
        )


@dataclass
class FakeMessage:
    value: bytes | None


class FakeConsumer:
    """Returns the given poll results in order, then empty dicts."""

    def __init__(self, batches=(), *, stop_after=None, stop_event=None, honour_max_records=True):
        self.batches = list(batches)
        self.polls = []
        self.closed = False
        self.stop_after = stop_after
        self.stop_event = stop_event
        self.honour_max_records = honour_max_records

    def poll(self, timeout_ms=0, max_records=None):
        self.polls.append({"timeout_ms": timeout_ms, "max_records": max_records})
        if self.stop_after is not None and len(self.polls) >= self.stop_after:
            self.stop_event.set()
        if not self.batches:
            return {}
        batch = self.batches.pop(0)
        if self.honour_max_records and max_records is not None:
            batch = {tp: msgs[:max_records] for tp, msgs in batch.items()}
        return batch

    def close(self):
        self.closed = True


class FakeSink:
    def __init__(self):
        self.docs = []
        self.flushes = 0
        self.due_checks = 0
        self.closed = False

    def emit(self, doc):
        self.docs.append(doc)

    def flush(self):
        self.flushes += 1

    def flush_if_due(self, now=None):
        self.due_checks += 1

    def close(self):
        self.closed = True


class RaisingSink(FakeSink):
    def emit(self, doc):
        raise RuntimeError("backend down")

    def flush_if_due(self, now=None):
        raise RuntimeError("backend down")

    def close(self):
        raise RuntimeError("cannot close")


class FakeThrottler:
    """Lets every threat through (or suppresses all of them)."""

    def __init__(self, suppress=False):
        self.suppress = suppress
        self.checked = []
        self.suppressed_total = 0

    def check(self, detection):
        self.checked.append(detection)
        if self.suppress:
            self.suppressed_total += 1
            return None
        return 0

    def flush(self, now=None, *, everything=False):
        return []


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def record(ts=1_700_000_000.0, dport=80, src="10.0.0.1", dst="10.0.0.2", **extra):
    return {
        "timestamp": ts,
        "source_ip": src,
        "destination_ip": dst,
        "protocol": "TCP",
        "source_port": 40000,
        "destination_port": dport,
        "length": 60,
        "ttl": 64,
        "tcp_flags": "S",
        **extra,
    }


def encode(rec):
    return json.dumps(rec).encode()


def make_service(sinks=None, throttler=None, clock=None, **kwargs):
    detector = Detector(FakeModel(), threshold=0.5)
    return DetectionService(
        detector,
        [FakeSink()] if sinks is None else sinks,
        throttler=FakeThrottler() if throttler is None else throttler,
        model_version="fake-v1",
        clock=FakeClock() if clock is None else clock,
        **kwargs,
    )


def partition(n=0):
    return TopicPartition("network-traffic", n)


# -------------------------------------------------------------------- decode_message


def test_decode_message_returns_the_json_object():
    assert decode_message(b'{"source_ip": "10.0.0.1", "length": 60}') == {
        "source_ip": "10.0.0.1",
        "length": 60,
    }


@pytest.mark.parametrize(
    "value",
    [
        b"\xff\xfe not utf-8",
        b"not json at all",
        b'{"truncated": ',
        b"[1, 2, 3]",
        b'"just a string"',
        b"42",
        b"null",
        b"",
        None,
        12345,
    ],
)
def test_decode_message_rejects_undecodable_payloads(value, caplog):
    with caplog.at_level(logging.WARNING, logger="neuralguard.consumer"):
        assert decode_message(value) is None
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_decode_message_truncates_the_payload_in_the_log(caplog):
    payload = b"x" * 10_000
    with caplog.at_level(logging.WARNING, logger="neuralguard.consumer"):
        assert decode_message(payload) is None
    message = caplog.records[-1].getMessage()
    assert len(message) < 400
    assert "10000 bytes" in message


def test_decode_message_survives_deeply_nested_json():
    assert decode_message(b"[" * 100_000 + b"]" * 100_000) is None


# ---------------------------------------------------------------------- handle_batch


def test_handle_batch_counts_undecodable_messages_as_invalid():
    service = make_service()
    emitted = service.handle_batch([b"garbage", b"\xff", encode(record()), b"[]"])
    assert emitted == 0
    assert service.detector.stats.invalid == 3
    assert service.detector.stats.processed == 1
    assert service.messages_consumed == 4


def test_handle_batch_counts_invalid_records_via_the_detector():
    service = make_service()
    service.handle_batch([encode(record(src="not-an-ip")), encode(record())])
    assert service.detector.stats.invalid == 1
    assert service.detector.stats.processed == 1


def test_handle_batch_only_undecodable_messages_does_not_call_the_detector():
    service = make_service()

    def boom(_records):
        raise AssertionError("detector must not be called")

    service.detector.process_many = boom
    assert service.handle_batch([b"nope", b"{"]) == 0
    assert service.detector.stats.invalid == 2


def test_handle_batch_fans_alerts_out_to_every_sink():
    sinks = [FakeSink(), FakeSink(), FakeSink()]
    service = make_service(sinks=sinks)
    emitted = service.handle_batch(
        [encode(record(dport=80)), encode(record(dport=SSH, label="port_scan"))]
    )
    assert emitted == 1
    assert service.alerts_emitted == 1
    for sink in sinks:
        assert len(sink.docs) == 1
        doc = sink.docs[0]
        assert doc["attack_type"] == "port_scan"
        assert doc["destination_port"] == SSH
        assert doc["model_version"] == "fake-v1"
        assert doc["simulated_label"] == "port_scan"
        assert doc["suppressed_count"] == 0
    # every sink gets the same document
    assert sinks[0].docs[0] == sinks[1].docs[0] == sinks[2].docs[0]
    json.dumps(sinks[0].docs[0])  # plain JSON types only


def test_a_raising_sink_does_not_stop_the_others(caplog):
    bad, good = RaisingSink(), FakeSink()
    service = make_service(sinks=[bad, good])
    with caplog.at_level(logging.ERROR, logger="neuralguard.consumer"):
        emitted = service.handle_batch([encode(record(dport=SSH)), encode(record(dport=SSH))])
    assert emitted == 2
    assert len(good.docs) == 2
    assert service.sink_errors == 2
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and all(r.exc_info for r in errors)
    assert "RaisingSink" in errors[0].getMessage()


def test_only_threats_are_offered_to_the_throttler():
    throttler = FakeThrottler()
    service = make_service(throttler=throttler)
    service.handle_batch([encode(record(dport=80)), encode(record(dport=SSH))])
    assert len(throttler.checked) == 1
    assert throttler.checked[0].record["destination_port"] == SSH


def test_suppressed_alerts_are_not_emitted():
    sink = FakeSink()
    service = make_service(sinks=[sink], throttler=FakeThrottler(suppress=True))
    assert service.handle_batch([encode(record(dport=SSH))]) == 0
    assert sink.docs == []
    assert service.alerts_emitted == 0
    assert service.detector.stats.threats == 1


def test_real_throttler_passes_the_suppressed_count_on():
    sink = FakeSink()
    service = make_service(sinks=[sink], throttler=AlertThrottler(cooldown_seconds=5.0))
    t0 = 1_700_000_000.0
    batch = [encode(record(ts=t0 + i, dport=SSH)) for i in (0.0, 1.0, 2.0, 6.0)]
    assert service.handle_batch(batch) == 2
    assert [doc["suppressed_count"] for doc in sink.docs] == [0, 2]


def test_held_back_alerts_are_summarised_when_the_target_goes_quiet():
    sink = FakeSink()
    service = make_service(sinks=[sink], throttler=AlertThrottler(cooldown_seconds=5.0))
    t0 = 1_700_000_000.0
    flood = [encode(record(ts=t0 + i * 0.1, dport=SSH)) for i in range(10)]
    assert service.handle_batch(flood) == 1
    assert [doc["suppressed_count"] for doc in sink.docs] == [0]
    # Only normal traffic afterwards: once the cooldown is over (by packet time), the
    # held-back tail is reported on the latest suppressed packet.
    assert service.handle_batch([encode(record(ts=t0 + 3.0, dport=80))]) == 0
    assert service.handle_batch([encode(record(ts=t0 + 6.0, dport=80))]) == 1
    assert [doc["suppressed_count"] for doc in sink.docs] == [0, 8]
    assert sink.docs[-1]["@timestamp"].startswith("2023-11-14T22:13:20.9")  # t0 + 0.9
    assert service.alerts_emitted == 2
    assert sum(1 + doc["suppressed_count"] for doc in sink.docs) == 10


def test_run_reports_held_back_alerts_before_closing_the_sinks():
    sink = FakeSink()
    service = make_service(sinks=[sink], throttler=AlertThrottler(cooldown_seconds=60.0))
    flood = [record(ts=1_700_000_000.0 + i, dport=SSH) for i in range(5)]
    service.run(FakeConsumer([{partition(0): messages(*flood)}]), max_messages=5)
    assert [doc["suppressed_count"] for doc in sink.docs] == [0, 3]
    assert sink.closed
    assert service.alerts_emitted == 2


# ------------------------------------------------------------------------------ tick


def test_tick_calls_flush_if_due_on_every_sink():
    sinks = [FakeSink(), RaisingSink(), FakeSink()]
    service = make_service(sinks=sinks)
    service.tick()
    service.tick()
    assert sinks[0].due_checks == 2
    assert sinks[2].due_checks == 2
    assert service.sink_errors == 2


def test_tick_logs_stats_every_interval(caplog):
    clock = FakeClock(100.0)
    service = make_service(clock=clock, stats_interval=30.0)
    service.handle_batch([encode(record(dport=SSH)), encode(record()), b"bad"])
    with caplog.at_level(logging.INFO, logger="neuralguard.consumer"):
        clock.now = 110.0
        service.tick()
        assert not [r for r in caplog.records if "stats" in r.getMessage()]
        clock.now = 130.0
        service.tick()
    lines = [r for r in caplog.records if "stats" in r.getMessage()]
    assert len(lines) == 1
    message = lines[0].getMessage()
    for expected in ("processed=2", "threats=1", "alerts=1", "suppressed=0", "invalid=1"):
        assert expected in message
    assert "rate=0.1 msg/s" in message  # 3 messages in 30 s
    assert lines[0].msgs_per_sec == 0.1
    assert lines[0].alerts_emitted == 1

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="neuralguard.consumer"):
        clock.now = 150.0  # only 20 s since the last stats line
        service.tick()
        assert not [r for r in caplog.records if "stats" in r.getMessage()]
        clock.now = 160.0
        service.tick()
    assert "rate=0.0 msg/s" in caplog.records[-1].getMessage()


def test_stats_interval_must_be_positive():
    with pytest.raises(ValueError, match="stats_interval"):
        make_service(stats_interval=0)


# ------------------------------------------------------------------------------- run


def messages(*records):
    return [FakeMessage(encode(rec)) for rec in records]


def test_run_stops_after_max_messages_and_closes_everything():
    sink = FakeSink()
    service = make_service(sinks=[sink])
    consumer = FakeConsumer(
        [
            {partition(0): messages(record(dport=SSH), record())},
            {partition(0): messages(record(), record())},
            {partition(0): messages(record(), record())},
        ]
    )
    stats = service.run(consumer, max_messages=3, poll_timeout_ms=50)
    assert isinstance(stats, DetectorStats)
    assert stats is service.detector.stats
    assert stats.processed == 3
    assert [poll["max_records"] for poll in consumer.polls] == [3, 1]
    assert all(poll["timeout_ms"] == 50 for poll in consumer.polls)
    assert consumer.closed
    assert sink.closed and sink.flushes >= 1
    assert len(sink.docs) == 1


def test_run_never_handles_more_than_max_messages():
    service = make_service()
    big = {partition(0): messages(*[record()] * 5), partition(1): messages(*[record()] * 5)}
    consumer = FakeConsumer([big], honour_max_records=False)
    stats = service.run(consumer, max_messages=7)
    assert stats.processed == 7
    assert service.messages_consumed == 7
    assert len(consumer.polls) == 1


def test_run_handles_each_partition_as_a_batch():
    service = make_service()
    calls = []
    original = service.handle_batch

    def spy(values):
        calls.append(len(values))
        return original(values)

    service.handle_batch = spy
    consumer = FakeConsumer(
        [{partition(0): messages(record(), record()), partition(1): messages(record())}]
    )
    service.run(consumer, max_messages=3)
    assert calls == [2, 1]


def test_run_stops_when_the_stop_event_is_set():
    stop = threading.Event()
    sink = FakeSink()
    service = make_service(sinks=[sink])
    consumer = FakeConsumer(
        [{partition(0): messages(record(dport=SSH))}], stop_after=3, stop_event=stop
    )
    stats = service.run(consumer, stop_event=stop)
    assert len(consumer.polls) == 3
    assert stats.processed == 1
    assert sink.due_checks == 3  # tick() after every poll, including empty ones
    assert consumer.closed and sink.closed


def test_run_with_stop_event_already_set_does_not_poll():
    stop = threading.Event()
    stop.set()
    sink = FakeSink()
    consumer = FakeConsumer([{partition(0): messages(record())}])
    make_service(sinks=[sink]).run(consumer, stop_event=stop)
    assert consumer.polls == []
    assert consumer.closed and sink.closed


def test_run_max_messages_zero_consumes_nothing():
    consumer = FakeConsumer([{partition(0): messages(record())}])
    stats = make_service().run(consumer, max_messages=0)
    assert consumer.polls == [] and consumer.closed
    assert stats.processed == 0


def test_run_counts_undecodable_kafka_messages():
    service = make_service()
    batch = {partition(0): [FakeMessage(b"\x00garbage"), FakeMessage(None), *messages(record())]}
    stats = service.run(FakeConsumer([batch]), max_messages=3)
    assert stats.invalid == 2
    assert stats.processed == 1


def test_run_closes_sinks_and_consumer_when_the_loop_fails():
    sinks = [RaisingSink(), FakeSink()]
    service = make_service(sinks=sinks)

    class BrokenConsumer(FakeConsumer):
        def poll(self, timeout_ms=0, max_records=None):
            raise RuntimeError("poll failed")

    consumer = BrokenConsumer()
    with pytest.raises(RuntimeError, match="poll failed"):
        service.run(consumer)
    assert sinks[1].closed and sinks[1].flushes == 1
    assert consumer.closed


def test_run_closes_everything_when_the_detector_fails():
    sink = FakeSink()
    service = make_service(sinks=[sink])

    def broken(_records):
        raise RuntimeError("model exploded")

    service.detector.process_many = broken
    consumer = FakeConsumer([{partition(0): messages(record())}])
    with pytest.raises(RuntimeError, match="model exploded"):
        service.run(consumer)
    assert sink.closed and consumer.closed


def test_run_survives_a_consumer_that_fails_to_close(caplog):
    class StubbornConsumer(FakeConsumer):
        def close(self):
            raise RuntimeError("cannot close")

    sink = FakeSink()
    with caplog.at_level(logging.ERROR, logger="neuralguard.consumer"):
        make_service(sinks=[sink]).run(StubbornConsumer(), max_messages=0)
    assert sink.closed
    assert any("closing the Kafka consumer" in r.getMessage() for r in caplog.records)


def test_run_logs_final_stats(caplog):
    with caplog.at_level(logging.INFO, logger="neuralguard.consumer"):
        make_service().run(FakeConsumer([{partition(0): messages(record())}]), max_messages=1)
    assert any(r.getMessage().startswith("final stats: processed=1") for r in caplog.records)


@pytest.mark.parametrize("kwargs", [{"max_messages": -1}, {"poll_timeout_ms": -5}])
def test_run_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        make_service().run(FakeConsumer(), **kwargs)


# -------------------------------------------------------------- create_kafka_consumer

RETRYABLE = [
    getattr(kafka.errors, name)
    for name in ("NoBrokersAvailable", "KafkaTimeoutError", "KafkaConnectionError")
    if hasattr(kafka.errors, name)
]


class FlakyFactory:
    def __init__(self, failures, error):
        self.failures = failures
        self.error = error
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(self.calls) <= self.failures:
            raise self.error("no brokers")
        return "consumer"


SETTINGS = Settings(
    kafka_bootstrap_servers=("kafka-1:9092", "kafka-2:9092"),
    kafka_topic="traffic",
    kafka_group_id="detectors",
)


def test_create_kafka_consumer_configuration():
    factory = FlakyFactory(0, RuntimeError)
    assert create_kafka_consumer(SETTINGS, consumer_factory=factory) == "consumer"
    ((args, kwargs),) = factory.calls
    assert args == ("traffic",)
    assert kwargs == {
        "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092"],
        "group_id": "detectors",
        "auto_offset_reset": "latest",
        "enable_auto_commit": True,
    }
    assert "value_deserializer" not in kwargs


def test_create_kafka_consumer_bounds_each_bootstrap_attempt_on_kafka_python_3():
    class Kafka3Consumer(FlakyFactory):
        DEFAULT_CONFIG: ClassVar[dict] = {"group_id": None, "bootstrap_timeout_ms": 30000}

    factory = Kafka3Consumer(0, RuntimeError)
    create_kafka_consumer(SETTINGS, consumer_factory=factory)
    ((_, kwargs),) = factory.calls
    assert kwargs["bootstrap_timeout_ms"] == BOOTSTRAP_TIMEOUT_MS


@pytest.mark.parametrize("error", RETRYABLE, ids=lambda cls: cls.__name__)
def test_create_kafka_consumer_retries_while_brokers_are_unavailable(error, caplog):
    factory = FlakyFactory(2, error)
    sleeps = []
    with caplog.at_level(logging.WARNING, logger="neuralguard.consumer"):
        consumer = create_kafka_consumer(
            SETTINGS, retries=5, backoff_seconds=1.5, consumer_factory=factory, sleep=sleeps.append
        )
    assert consumer == "consumer"
    assert len(factory.calls) == 3
    assert sleeps == [1.5, 1.5]
    attempts = [r.getMessage() for r in caplog.records if "attempt" in r.getMessage()]
    assert len(attempts) == 2 and "attempt 1/6" in attempts[0]


@pytest.mark.parametrize("error", RETRYABLE, ids=lambda cls: cls.__name__)
def test_create_kafka_consumer_gives_up_after_the_last_retry(error):
    factory = FlakyFactory(100, error)
    sleeps = []
    with pytest.raises(error):
        create_kafka_consumer(SETTINGS, retries=3, consumer_factory=factory, sleep=sleeps.append)
    assert len(factory.calls) == 4
    assert len(sleeps) == 3


def test_create_kafka_consumer_zero_retries_tries_once():
    factory = FlakyFactory(1, RETRYABLE[0])
    sleeps = []
    with pytest.raises(RETRYABLE[0]):
        create_kafka_consumer(SETTINGS, retries=0, consumer_factory=factory, sleep=sleeps.append)
    assert len(factory.calls) == 1 and sleeps == []


def test_create_kafka_consumer_does_not_retry_other_errors():
    factory = FlakyFactory(1, kafka.errors.KafkaConfigurationError)
    sleeps = []
    with pytest.raises(kafka.errors.KafkaConfigurationError):
        create_kafka_consumer(SETTINGS, consumer_factory=factory, sleep=sleeps.append)
    assert len(factory.calls) == 1 and sleeps == []


def test_create_kafka_consumer_rejects_negative_retries():
    with pytest.raises(ValueError):
        create_kafka_consumer(SETTINGS, retries=-1, consumer_factory=FlakyFactory(0, RuntimeError))


# ------------------------------------------------------------ install_signal_handlers


@pytest.fixture
def restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.mark.usefixtures("restore_signal_handlers")
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signals_set_the_stop_event(sig):
    stop = threading.Event()
    install_signal_handlers(stop)
    signal.raise_signal(sig)
    assert stop.is_set()


@pytest.mark.usefixtures("restore_signal_handlers")
def test_second_sigint_aborts():
    stop = threading.Event()
    install_signal_handlers(stop)
    signal.raise_signal(signal.SIGINT)
    assert stop.is_set()
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)


@pytest.mark.usefixtures("restore_signal_handlers")
def test_install_signal_handlers_is_a_no_op_outside_the_main_thread():
    before = signal.getsignal(signal.SIGTERM)
    errors = []

    def target():
        try:
            install_signal_handlers(threading.Event())
        except Exception as exc:  # pragma: no cover - the assertion below reports it
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    assert errors == []
    assert signal.getsignal(signal.SIGTERM) is before
