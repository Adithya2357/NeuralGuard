"""Tests for the record producer (fake Kafka producers, fake clocks: no broker, no waiting)."""

from __future__ import annotations

import json
import logging
import threading
from typing import ClassVar

import kafka.errors
import pytest

from neuralguard.config import Settings
from neuralguard.kafkautil import (
    BOOTSTRAP_TIMEOUT_MS,
    bootstrap_timeout_options,
    retryable_kafka_errors,
)
from neuralguard.producer import (
    JsonlRecordSink,
    KafkaRecordSink,
    create_kafka_producer,
    publish,
)
from neuralguard.schema import normalize_record
from neuralguard.simulator import TrafficSimulator

RETRYABLE = retryable_kafka_errors()


def record(ts=1_700_000_000.0, src="192.168.1.10", **extra):
    return normalize_record(
        {
            "timestamp": ts,
            "source_ip": src,
            "destination_ip": "10.0.0.3",
            "protocol": "TCP",
            "source_port": 40000,
            "destination_port": 443,
            "length": 74,
            "ttl": 64,
            "tcp_flags": "S",
            **extra,
        }
    )


# ----------------------------------------------------------------------------- fakes


class FakeFuture:
    def __init__(self):
        self.errbacks = []

    def add_errback(self, callback, *args, **kwargs):
        self.errbacks.append(callback)
        return self


class FakeProducer:
    def __init__(self):
        self.messages = []
        self.futures = []
        self.calls = []

    def send(self, topic, value=None, key=None):
        self.messages.append((topic, key, value))
        future = FakeFuture()
        self.futures.append(future)
        return future

    def flush(self):
        self.calls.append("flush")

    def close(self):
        self.calls.append("close")


class RecordingSink:
    def __init__(self, fail_on=None):
        self.records = []
        self.flushes = 0
        self.fail_on = fail_on

    def send(self, record):
        if self.fail_on is not None and len(self.records) == self.fail_on:
            raise RuntimeError("sink broke")
        self.records.append(record)

    def flush(self):
        self.flushes += 1

    def close(self):
        pass


class FakeClock:
    """``clock()`` and ``sleep()`` sharing one simulated time line."""

    def __init__(self, now):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.sleeps.append(seconds)
        self.now += seconds


# ------------------------------------------------------------------ KafkaRecordSink


def test_kafka_sink_sends_to_the_topic_keyed_by_source_ip():
    producer = FakeProducer()
    sink = KafkaRecordSink(producer, "network-traffic")
    first, second = record(src="192.168.1.10"), record(src="2001:db8::1")
    sink.send(first)
    sink.send(second)
    assert producer.messages == [
        ("network-traffic", b"192.168.1.10", first),
        ("network-traffic", b"2001:db8::1", second),
    ]
    assert sink.sent == 2
    assert all(len(future.errbacks) == 1 for future in producer.futures)


def test_kafka_sink_key_is_none_without_a_source_ip():
    producer = FakeProducer()
    KafkaRecordSink(producer, "t").send(record(src=None))
    ((_, key, _),) = producer.messages
    assert key is None


def test_kafka_sink_errback_logs_and_counts_without_raising(caplog):
    producer = FakeProducer()
    sink = KafkaRecordSink(producer, "traffic")
    sink.send(record())
    (errback,) = producer.futures[0].errbacks
    with caplog.at_level(logging.ERROR, logger="neuralguard.producer"):
        errback(kafka.errors.KafkaTimeoutError("Batch expired"))
    assert sink.failed == 1
    (log,) = caplog.records
    assert "traffic" in log.getMessage() and "Batch expired" in log.getMessage()


def test_kafka_sink_flush_and_close():
    producer = FakeProducer()
    sink = KafkaRecordSink(producer, "t")
    sink.flush()
    assert producer.calls == ["flush"]
    sink.close()
    assert producer.calls == ["flush", "flush", "close"]
    sink.close()  # idempotent
    assert producer.calls == ["flush", "flush", "close"]


def test_kafka_sink_close_closes_even_if_flush_fails():
    producer = FakeProducer()

    def broken_flush():
        raise kafka.errors.KafkaTimeoutError("flush timed out")

    producer.flush = broken_flush
    with pytest.raises(kafka.errors.KafkaTimeoutError):
        KafkaRecordSink(producer, "t").close()
    assert producer.calls == ["close"]


def test_kafka_sink_needs_a_topic():
    with pytest.raises(ValueError):
        KafkaRecordSink(FakeProducer(), "")


# ------------------------------------------------------------------ create_kafka_producer


class FlakyFactory:
    def __init__(self, failures, error):
        self.failures = failures
        self.error = error
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(self.calls) <= self.failures:
            raise self.error("Unable to bootstrap from [('kafka', 9092)]")
        return "producer"


SETTINGS = Settings(kafka_bootstrap_servers=("kafka-1:9092", "kafka-2:9092"))


def test_retryable_errors_cover_kafka_python_3():
    names = {cls.__name__ for cls in RETRYABLE}
    assert {"KafkaTimeoutError", "KafkaConnectionError"} <= names
    assert all(issubclass(cls, Exception) for cls in RETRYABLE)


def test_bootstrap_timeout_only_for_clients_that_support_it():
    class Kafka3Client(FlakyFactory):  # kafka-python 3.x declares the settings
        DEFAULT_CONFIG: ClassVar[dict] = {
            "bootstrap_servers": "x",
            "bootstrap_timeout_ms": 30000,
            "enable_idempotence": True,
        }

    class Kafka2Client(FlakyFactory):  # 2.x does not, and rejects unknown settings
        DEFAULT_CONFIG: ClassVar[dict] = {"bootstrap_servers": "x"}

    new, old = Kafka3Client(0, RuntimeError), Kafka2Client(0, RuntimeError)
    create_kafka_producer(SETTINGS, producer_factory=new)
    create_kafka_producer(SETTINGS, producer_factory=old)
    assert new.calls[0][1]["bootstrap_timeout_ms"] == BOOTSTRAP_TIMEOUT_MS
    assert new.calls[0][1]["enable_idempotence"] is False  # needs acks="all"
    assert "bootstrap_timeout_ms" not in old.calls[0][1]
    assert "enable_idempotence" not in old.calls[0][1]
    assert bootstrap_timeout_options(Kafka3Client, 1234) == {"bootstrap_timeout_ms": 1234}


def test_bootstrap_timeout_matches_the_installed_kafka_python():
    from kafka import KafkaConsumer, KafkaProducer

    for client in (KafkaConsumer, KafkaProducer):
        supported = "bootstrap_timeout_ms" in client.DEFAULT_CONFIG
        assert bootstrap_timeout_options(client) == (
            {"bootstrap_timeout_ms": BOOTSTRAP_TIMEOUT_MS} if supported else {}
        )


def test_create_kafka_producer_configuration():
    factory = FlakyFactory(0, RuntimeError)
    assert create_kafka_producer(SETTINGS, producer_factory=factory) == "producer"
    ((args, kwargs),) = factory.calls
    assert args == ()
    serializer = kwargs.pop("value_serializer")
    assert kwargs == {
        "bootstrap_servers": ["kafka-1:9092", "kafka-2:9092"],
        "acks": 1,
        "linger_ms": 20,
    }
    value = record()
    encoded = serializer(value)
    assert encoded == json.dumps(value, separators=(",", ":")).encode()
    assert json.loads(encoded) == value


@pytest.mark.parametrize("error", RETRYABLE, ids=lambda cls: cls.__name__)
def test_create_kafka_producer_retries_while_brokers_are_unavailable(error, caplog):
    factory = FlakyFactory(2, error)
    sleeps = []
    with caplog.at_level(logging.WARNING, logger="neuralguard.producer"):
        producer = create_kafka_producer(
            SETTINGS, retries=4, backoff_seconds=1.5, producer_factory=factory, sleep=sleeps.append
        )
    assert producer == "producer"
    assert len(factory.calls) == 3
    assert sleeps == [1.5, 1.5]
    attempts = [r.getMessage() for r in caplog.records if "attempt" in r.getMessage()]
    assert len(attempts) == 2
    assert "attempt 1/5" in attempts[0] and "kafka-1:9092,kafka-2:9092" in attempts[0]


@pytest.mark.parametrize("error", RETRYABLE, ids=lambda cls: cls.__name__)
def test_create_kafka_producer_gives_up_after_the_last_retry(error, caplog):
    factory = FlakyFactory(100, error)
    sleeps = []
    with pytest.raises(error), caplog.at_level(logging.ERROR, logger="neuralguard.producer"):
        create_kafka_producer(SETTINGS, retries=3, producer_factory=factory, sleep=sleeps.append)
    assert len(factory.calls) == 4
    assert sleeps == [2.0, 2.0, 2.0]
    assert any("after 4 attempts" in r.getMessage() for r in caplog.records)


def test_create_kafka_producer_zero_retries_tries_once():
    factory = FlakyFactory(1, RETRYABLE[0])
    sleeps = []
    with pytest.raises(RETRYABLE[0]):
        create_kafka_producer(SETTINGS, retries=0, producer_factory=factory, sleep=sleeps.append)
    assert len(factory.calls) == 1 and sleeps == []


def test_create_kafka_producer_does_not_retry_other_errors():
    factory = FlakyFactory(1, kafka.errors.KafkaConfigurationError)
    sleeps = []
    with pytest.raises(kafka.errors.KafkaConfigurationError):
        create_kafka_producer(SETTINGS, producer_factory=factory, sleep=sleeps.append)
    assert len(factory.calls) == 1 and sleeps == []


@pytest.mark.parametrize("kwargs", [{"retries": -1}, {"backoff_seconds": -0.5}])
def test_create_kafka_producer_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        create_kafka_producer(SETTINGS, producer_factory=FlakyFactory(0, RuntimeError), **kwargs)


# ------------------------------------------------------------------ JsonlRecordSink


def test_jsonl_sink_writes_one_compact_object_per_line(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("old content\n")
    sink = JsonlRecordSink(path)
    records = [record(ts=1.0), record(ts=2.0, src=None)]
    for item in records:
        sink.send(item)
    sink.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records  # the old content is replaced
    assert lines[0] == json.dumps(records[0], separators=(",", ":"))
    sink.close()  # idempotent


def test_jsonl_sink_accepts_a_string_path(tmp_path):
    sink = JsonlRecordSink(str(tmp_path / "r.jsonl"))
    sink.send(record())
    sink.flush()
    assert (tmp_path / "r.jsonl").read_text().count("\n") == 1
    sink.close()


def test_jsonl_sink_send_after_close(tmp_path):
    sink = JsonlRecordSink(tmp_path / "r.jsonl")
    sink.close()
    with pytest.raises(ValueError):
        sink.send(record())


def test_jsonl_sink_stdout(capsys):
    import sys

    sink = JsonlRecordSink("-")
    sink.send(record(ts=5.0))
    sink.close()
    assert not sys.stdout.closed  # stdout is flushed, never closed
    out = capsys.readouterr().out
    assert [json.loads(line) for line in out.splitlines()] == [record(ts=5.0)]


def test_jsonl_sink_missing_directory(tmp_path):
    with pytest.raises(OSError):
        JsonlRecordSink(tmp_path / "missing" / "r.jsonl")


# ------------------------------------------------------------------ publish


def test_publish_sends_everything_and_flushes():
    sink = RecordingSink()
    records = [record(ts=float(i)) for i in range(5)]
    assert publish(iter(records), sink) == 5
    assert sink.records == records
    assert sink.flushes == 1


def test_publish_without_pace_never_sleeps():
    clock = FakeClock(now=0.0)
    sink = RecordingSink()
    future = [record(ts=1e9 + i) for i in range(3)]
    assert publish(future, sink, clock=clock, sleep=clock.sleep) == 3
    assert clock.sleeps == []


def test_publish_paces_records_to_their_timestamps():
    clock = FakeClock(now=100.0)
    sent_at = []
    sink = RecordingSink()
    original_send = sink.send

    def send(item):
        sent_at.append(clock.now)
        original_send(item)

    sink.send = send
    records = [record(ts=t) for t in (100.0, 100.25, 101.0, 101.0, 102.75)]
    assert publish(records, sink, pace=True, clock=clock, sleep=clock.sleep) == 5
    assert sent_at == pytest.approx([100.0, 100.25, 101.0, 101.0, 102.75])
    assert all(seconds <= 0.5 for seconds in clock.sleeps)  # long waits are chunked


def test_publish_never_sleeps_for_past_records():
    clock = FakeClock(now=1_000.0)
    records = [record(ts=t) for t in (10.0, 999.0, 999.999, 1_000.0)]
    assert publish(records, RecordingSink(), pace=True, clock=clock, sleep=clock.sleep) == 4
    assert clock.sleeps == []


def test_publish_paces_only_records_with_a_timestamp():
    clock = FakeClock(now=0.0)
    records = [{"source_ip": "10.0.0.1"}, {"timestamp": None}, {"timestamp": True}]
    assert publish(records, RecordingSink(), pace=True, clock=clock, sleep=clock.sleep) == 3
    assert clock.sleeps == []


def test_publish_stops_when_the_stop_event_is_set():
    stop = threading.Event()
    sink = RecordingSink()

    def records():
        for i in range(100):
            if i == 3:
                stop.set()
            yield record(ts=float(i))

    assert publish(records(), sink, stop_event=stop) == 3
    assert len(sink.records) == 3
    assert sink.flushes == 1


def test_publish_stop_event_interrupts_pacing():
    stop = threading.Event()
    clock = FakeClock(now=0.0)

    def sleep(seconds):
        clock.sleep(seconds)
        if clock.now >= 2.0:
            stop.set()

    sink = RecordingSink()
    records = [record(ts=0.0), record(ts=3600.0)]  # the second one is an hour away
    assert publish(records, sink, pace=True, stop_event=stop, clock=clock, sleep=sleep) == 1
    assert clock.now == pytest.approx(2.0)  # stopped within one sleep chunk


def test_publish_with_an_already_set_stop_event_sends_nothing():
    stop = threading.Event()
    stop.set()
    sink = RecordingSink()
    assert publish([record()], sink, stop_event=stop) == 0
    assert sink.records == [] and sink.flushes == 1


def test_publish_flushes_even_when_sending_fails():
    sink = RecordingSink(fail_on=2)
    with pytest.raises(RuntimeError):
        publish([record() for _ in range(5)], sink)
    assert sink.flushes == 1


def test_publish_logs_progress(caplog):
    with caplog.at_level(logging.INFO, logger="neuralguard.producer"):
        publish((record(ts=float(i)) for i in range(2500)), RecordingSink())
    progress = [r.getMessage() for r in caplog.records if r.getMessage().startswith("published")]
    assert progress == ["published 1000 records", "published 2000 records"]


def test_simulated_traffic_round_trips_through_a_jsonl_file(tmp_path):
    path = tmp_path / "simulated.jsonl"
    records = list(TrafficSimulator(seed=3, attack_ratio=0.3, start_time=1e9).records(300))
    sink = JsonlRecordSink(path)
    try:
        assert publish(records, sink) == 300
    finally:
        sink.close()
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == records
    assert all(normalize_record(item) == item for item in loaded)
