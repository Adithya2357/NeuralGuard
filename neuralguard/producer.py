"""Publishes traffic records to Kafka (or to a JSON Lines file).

A *record sink* is anything with ``send(record)``, ``flush()`` and ``close()`` (see
:class:`RecordSink`):

* :class:`KafkaRecordSink` wraps a ``kafka.KafkaProducer`` (build one with
  :func:`create_kafka_producer`, which waits for the brokers to come up). Each record is
  keyed by its source IP, so one source's packets stay ordered within a partition - the
  detector's sliding windows rely on seeing them in time order.
* :class:`JsonlRecordSink` writes one compact JSON object per line to a file, or to
  stdout for ``-``. That is handy for recording a dataset for ``neuralguard train --data``.

:func:`publish` pumps any iterable of records (the simulator, a live capture or a pcap
file) into a sink, optionally pacing them in real time by their timestamps.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any, Protocol

from neuralguard.config import Settings
from neuralguard.kafkautil import (
    BOOTSTRAP_TIMEOUT_MS,
    retryable_kafka_errors,
    supported_options,
)

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 1000  # records between two INFO progress lines in publish()
# Longest single sleep while pacing, so a set stop_event is noticed promptly even when
# the next record is far in the future.
_MAX_PACE_SLEEP_SECONDS = 0.5


class RecordSink(Protocol):
    """Where :func:`publish` sends records."""

    def send(self, record: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


def _serialize_value(value: Any) -> bytes:
    """Kafka value serializer: compact JSON, UTF-8 encoded."""
    return json.dumps(value, separators=(",", ":")).encode()


class KafkaRecordSink:
    """Sends records to a Kafka ``topic`` through a ``kafka.KafkaProducer``-like object.

    The message key is the record's source IP (UTF-8), or ``None`` when it is unknown;
    the value is the record dict itself (the producer's value serializer turns it into
    JSON). Delivery happens asynchronously: a failed delivery is logged and counted in
    ``failed`` by an errback, which never raises inside kafka-python's I/O thread.
    ``sent`` counts the records handed to the producer.
    """

    def __init__(self, producer: Any, topic: str) -> None:
        if not topic:
            raise ValueError("topic must not be empty")
        self.producer = producer
        self.topic = topic
        self.sent = 0
        self.failed = 0
        self._closed = False

    def send(self, record: dict[str, Any]) -> None:
        source_ip = record.get("source_ip")
        key = source_ip.encode("utf-8") if isinstance(source_ip, str) and source_ip else None
        future = self.producer.send(self.topic, value=record, key=key)
        future.add_errback(self._on_delivery_error)
        self.sent += 1

    def _on_delivery_error(self, exc: BaseException) -> None:
        # Runs in the producer's I/O thread: log and count only, never raise.
        self.failed += 1
        logger.error("failed to deliver a record to Kafka topic %r: %s", self.topic, exc)

    def flush(self) -> None:
        """Block until every record sent so far was delivered (or failed)."""
        self.producer.flush()

    def close(self) -> None:
        """Flush, then close the producer. Calling it again does nothing."""
        if self._closed:
            return
        self._closed = True
        try:
            self.producer.flush()
        finally:
            self.producer.close()


class JsonlRecordSink:
    """Writes records as JSON Lines (one compact object per line).

    ``path`` is a file, which is created or truncated, or ``"-"`` for stdout (which is
    flushed but never closed).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = path
        self._stream: IO[str]
        if str(path) == "-":
            self._stream = sys.stdout
            self._owns_stream = False
        else:
            self._stream = Path(path).open("w", encoding="utf-8", newline="\n")  # noqa: SIM115
            self._owns_stream = True
        self._closed = False

    def send(self, record: dict[str, Any]) -> None:
        if self._closed:
            raise ValueError("send() on a closed JsonlRecordSink")
        self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def flush(self) -> None:
        if not self._closed:
            self._stream.flush()

    def close(self) -> None:
        """Flush and, for a file, close it. Calling it again does nothing."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.flush()
        finally:
            if self._owns_stream:
                self._stream.close()


def create_kafka_producer(
    settings: Settings,
    *,
    retries: int = 10,
    backoff_seconds: float = 2.0,
    producer_factory: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """A ``kafka.KafkaProducer`` for ``settings.kafka_bootstrap_servers``.

    Values are serialised as compact JSON; ``acks=1`` (the partition leader confirms)
    and ``linger_ms=20`` (small batches) suit a stream of small records.

    While no broker is reachable (see :func:`~neuralguard.kafkautil.retryable_kafka_errors`)
    the connection is retried ``retries`` more times, ``backoff_seconds`` apart (each
    attempt is logged, and with kafka-python 3.x waits at most
    ``kafkautil.BOOTSTRAP_TIMEOUT_MS``); the last error is re-raised. Other errors are
    raised at once.
    ``producer_factory`` (default ``kafka.KafkaProducer``) and ``sleep`` are injectable
    for tests.
    """
    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries}")
    if backoff_seconds < 0:
        raise ValueError(f"backoff_seconds must be >= 0, got {backoff_seconds}")
    if producer_factory is None:
        from kafka import KafkaProducer

        producer_factory = KafkaProducer
    retryable = retryable_kafka_errors()
    # kafka-python 3.x only: a bounded bootstrap wait, and no idempotence - it needs
    # acks="all", so with acks=1 3.x would turn it off anyway (with a warning every run).
    version_options = supported_options(
        producer_factory,
        {"bootstrap_timeout_ms": BOOTSTRAP_TIMEOUT_MS, "enable_idempotence": False},
    )
    servers = ",".join(settings.kafka_bootstrap_servers)
    attempt = 1
    while True:
        try:
            producer = producer_factory(
                bootstrap_servers=list(settings.kafka_bootstrap_servers),
                value_serializer=_serialize_value,
                acks=1,
                linger_ms=20,
                **version_options,
            )
            break
        except retryable as exc:
            if attempt > retries:
                logger.error(
                    "Kafka is not reachable at %s after %d attempts: %s", servers, attempt, exc
                )
                raise
            logger.warning(
                "Kafka is not reachable at %s (attempt %d/%d: %s); retrying in %.1fs",
                servers,
                attempt,
                retries + 1,
                exc,
                backoff_seconds,
            )
            sleep(backoff_seconds)
            attempt += 1
    logger.info("connected to Kafka at %s", servers)
    return producer


def publish(
    records: Iterable[dict[str, Any]],
    sink: RecordSink,
    *,
    pace: bool = False,
    stop_event: threading.Event | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Send every record to ``sink``; returns how many were sent.

    With ``pace=True`` each record is sent no earlier than its ``timestamp`` (epoch
    seconds, compared with ``clock()``), which replays simulated traffic in real time;
    records whose time has already passed are sent immediately. Stops early once
    ``stop_event`` is set (checked before every record and while pacing). Logs progress
    every ``PROGRESS_EVERY`` records. The sink is always flushed before returning, but
    not closed.
    """
    sent = 0
    try:
        for record in records:
            if _stopped(stop_event):
                break
            if pace and not _wait_until(record.get("timestamp"), stop_event, clock, sleep):
                break
            sink.send(record)
            sent += 1
            if sent % PROGRESS_EVERY == 0:
                logger.info("published %d records", sent)
    finally:
        sink.flush()
    return sent


def _stopped(stop_event: threading.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _wait_until(
    timestamp: Any,
    stop_event: threading.Event | None,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    """Sleep until ``clock()`` reaches ``timestamp``; ``False`` if stopped meanwhile.

    A missing or non-numeric timestamp means "send now".
    """
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return True
    while True:
        delay = timestamp - clock()
        if delay <= 0:
            return True
        sleep(min(delay, _MAX_PACE_SLEEP_SECONDS))
        if _stopped(stop_event):
            return False
