"""Publishes traffic records to Kafka (or to a JSON Lines file).

CONTRACT (implementation TODO):
* ``RecordSink`` protocol: ``send(record: dict) -> None``, ``flush()``, ``close()``.
* ``KafkaRecordSink(producer)``: wraps a ``kafka.KafkaProducer``-like object. ``send``
  uses topic ``settings.kafka_topic`` (pass topic in the constructor), key = source IP
  encoded as UTF-8 (or None) so each source's packets stay ordered within a partition,
  value = the record dict (the producer's value_serializer handles JSON). Attach an
  errback that logs delivery failures (never raises inside the I/O thread). ``close``
  flushes then closes.
* ``create_kafka_producer(settings, *, retries=10, backoff_seconds=2.0,
  producer_factory=None)`` -> KafkaProducer with JSON value serializer
  (``json.dumps(v, separators=(",", ":")).encode()``), ``acks=1``, ``linger_ms=20``.
  Retries ``kafka.errors.NoBrokersAvailable`` with fixed backoff (Kafka is often still
  starting under docker compose), logging each attempt; re-raise after the last one.
  ``producer_factory`` (default ``kafka.KafkaProducer``) is injectable for tests.
* ``JsonlRecordSink(path_or_dash)``: writes one compact JSON object per line to a file,
  or stdout when the path is "-". Useful for recording datasets for ``train --data``.
* ``publish(records, sink, *, pace=False, stop_event=None, clock=time.time,
  sleep=time.sleep) -> int``: sends every record, returns how many were sent. With
  ``pace=True`` it sleeps so records are sent no earlier than their ``timestamp``
  (real-time replay of simulated traffic; never sleeps for records in the past).
  Stops early when ``stop_event`` is set. Logs progress at INFO every 1000 records.
  Always flushes the sink before returning.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from neuralguard.config import Settings


class RecordSink(Protocol):
    def send(self, record: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class KafkaRecordSink:
    def __init__(self, producer: Any, topic: str) -> None:
        raise NotImplementedError

    def send(self, record: dict[str, Any]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class JsonlRecordSink:
    def __init__(self, path: str | Path) -> None:
        raise NotImplementedError

    def send(self, record: dict[str, Any]) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def create_kafka_producer(
    settings: Settings,
    *,
    retries: int = 10,
    backoff_seconds: float = 2.0,
    producer_factory: Callable[..., Any] | None = None,
) -> Any:
    raise NotImplementedError


def publish(
    records: Iterable[dict[str, Any]],
    sink: RecordSink,
    *,
    pace: bool = False,
    stop_event: threading.Event | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    raise NotImplementedError
