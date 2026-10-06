"""The detection service: Kafka -> Detector -> throttling -> alert sinks.

:func:`create_kafka_consumer` connects to Kafka (retrying while the brokers are still
starting, as they often are under ``docker compose up``) and subscribes to the traffic
topic. Messages are consumed as raw bytes and decoded by :func:`decode_message` inside the
loop, so one malformed message is skipped and counted instead of killing the consumer.

:class:`DetectionService` runs the loop: every polled batch goes through the
:class:`~neuralguard.detector.Detector` in one batched model call; each threat is offered
to the :class:`~neuralguard.alerts.AlertThrottler`, and every alert that gets through is
built with :func:`~neuralguard.alerts.build_alert_document` and handed to *every* sink. A
failing sink is logged and the others (and the loop) carry on. A stats line is logged
periodically, and on shutdown every sink is flushed and closed before the consumer.

:func:`install_signal_handlers` turns SIGINT / SIGTERM into a graceful stop
(``docker stop`` and Ctrl+C).
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from neuralguard.config import Settings
from neuralguard.kafkautil import bootstrap_timeout_options, retryable_kafka_errors

if TYPE_CHECKING:
    # Imported lazily at runtime: they pull in scikit-learn, and this module is also
    # used by commands that never touch the model (``install_signal_handlers``).
    from neuralguard.alerts import AlertThrottler
    from neuralguard.detector import Detection, Detector, DetectorStats

logger = logging.getLogger(__name__)

_PREVIEW_BYTES = 80


def create_kafka_consumer(
    settings: Settings,
    *,
    retries: int = 10,
    backoff_seconds: float = 2.0,
    consumer_factory: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """A ``kafka.KafkaConsumer`` subscribed to ``settings.kafka_topic``.

    The consumer joins group ``settings.kafka_group_id``, starts from the latest offset
    when the group has none, auto-commits, and has NO value deserializer: values arrive
    as raw bytes and are decoded with :func:`decode_message` in the loop.

    While no broker is reachable the connection is retried ``retries`` more times,
    ``backoff_seconds`` apart (each attempt is logged, and with kafka-python 3.x waits at
    most ``kafkautil.BOOTSTRAP_TIMEOUT_MS``); the last error is re-raised.
    ``consumer_factory`` (default ``kafka.KafkaConsumer``) and ``sleep`` are injectable
    for tests.
    """
    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries}")
    if backoff_seconds < 0:
        raise ValueError(f"backoff_seconds must be >= 0, got {backoff_seconds}")
    if consumer_factory is None:
        from kafka import KafkaConsumer

        consumer_factory = KafkaConsumer
    retryable = retryable_kafka_errors()
    timeout_options = bootstrap_timeout_options(consumer_factory)
    servers = ",".join(settings.kafka_bootstrap_servers)
    attempt = 1
    while True:
        try:
            consumer = consumer_factory(
                settings.kafka_topic,
                bootstrap_servers=list(settings.kafka_bootstrap_servers),
                group_id=settings.kafka_group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                **timeout_options,
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
    logger.info(
        "consuming topic %r from %s as group %r",
        settings.kafka_topic,
        servers,
        settings.kafka_group_id,
    )
    return consumer


def decode_message(value: bytes) -> dict[str, Any] | None:
    """The JSON object in a Kafka message value, or ``None`` if it is not one.

    Undecodable payloads (not UTF-8, not JSON, not a JSON *object*, or no value at all)
    are logged at WARNING with a truncated preview and ``None`` is returned.
    """
    if value is None:
        logger.warning("skipping message without a value")
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        try:
            text = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.warning("skipping message that is not UTF-8 (%s): %s", exc, _preview(value))
            return None
    else:
        logger.warning("skipping message with a %s value", type(value).__name__)
        return None
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError is a ValueError
        logger.warning("skipping message that is not JSON (%s): %s", exc, _preview(value))
        return None
    if not isinstance(obj, dict):
        logger.warning(
            "skipping message that is not a JSON object (got %s): %s",
            type(obj).__name__,
            _preview(value),
        )
        return None
    return obj


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Make SIGINT and SIGTERM set ``stop_event`` (graceful shutdown).

    A second SIGINT while stopping raises ``KeyboardInterrupt``, so a shutdown stuck on
    an unreachable backend can still be aborted. Python only allows signal handlers in
    the main thread; called from any other thread this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        logger.debug("not in the main thread; signal handlers not installed")
        return

    def _handle(signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        if signum == signal.SIGINT and stop_event.is_set():
            raise KeyboardInterrupt
        logger.info("received %s, shutting down (Ctrl+C again to abort)", name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle)


class DetectionService:
    """Runs raw Kafka message values through the detector and fans alerts out to sinks.

    ``sinks`` implement ``emit(doc)``, ``flush()``, ``flush_if_due()`` and ``close()``
    (see :mod:`neuralguard.sinks`). ``model_version`` is stamped on every alert.
    ``clock`` (monotonic seconds) only drives the periodic stats line. Counters:
    ``alerts_emitted``, ``messages_consumed`` (raw values handled, decodable or not)
    and ``sink_errors`` (sink calls that raised). Not thread-safe.
    """

    def __init__(
        self,
        detector: Detector,
        sinks: Sequence[Any],
        *,
        throttler: AlertThrottler,
        model_version: str,
        stats_interval: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stats_interval <= 0:
            raise ValueError(f"stats_interval must be > 0, got {stats_interval}")
        self.detector = detector
        self.sinks: tuple[Any, ...] = tuple(sinks)
        self.throttler = throttler
        self.model_version = str(model_version)
        self.stats_interval = float(stats_interval)
        self.alerts_emitted = 0
        self.messages_consumed = 0
        self.sink_errors = 0
        self._clock = clock
        self._started = clock()
        self._last_stats_time = self._started
        self._last_stats_messages = 0

    def handle_batch(self, raw_values: list[bytes]) -> int:
        """Decode and detect on a batch of raw message values; returns alerts emitted.

        Undecodable values are counted in ``detector.stats.invalid``; records that
        decode but fail validation are skipped and counted by the detector itself.
        """
        self.messages_consumed += len(raw_values)
        records: list[dict[str, Any]] = []
        for value in raw_values:
            record = decode_message(value)
            if record is None:
                self.detector.stats.invalid += 1
            else:
                records.append(record)
        if not records:
            return 0
        return self.emit_alerts(self.detector.process_many(records))

    def emit_alerts(self, detections: Iterable[Detection]) -> int:
        """Throttle the threats among ``detections`` and send each resulting alert to
        every sink - plus a summary alert for every (attack type, target) that went
        quiet with alerts still held back (see :meth:`AlertThrottler.flush`). Returns
        the number of alerts emitted."""
        emitted = 0
        latest: float | None = None
        for detection in detections:
            timestamp = float(detection.record["timestamp"])
            latest = timestamp if latest is None else max(latest, timestamp)
            if not detection.is_threat:
                continue
            suppressed = self.throttler.check(detection)
            if suppressed is not None:
                self._send_alert(detection, suppressed)
                emitted += 1
        for detection, suppressed in self.throttler.flush(latest):
            self._send_alert(detection, suppressed)
            emitted += 1
        self.alerts_emitted += emitted
        return emitted

    def flush_alerts(self) -> int:
        """Send a summary alert for everything throttling still holds back (on shutdown,
        or at the end of a replay); returns the number of alerts emitted."""
        emitted = 0
        for detection, suppressed in self.throttler.flush(everything=True):
            self._send_alert(detection, suppressed)
            emitted += 1
        self.alerts_emitted += emitted
        return emitted

    def _send_alert(self, detection: Detection, suppressed: int) -> None:
        from neuralguard.alerts import build_alert_document

        doc = build_alert_document(
            detection, model_version=self.model_version, suppressed_count=suppressed
        )
        for sink in self.sinks:
            self._guarded(sink, "emit", doc)

    def tick(self) -> None:
        """Let buffering sinks flush, and log a stats line every ``stats_interval``."""
        for sink in self.sinks:
            self._guarded(sink, "flush_if_due")
        now = self._clock()
        if now - self._last_stats_time >= self.stats_interval:
            self._log_stats(now)

    def run(
        self,
        consumer: Any,
        *,
        stop_event: threading.Event | None = None,
        max_messages: int | None = None,
        poll_timeout_ms: int = 1000,
    ) -> DetectorStats:
        """Consume until ``stop_event`` is set or ``max_messages`` were consumed.

        ``tick()`` runs after every poll. However the loop ends - including by an
        exception - the alerts throttling still holds back are sent, every sink is
        flushed and closed and then the consumer is closed. Returns the detector's stats.
        """
        if max_messages is not None and max_messages < 0:
            raise ValueError(f"max_messages must be >= 0, got {max_messages}")
        if poll_timeout_ms < 0:
            raise ValueError(f"poll_timeout_ms must be >= 0, got {poll_timeout_ms}")
        consumed = 0
        logger.info(
            "detection service running (sinks: %s)",
            ", ".join(type(sink).__name__ for sink in self.sinks) or "none",
        )
        try:
            while not (stop_event is not None and stop_event.is_set()):
                remaining = None if max_messages is None else max_messages - consumed
                if remaining == 0:
                    break
                for messages in self._poll(consumer, poll_timeout_ms, remaining).values():
                    values = [message.value for message in messages]
                    if max_messages is not None:
                        values = values[: max_messages - consumed]
                    if values:
                        consumed += len(values)
                        self.handle_batch(values)
                self.tick()
        finally:
            self._shutdown(consumer)
        return self.detector.stats

    # -- internals --------------------------------------------------------------

    @staticmethod
    def _poll(consumer: Any, timeout_ms: int, max_records: int | None) -> dict[Any, list[Any]]:
        if max_records is None:
            batches = consumer.poll(timeout_ms=timeout_ms)
        else:
            # Never fetch more than we will process: the rest would be auto-committed.
            batches = consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        return batches or {}

    def _guarded(self, sink: Any, method: str, *args: Any) -> None:
        """Call ``sink.<method>(*args)``; a failure is logged, never propagated."""
        try:
            getattr(sink, method)(*args)
        except Exception:
            self.sink_errors += 1
            logger.exception("alert sink %s failed in %s()", type(sink).__name__, method)

    def _shutdown(self, consumer: Any) -> None:
        try:
            self.flush_alerts()
        except Exception:  # never skip closing the sinks and the consumer
            logger.exception("error while sending the held-back alerts")
        for sink in self.sinks:
            self._guarded(sink, "flush")
            self._guarded(sink, "close")
        try:
            consumer.close()
        except Exception:
            logger.exception("error while closing the Kafka consumer")
        self._log_stats(self._clock(), final=True)

    def _log_stats(self, now: float, *, final: bool = False) -> None:
        stats = self.detector.stats
        if final:
            elapsed = now - self._started
            messages = self.messages_consumed
        else:
            elapsed = now - self._last_stats_time
            messages = self.messages_consumed - self._last_stats_messages
        rate = messages / elapsed if elapsed > 0 else 0.0
        suppressed = int(getattr(self.throttler, "suppressed_total", 0))
        logger.info(
            "%s: processed=%d threats=%d alerts=%d suppressed=%d invalid=%d rate=%.1f msg/s",
            "final stats" if final else "stats",
            stats.processed,
            stats.threats,
            self.alerts_emitted,
            suppressed,
            stats.invalid,
            rate,
            extra={
                "processed": stats.processed,
                "threats": stats.threats,
                "alerts_emitted": self.alerts_emitted,
                "suppressed": suppressed,
                "invalid": stats.invalid,
                "msgs_per_sec": round(rate, 1),
            },
        )
        self._last_stats_time = now
        self._last_stats_messages = self.messages_consumed


def _preview(value: Any) -> str:
    """A short, safe rendering of an untrusted payload for log messages."""
    data = value.encode("utf-8", "replace") if isinstance(value, str) else bytes(value)
    text = repr(data[:_PREVIEW_BYTES])
    return f"{text}... ({len(data)} bytes)" if len(data) > _PREVIEW_BYTES else text
