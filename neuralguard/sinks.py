"""Alert sinks: where alert documents go (Elasticsearch, console, JSON Lines file).

Every sink implements ``emit(doc)``, ``flush()``, ``flush_if_due(now=None)`` and
``close()``. A sink must NEVER crash the detector loop because its backend is down: it
logs the problem and carries on.

* :class:`ElasticsearchSink` buffers alert documents and indexes them in batches with
  the bulk API. The alert index is created on the first flush with
  :data:`INDEX_MAPPINGS`, so IP addresses, dates and scores get proper field types
  (Grafana's panels rely on them). While Elasticsearch is unreachable, documents stay
  buffered (bounded by ``max_buffer``; the oldest are dropped first) and delivery is
  retried with a growing back-off, so a dead cluster never stalls packet processing.
* :class:`ConsoleSink` logs one readable line per alert on the ``neuralguard.alerts``
  logger, e.g.
  ``ALERT [high] port_scan score=0.97 203.0.113.5:40000 -> 192.168.1.20:22 TCP S (+12 suppressed)``.
* :class:`JsonlSink` appends one JSON document per line to a file.
"""

from __future__ import annotations

import ipaddress
import itertools
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from elasticsearch import ApiError, Elasticsearch, SerializationError, TransportError
from elasticsearch.helpers import bulk

from neuralguard.config import Settings

logger = logging.getLogger(__name__)
alert_logger = logging.getLogger("neuralguard.alerts")

REQUEST_TIMEOUT_SECONDS = 10.0
ALREADY_EXISTS_ERROR = "resource_already_exists_exception"

# Statuses of a whole bulk request worth retrying later (overloaded / restarting cluster).
_RETRYABLE_STATUSES = frozenset({408, 429, 502, 503, 504})
_MAX_RETRY_DELAY = 60.0
_LOG_REPEAT_SECONDS = 60.0
_MAX_LOGGED_ERRORS = 3

_FIELD_TYPES = {
    "@timestamp": "date",
    "detected_at": "date",
    "source_ip": "ip",
    "destination_ip": "ip",
    "protocol": "keyword",
    "tcp_flags": "keyword",
    "attack_type": "keyword",
    "severity": "keyword",
    "model_version": "keyword",
    "simulated_label": "keyword",
    "source_port": "integer",
    "destination_port": "integer",
    "length": "integer",
    "ttl": "integer",
    "suppressed_count": "integer",
    "threat_score": "float",
}

# The ``mappings`` of the alert index (passed as ``indices.create(mappings=...)``).
# ``features`` holds one float per feature name; a dynamic template types them all as
# float, whatever value the first document happens to carry.
INDEX_MAPPINGS: dict[str, Any] = {
    "dynamic_templates": [
        {"features_as_float": {"path_match": "features.*", "mapping": {"type": "float"}}}
    ],
    "properties": {
        **{name: {"type": field_type} for name, field_type in _FIELD_TYPES.items()},
        "features": {"type": "object", "dynamic": True},
    },
}


class ElasticsearchSink:
    """Buffers alert documents and bulk-indexes them into ``index``.

    The ``elasticsearch.Elasticsearch`` client is built lazily on first use (pass
    ``client`` to inject one). Documents are sent when ``batch_size`` are buffered, when
    ``flush_if_due`` finds ``flush_interval`` seconds have passed, on ``flush()`` and on
    ``close()``. Counters: ``indexed`` (accepted by Elasticsearch), ``failed`` (rejected
    by it - logged, never re-sent) and ``dropped`` (discarded because the buffer exceeded
    ``max_buffer`` while Elasticsearch was unavailable). ``clock`` drives the flush
    interval and retry back-off. Not thread-safe.
    """

    def __init__(
        self,
        hosts: Sequence[str],
        index: str,
        *,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        verify_certs: bool = True,
        ca_certs: str | None = None,
        batch_size: int = 200,
        flush_interval: float = 2.0,
        max_buffer: int = 10_000,
        client: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(hosts, str):
            hosts = [hosts]
        if not hosts:
            raise ValueError("hosts must not be empty")
        if not index:
            raise ValueError("index must not be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if flush_interval < 0:
            raise ValueError(f"flush_interval must be >= 0, got {flush_interval}")
        if max_buffer < 1:
            raise ValueError(f"max_buffer must be >= 1, got {max_buffer}")
        if bool(username) != bool(password):
            raise ValueError("username and password must be given together")

        self.hosts: tuple[str, ...] = tuple(hosts)
        self.index = index
        self.batch_size = int(batch_size)
        self.flush_interval = float(flush_interval)
        self.max_buffer = int(max_buffer)
        self.indexed = 0
        self.failed = 0
        self.dropped = 0

        self._client_options: dict[str, Any] = {
            "request_timeout": REQUEST_TIMEOUT_SECONDS,
            "verify_certs": verify_certs,
        }
        if ca_certs:
            self._client_options["ca_certs"] = ca_certs
        if username and password:
            self._client_options["basic_auth"] = (username, password)
        if api_key:
            self._client_options["api_key"] = api_key

        self._client = client
        self._clock = clock
        # (document id, document): ids make a re-sent batch overwrite, not duplicate.
        self._buffer: deque[tuple[str, dict[str, Any]]] = deque()
        self._index_ready = False
        self._last_flush = clock()
        self._failures = 0  # consecutive failed delivery attempts
        self._last_failure_log: float | None = None
        self._last_drop_log: float | None = None
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> ElasticsearchSink:
        """A sink configured from ``settings`` (hosts, index, credentials, TLS);
        ``kwargs`` are passed through (``batch_size``, ``client``, ...)."""
        return cls(
            settings.es_hosts,
            settings.es_index,
            username=settings.es_username,
            password=settings.es_password,
            api_key=settings.es_api_key,
            verify_certs=settings.es_verify_certs,
            ca_certs=settings.es_ca_certs,
            **kwargs,
        )

    @property
    def buffered(self) -> int:
        """Documents waiting to be sent."""
        return len(self._buffer)

    @property
    def client(self) -> Any:
        """The Elasticsearch client (built on first access)."""
        if self._client is None:
            self._client = Elasticsearch(list(self.hosts), **self._client_options)
        return self._client

    def ensure_index(self) -> bool:
        """Create the alert index with :data:`INDEX_MAPPINGS` unless it exists.

        Returns ``True`` when the index exists afterwards. Errors are logged, not raised.
        """
        try:
            client = self.client
            if not client.indices.exists(index=self.index):
                client.indices.create(index=self.index, mappings=INDEX_MAPPINGS)
                logger.info("created Elasticsearch index %r", self.index)
        except ApiError as exc:
            if _error_type(exc) != ALREADY_EXISTS_ERROR:  # else: created concurrently
                self._log_failure(
                    "cannot check or create Elasticsearch index %r: %s", self.index, exc
                )
                return False
        except TransportError as exc:
            self._log_failure("cannot reach Elasticsearch to check index %r: %s", self.index, exc)
            return False
        except Exception as exc:  # e.g. invalid host URL; must not kill the detector
            self._log_failure("Elasticsearch client error: %s: %s", type(exc).__name__, exc)
            return False
        self._index_ready = True
        return True

    def emit(self, doc: dict[str, Any]) -> None:
        """Buffer one alert document; flush when ``batch_size`` documents are waiting.

        Never raises because of Elasticsearch: delivery problems are logged and the
        documents stay buffered (within ``max_buffer``).
        """
        if self._closed:
            logger.error("ElasticsearchSink is closed; alert discarded")
            return
        self._buffer.append((uuid.uuid4().hex, doc))
        self._enforce_max_buffer()
        if len(self._buffer) >= self.batch_size and self._retry_allowed(self._clock()):
            self.flush()

    def flush(self) -> None:
        """Send every buffered document now, in batches of ``batch_size``.

        Stops at the first batch that cannot be delivered (Elasticsearch unreachable or
        overloaded); it and everything after it stay buffered for the next attempt.
        Does nothing once the sink is closed.
        """
        if self._closed or not self._buffer:
            return
        self._last_flush = self._clock()
        if not self._index_ready and not self.ensure_index():
            self._failed_attempt()
            return
        delivered_before = self.indexed
        while self._buffer:
            batch = list(itertools.islice(self._buffer, self.batch_size))
            if not self._send(batch):
                self._failed_attempt()
                return
            for _ in batch:
                self._buffer.popleft()
        if self._failures:
            logger.info(
                "Elasticsearch is reachable again; indexed %d buffered alerts",
                self.indexed - delivered_before,
            )
        self._failures = 0
        self._last_failure_log = None

    def flush_if_due(self, now: float | None = None) -> None:
        """Flush when documents are waiting and ``flush_interval`` seconds have passed
        since the last flush (longer while backing off after failures)."""
        if not self._buffer:
            return
        now = self._clock() if now is None else now
        if now - self._last_flush >= self.flush_interval and self._retry_allowed(now):
            self.flush()

    def close(self) -> None:
        """Flush what is buffered, then close the client. Safe to call twice."""
        if self._closed:
            return
        self.flush()
        self._closed = True
        if self._buffer:
            logger.warning(
                "closing with %d alerts that could not be sent to Elasticsearch",
                len(self._buffer),
            )
        if self._client is not None:
            try:
                self._client.close()
            except Exception as exc:  # closing is best effort
                logger.debug("error closing Elasticsearch client: %s", exc)

    # -- internals -------------------------------------------------------------------

    def _send(self, batch: list[tuple[str, dict[str, Any]]]) -> bool:
        """Bulk-index one batch. ``False`` means "keep it buffered and retry later"."""
        actions = ({"_index": self.index, "_id": doc_id, "_source": doc} for doc_id, doc in batch)
        try:
            success, errors = bulk(
                self.client, actions, chunk_size=len(batch), raise_on_error=False
            )
        except SerializationError as exc:  # a bad document would fail forever: drop it
            self._reject(batch, f"cannot serialise alert documents: {exc}")
            return True
        except TransportError as exc:  # connection refused, timeout, TLS failure, ...
            self._log_failure(
                "cannot reach Elasticsearch (%s); %d alerts buffered",
                exc,
                len(self._buffer),
            )
            return False
        except ApiError as exc:
            if exc.meta.status in _RETRYABLE_STATUSES:
                self._log_failure(
                    "Elasticsearch is unavailable (%s); %d alerts buffered",
                    exc,
                    len(self._buffer),
                )
                return False
            self._reject(batch, f"Elasticsearch rejected the bulk request: {exc}")
            return True
        except Exception as exc:  # unexpected: never crash the detector loop
            self._reject(batch, f"unexpected error sending alerts: {type(exc).__name__}: {exc}")
            return True

        self.indexed += int(success)
        if errors:
            self.failed += len(errors)
            logger.error(
                "Elasticsearch rejected %d of %d alert documents (not retried); first errors: %s",
                len(errors),
                len(batch),
                "; ".join(_describe_item_error(item) for item in errors[:_MAX_LOGGED_ERRORS]),
            )
        return True

    def _reject(self, batch: list[tuple[str, dict[str, Any]]], reason: str) -> None:
        self.failed += len(batch)
        logger.error("%s; %d alerts discarded", reason, len(batch))

    def _enforce_max_buffer(self) -> None:
        overflow = len(self._buffer) - self.max_buffer
        if overflow <= 0:
            return
        for _ in range(overflow):
            self._buffer.popleft()
        self.dropped += overflow
        now = self._clock()
        if self._last_drop_log is None or now - self._last_drop_log >= _LOG_REPEAT_SECONDS:
            self._last_drop_log = now
            logger.warning(
                "alert buffer full (%d documents, Elasticsearch unavailable); dropping the "
                "oldest - %d dropped so far",
                self.max_buffer,
                self.dropped,
            )

    def _failed_attempt(self) -> None:
        self._failures += 1

    def _retry_delay(self) -> float:
        if not self._failures:
            return 0.0
        base = max(self.flush_interval, 1.0)
        return min(_MAX_RETRY_DELAY, base * 2 ** min(self._failures - 1, 16))

    def _retry_allowed(self, now: float) -> bool:
        return now - self._last_flush >= self._retry_delay()

    def _log_failure(self, message: str, *args: Any) -> None:
        """Log a delivery failure: at ERROR the first time, then at most once a minute
        while the outage lasts (DEBUG in between) - an outage must not flood the logs."""
        now = self._clock()
        last = self._last_failure_log
        if last is None or now - last >= _LOG_REPEAT_SECONDS:
            self._last_failure_log = now
            logger.error(message, *args)
        else:
            logger.debug(message, *args)


class ConsoleSink:
    """Logs each alert as one readable line at WARNING on the ``neuralguard.alerts``
    logger, or writes the lines to ``stream`` when one is given."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream
        self.emitted = 0

    def emit(self, doc: dict[str, Any]) -> None:
        line = format_alert_line(doc)
        self.emitted += 1
        if self.stream is None:
            alert_logger.warning("%s", line)
            return
        try:
            self.stream.write(line + "\n")
        except (OSError, ValueError) as exc:  # closed pipe / closed file
            logger.error("cannot write alert to console stream: %s", exc)

    def flush(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.flush()
        except (OSError, ValueError) as exc:
            logger.error("cannot flush console stream: %s", exc)

    def flush_if_due(self, now: float | None = None) -> None:
        self.flush()

    def close(self) -> None:
        self.flush()


def format_alert_line(doc: dict[str, Any]) -> str:
    """One-line summary of an alert document, e.g.
    ``ALERT [high] port_scan score=0.97 203.0.113.5:40000 -> 192.168.1.20:22 TCP S``."""
    parts = [
        f"ALERT [{doc.get('severity')}] {doc.get('attack_type')}",
        f"score={float(doc.get('threat_score') or 0.0):.2f}",
        _endpoint(doc.get("source_ip"), doc.get("source_port")),
        "->",
        _endpoint(doc.get("destination_ip"), doc.get("destination_port")),
        str(doc.get("protocol") or "OTHER"),
    ]
    if doc.get("tcp_flags"):
        parts.append(str(doc["tcp_flags"]))
    suppressed = int(doc.get("suppressed_count") or 0)
    if suppressed:
        parts.append(f"(+{suppressed} suppressed)")
    return " ".join(parts)


class JsonlSink:
    """Appends each alert document as one JSON line to ``path`` (parents created).

    Lines are written on ``emit`` and flushed to disk on ``flush`` / ``flush_if_due`` /
    ``close``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.path.open("a", encoding="utf-8")
        self._dirty = False
        self.written = 0

    def emit(self, doc: dict[str, Any]) -> None:
        if self._file is None:
            logger.error("JsonlSink for %s is closed; alert discarded", self.path)
            return
        try:
            line = json.dumps(doc, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.error("cannot serialise alert document: %s", exc)
            return
        try:
            self._file.write(line + "\n")
        except OSError as exc:
            logger.error("cannot write alert to %s: %s", self.path, exc)
            return
        self._dirty = True
        self.written += 1

    def flush(self) -> None:
        if self._file is None or not self._dirty:
            return
        try:
            self._file.flush()
        except OSError as exc:
            logger.error("cannot flush %s: %s", self.path, exc)
            return
        self._dirty = False

    def flush_if_due(self, now: float | None = None) -> None:
        self.flush()

    def close(self) -> None:
        if self._file is None:
            return
        self.flush()
        try:
            self._file.close()
        except OSError as exc:
            logger.error("cannot close %s: %s", self.path, exc)
        self._file = None


def _endpoint(ip: Any, port: Any) -> str:
    host = "?" if ip is None else str(ip)
    if not port:
        return host
    if _is_ipv6(host):
        host = f"[{host}]"
    return f"{host}:{port}"


def _is_ipv6(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).version == 6
    except ValueError:
        return False


def _error_type(exc: ApiError) -> str | None:
    body = exc.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return body["error"].get("type")
    return exc.message if isinstance(exc.message, str) else None


def _describe_item_error(item: Any) -> str:
    """Short description of one failed bulk item (without the document itself)."""
    if not isinstance(item, dict) or not item:
        return repr(item)[:200]
    info = next(iter(item.values()))
    if not isinstance(info, dict):
        return repr(info)[:200]
    error = info.get("error")
    if isinstance(error, dict):
        error = f"{error.get('type')}: {error.get('reason')}"
    return f"HTTP {info.get('status')} {error}"[:300]
