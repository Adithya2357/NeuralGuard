import io
import json
import logging
from types import SimpleNamespace

import elastic_transport
import elasticsearch
import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch.helpers import expand_action

from neuralguard import sinks
from neuralguard.config import Settings
from neuralguard.sinks import (
    INDEX_MAPPINGS,
    ConsoleSink,
    ElasticsearchSink,
    JsonlSink,
    format_alert_line,
)

INDEX = "threat-detection"


# --- test doubles ---------------------------------------------------------------------


def api_error(cls, status, error_type, reason="test"):
    """An elasticsearch 8 ApiError exactly as the client raises it for an HTTP error."""
    meta = ApiResponseMeta(
        status=status,
        http_version="1.1",
        headers=HttpHeaders(),
        duration=0.0,
        node=NodeConfig("http", "localhost", 9200),
    )
    body = {
        "error": {
            "root_cause": [{"type": error_type, "reason": reason}],
            "type": error_type,
            "reason": reason,
        },
        "status": status,
    }
    return cls(message=error_type, meta=meta, body=body)


def connection_error():
    return elasticsearch.ConnectionError("Connection error caused by: Connection refused")


class FakeIndices:
    def __init__(self, exists=False):
        self.exists_result = exists
        self.exists_errors = []
        self.create_errors = []
        self.exists_calls = []
        self.create_calls = []

    def exists(self, *, index):
        self.exists_calls.append(index)
        if self.exists_errors:
            raise self.exists_errors.pop(0)
        return self.exists_result

    def create(self, *, index, mappings):
        self.create_calls.append((index, mappings))
        if self.create_errors:
            raise self.create_errors.pop(0)
        self.exists_result = True
        return {"acknowledged": True}


class FakeClient:
    def __init__(self, exists=False):
        self.indices = FakeIndices(exists=exists)
        self.closed = False

    def close(self):
        self.closed = True


class FakeBulk:
    """Replaces ``elasticsearch.helpers.bulk`` as referenced by neuralguard.sinks.

    Each call pops the next outcome: an exception to raise, or a ``(success, errors)``
    result; with no outcome queued every document succeeds.
    """

    def __init__(self):
        self.calls = []
        self.outcomes = []

    def __call__(self, client, actions, **kwargs):
        actions = list(actions)
        self.calls.append(SimpleNamespace(client=client, actions=actions, kwargs=kwargs))
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return len(actions), []

    def sent_docs(self, call=None):
        calls = self.calls if call is None else [self.calls[call]]
        return [action["_source"] for c in calls for action in c.actions]


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def fake_bulk(monkeypatch):
    fake = FakeBulk()
    monkeypatch.setattr(sinks, "bulk", fake)
    return fake


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def clock():
    return FakeClock()


def make_sink(client, clock, **kwargs):
    kwargs.setdefault("batch_size", 100)
    return ElasticsearchSink(["http://es:9200"], INDEX, client=client, clock=clock, **kwargs)


def doc(n=0, **extra):
    return {"n": n, "attack_type": "port_scan", "threat_score": 0.9, **extra}


def alert_doc(**overrides):
    base = {
        "@timestamp": "2024-11-03T15:30:00.123Z",
        "detected_at": "2024-11-03T15:30:05.500Z",
        "source_ip": "203.0.113.5",
        "destination_ip": "192.168.1.20",
        "protocol": "TCP",
        "source_port": 40000,
        "destination_port": 22,
        "length": 60,
        "ttl": 64,
        "tcp_flags": "S",
        "threat_score": 0.9731,
        "attack_type": "port_scan",
        "severity": "high",
        "suppressed_count": 12,
        "model_version": "3f2a9c1b7d4e",
        "features": {"is_tcp": 1.0},
    }
    base.update(overrides)
    return base


# --- mappings -------------------------------------------------------------------------


def test_index_mappings_field_types():
    properties = INDEX_MAPPINGS["properties"]
    types = {name: spec.get("type") for name, spec in properties.items()}
    assert types == {
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
        "features": "object",
    }
    assert properties["features"]["dynamic"] is True
    (template,) = INDEX_MAPPINGS["dynamic_templates"]
    (spec,) = template.values()
    assert spec == {"path_match": "features.*", "mapping": {"type": "float"}}
    json.dumps(INDEX_MAPPINGS)


# --- ElasticsearchSink: index creation ------------------------------------------------


def test_index_is_created_lazily_on_first_flush(client, clock, fake_bulk):
    sink = make_sink(client, clock)
    sink.emit(doc())
    assert client.indices.exists_calls == []
    assert fake_bulk.calls == []

    sink.flush()
    assert client.indices.exists_calls == [INDEX]
    assert client.indices.create_calls == [(INDEX, INDEX_MAPPINGS)]
    assert len(fake_bulk.calls) == 1

    sink.emit(doc(1))
    sink.flush()
    assert client.indices.exists_calls == [INDEX]  # checked once only
    assert len(client.indices.create_calls) == 1


def test_existing_index_is_not_recreated(clock, fake_bulk):
    client = FakeClient(exists=True)
    sink = make_sink(client, clock)
    assert sink.ensure_index() is True
    assert client.indices.create_calls == []


def test_index_already_exists_race_is_success(client, clock, fake_bulk):
    client.indices.create_errors.append(
        api_error(elasticsearch.BadRequestError, 400, "resource_already_exists_exception")
    )
    sink = make_sink(client, clock)
    assert sink.ensure_index() is True
    sink.emit(doc())
    sink.flush()
    assert sink.indexed == 1
    assert client.indices.exists_calls == [INDEX]  # no second check after success


def test_index_creation_failure_is_logged_and_retried(client, clock, fake_bulk, caplog):
    client.indices.create_errors.append(
        api_error(elasticsearch.BadRequestError, 400, "illegal_argument_exception", "bad")
    )
    sink = make_sink(client, clock)
    sink.emit(doc())
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        sink.flush()  # must not raise
    assert "cannot check or create Elasticsearch index" in caplog.text
    # Without the index (and its mappings) nothing is sent; the alert stays buffered.
    assert fake_bulk.calls == []
    assert sink.buffered == 1

    sink.flush()
    assert len(client.indices.create_calls) == 2
    assert fake_bulk.sent_docs() == [doc()]
    assert sink.buffered == 0


def test_index_check_connection_error_returns_false(client, clock, fake_bulk, caplog):
    client.indices.exists_errors.append(connection_error())
    sink = make_sink(client, clock)
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        assert sink.ensure_index() is False
    assert "cannot reach Elasticsearch" in caplog.text
    assert sink.ensure_index() is True


# --- ElasticsearchSink: batching and flushing -----------------------------------------


def test_emit_flushes_every_batch_size_docs(client, clock, fake_bulk):
    sink = make_sink(client, clock, batch_size=3)
    for n in range(7):
        sink.emit(doc(n))
    assert [len(call.actions) for call in fake_bulk.calls] == [3, 3]
    assert sink.buffered == 1
    assert sink.indexed == 6

    sink.flush()
    assert [len(call.actions) for call in fake_bulk.calls] == [3, 3, 1]
    assert fake_bulk.sent_docs() == [doc(n) for n in range(7)]
    assert sink.indexed == 7
    assert sink.buffered == 0


def test_bulk_call_and_action_format(client, clock, fake_bulk):
    sink = make_sink(client, clock, batch_size=2)
    sink.emit(doc(0))
    sink.emit(doc(1))

    (call,) = fake_bulk.calls
    assert call.client is client
    assert call.kwargs["raise_on_error"] is False
    assert call.kwargs["chunk_size"] == 2
    ids = [action["_id"] for action in call.actions]
    assert len(set(ids)) == 2
    for action, expected in zip(call.actions, (doc(0), doc(1)), strict=True):
        # What the real bulk helper turns each action into:
        header, body = expand_action(action)
        assert header == {"index": {"_index": INDEX, "_id": action["_id"]}}
        assert body == expected


def test_flush_splits_large_buffer_into_batches(client, clock, fake_bulk):
    sink = make_sink(client, clock, batch_size=4)
    fake_bulk.outcomes.append(connection_error())  # first auto-flush fails ...
    for n in range(10):
        sink.emit(doc(n))  # ... and the back-off stops emit from retrying every time
    assert len(fake_bulk.calls) == 1
    assert sink.buffered == 10

    sink.flush()
    assert [len(call.actions) for call in fake_bulk.calls] == [4, 4, 4, 2]
    retried = [action["_source"] for call in fake_bulk.calls[1:] for action in call.actions]
    assert retried == [doc(n) for n in range(10)]
    assert sink.indexed == 10


def test_flush_with_empty_buffer_does_nothing(client, clock, fake_bulk):
    sink = make_sink(client, clock)
    sink.flush()
    sink.flush_if_due()
    assert fake_bulk.calls == []
    assert client.indices.exists_calls == []


def test_flush_if_due_uses_injected_clock(client, clock, fake_bulk):
    sink = make_sink(client, clock, flush_interval=2.0)
    sink.emit(doc())
    clock.advance(1.9)
    sink.flush_if_due()
    assert fake_bulk.calls == []

    clock.advance(0.1)
    sink.flush_if_due()
    assert len(fake_bulk.calls) == 1

    sink.emit(doc(1))
    sink.flush_if_due()  # just flushed
    assert len(fake_bulk.calls) == 1
    sink.flush_if_due(now=clock.now + 2.0)  # explicit time on the same clock
    assert len(fake_bulk.calls) == 2
    assert sink.indexed == 2


def test_connection_error_keeps_docs_buffered(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock)
    fake_bulk.outcomes.append(connection_error())
    sink.emit(doc(0))
    sink.emit(doc(1))
    with caplog.at_level(logging.INFO, logger="neuralguard.sinks"):
        sink.flush()  # does not raise
        assert sink.buffered == 2
        assert sink.indexed == 0
        assert "cannot reach Elasticsearch" in caplog.text

        sink.flush()
    assert sink.buffered == 0
    assert sink.indexed == 2
    assert fake_bulk.sent_docs(1) == [doc(0), doc(1)]
    # Re-sent with the same ids, so a request that timed out after indexing cannot
    # produce duplicates.
    first_ids = [a["_id"] for a in fake_bulk.calls[0].actions]
    assert [a["_id"] for a in fake_bulk.calls[1].actions] == first_ids
    assert "reachable again" in caplog.text


def test_connection_timeout_keeps_docs_buffered(client, clock, fake_bulk):
    sink = make_sink(client, clock)
    fake_bulk.outcomes.append(elastic_transport.ConnectionTimeout("timed out"))
    sink.emit(doc())
    sink.flush()
    assert sink.buffered == 1
    assert sink.failed == 0


def test_backoff_after_failures(client, clock, fake_bulk):
    sink = make_sink(client, clock, flush_interval=2.0)
    fake_bulk.outcomes.extend([connection_error(), connection_error()])
    sink.emit(doc())
    sink.flush()  # failure 1 -> wait 2 s
    clock.advance(1.0)
    sink.flush_if_due()
    assert len(fake_bulk.calls) == 1
    clock.advance(1.0)
    sink.flush_if_due()  # failure 2 -> wait 4 s
    assert len(fake_bulk.calls) == 2
    clock.advance(2.0)
    sink.flush_if_due()
    assert len(fake_bulk.calls) == 2
    clock.advance(2.0)
    sink.flush_if_due()
    assert len(fake_bulk.calls) == 3
    assert sink.indexed == 1


def test_outage_does_not_flood_the_logs(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock, flush_interval=0.0)
    fake_bulk.outcomes.extend([connection_error()] * 5)
    sink.emit(doc())
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        for _ in range(5):
            sink.flush()
            clock.advance(1.0)
    assert len(caplog.records) == 1


def test_per_document_errors_are_logged_and_not_resent(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock)
    rejected = {
        "index": {
            "_index": INDEX,
            "_id": "x",
            "status": 400,
            "error": {"type": "mapper_parsing_exception", "reason": "failed to parse [ttl]"},
            "data": {"secret-ish": "document body"},
        }
    }
    fake_bulk.outcomes.append((2, [rejected]))
    for n in range(3):
        sink.emit(doc(n))
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        sink.flush()
    assert sink.indexed == 2
    assert sink.failed == 1
    assert sink.buffered == 0
    assert "rejected 1 of 3" in caplog.text
    assert "mapper_parsing_exception: failed to parse [ttl]" in caplog.text
    assert "document body" not in caplog.text

    sink.flush()
    assert len(fake_bulk.calls) == 1


def test_only_first_few_document_errors_are_logged(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock)
    errors = [
        {"index": {"status": 400, "error": {"type": f"error_{i}", "reason": "r"}}}
        for i in range(10)
    ]
    fake_bulk.outcomes.append((0, errors))
    for n in range(10):
        sink.emit(doc(n))
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        sink.flush()
    assert sink.failed == 10
    assert "error_2" in caplog.text
    assert "error_3" not in caplog.text


def test_overloaded_cluster_keeps_docs_buffered(client, clock, fake_bulk):
    sink = make_sink(client, clock)
    fake_bulk.outcomes.append(
        api_error(elasticsearch.ApiError, 429, "es_rejected_execution_exception")
    )
    sink.emit(doc())
    sink.flush()
    assert sink.buffered == 1
    sink.flush()
    assert sink.indexed == 1


def test_rejected_bulk_request_is_dropped(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock)
    fake_bulk.outcomes.append(api_error(elasticsearch.BadRequestError, 400, "parse_exception"))
    sink.emit(doc(0))
    sink.emit(doc(1))
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        sink.flush()
    assert sink.buffered == 0
    assert sink.failed == 2
    assert "rejected the bulk request" in caplog.text


def test_serialization_error_drops_the_batch(client, clock, fake_bulk):
    sink = make_sink(client, clock)
    fake_bulk.outcomes.append(elasticsearch.SerializationError("cannot serialize"))
    sink.emit(doc())
    sink.flush()
    assert sink.buffered == 0
    assert sink.failed == 1


def test_unexpected_error_does_not_raise(client, clock, fake_bulk):
    sink = make_sink(client, clock)
    fake_bulk.outcomes.append(RuntimeError("boom"))
    sink.emit(doc())
    sink.flush()
    assert sink.failed == 1


def test_max_buffer_drops_oldest(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock, batch_size=10, max_buffer=3)
    with caplog.at_level(logging.WARNING, logger="neuralguard.sinks"):
        for n in range(6):
            sink.emit(doc(n))
    assert sink.dropped == 3
    assert sink.buffered == 3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1  # rate limited
    assert "dropping the oldest" in warnings[0].getMessage()

    sink.flush()
    assert fake_bulk.sent_docs() == [doc(3), doc(4), doc(5)]


def test_max_buffer_while_elasticsearch_is_down(client, clock, fake_bulk):
    sink = make_sink(client, clock, batch_size=2, max_buffer=4)
    fake_bulk.outcomes.append(connection_error())
    for n in range(7):
        sink.emit(doc(n))
    assert sink.buffered == 4
    assert sink.dropped == 3
    sink.flush()
    assert fake_bulk.sent_docs() == [doc(n) for n in range(2)] + [doc(n) for n in range(3, 7)]
    assert sink.indexed == 4


def test_close_flushes_and_closes_client(client, clock, fake_bulk):
    sink = make_sink(client, clock)
    sink.emit(doc())
    sink.close()
    assert sink.indexed == 1
    assert client.closed
    sink.close()  # idempotent
    assert len(fake_bulk.calls) == 1


def test_close_while_elasticsearch_is_down(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock)
    fake_bulk.outcomes.append(connection_error())
    sink.emit(doc())
    with caplog.at_level(logging.WARNING, logger="neuralguard.sinks"):
        sink.close()  # does not raise
    assert client.closed
    assert "1 alerts that could not be sent" in caplog.text


def test_emit_and_flush_after_close_do_nothing(client, clock, fake_bulk, caplog):
    sink = make_sink(client, clock, batch_size=1)
    sink.close()
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        sink.emit(doc())
    assert sink.buffered == 0
    assert "closed" in caplog.text

    down = make_sink(FakeClient(), clock)
    fake_bulk.outcomes.append(connection_error())
    down.emit(doc())
    down.close()
    calls = len(fake_bulk.calls)
    down.flush()
    down.flush_if_due(now=clock.now + 3600)
    assert len(fake_bulk.calls) == calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"flush_interval": -1.0},
        {"max_buffer": 0},
        {"username": "elastic"},
        {"password": "secret"},
    ],
)
def test_invalid_arguments(kwargs, client):
    with pytest.raises(ValueError):
        ElasticsearchSink(["http://es:9200"], INDEX, client=client, **kwargs)


def test_empty_hosts_or_index_rejected(client):
    with pytest.raises(ValueError, match="hosts"):
        ElasticsearchSink([], INDEX, client=client)
    with pytest.raises(ValueError, match="index"):
        ElasticsearchSink(["http://es:9200"], "", client=client)


# --- ElasticsearchSink: client construction -------------------------------------------


@pytest.fixture
def es_factory(monkeypatch):
    created = []

    def factory(hosts, **kwargs):
        created.append(SimpleNamespace(hosts=hosts, kwargs=kwargs))
        return FakeClient()

    monkeypatch.setattr(sinks, "Elasticsearch", factory)
    return created


def test_client_is_built_lazily(es_factory, fake_bulk):
    sink = ElasticsearchSink(["http://es:9200"], INDEX)
    assert es_factory == []
    sink.emit(doc())
    assert es_factory == []
    sink.flush()
    assert len(es_factory) == 1
    sink.flush_if_due(now=1e12)
    assert len(es_factory) == 1


def test_from_settings_with_basic_auth(es_factory):
    settings = Settings(
        es_hosts=("https://es1:9200", "https://es2:9200"),
        es_index="alerts",
        es_username="elastic",
        es_password="changeme",
        es_verify_certs=False,
        es_ca_certs="/etc/ssl/ca.pem",
    )
    sink = ElasticsearchSink.from_settings(settings, batch_size=5)
    assert sink.index == "alerts"
    assert sink.batch_size == 5
    sink.client  # noqa: B018 - builds the client
    (created,) = es_factory
    assert created.hosts == ["https://es1:9200", "https://es2:9200"]
    assert created.kwargs == {
        "request_timeout": 10.0,
        "verify_certs": False,
        "ca_certs": "/etc/ssl/ca.pem",
        "basic_auth": ("elastic", "changeme"),
    }


def test_from_settings_with_api_key(es_factory):
    sink = ElasticsearchSink.from_settings(Settings(es_api_key="a2V5OnNlY3JldA=="))
    sink.client  # noqa: B018
    (created,) = es_factory
    assert created.hosts == ["http://localhost:9200"]
    assert created.kwargs["api_key"] == "a2V5OnNlY3JldA=="
    assert "basic_auth" not in created.kwargs


def test_from_settings_defaults(es_factory):
    sink = ElasticsearchSink.from_settings(Settings())
    assert sink.index == "threat-detection"
    sink.client  # noqa: B018
    (created,) = es_factory
    assert created.kwargs == {"request_timeout": 10.0, "verify_certs": True}


def test_injected_client_is_used(client, es_factory):
    sink = ElasticsearchSink.from_settings(Settings(), client=client)
    assert sink.client is client
    assert es_factory == []


def test_real_client_construction_needs_no_server():
    sink = ElasticsearchSink(["http://localhost:9200"], INDEX, username="u", password="p")
    try:
        assert isinstance(sink.client, elasticsearch.Elasticsearch)
    finally:
        sink.client.close()


def test_secrets_are_never_logged(es_factory, fake_bulk, caplog):
    fake_bulk.outcomes.append(connection_error())
    sink = ElasticsearchSink(
        ["http://es:9200"], INDEX, username="elastic", password="s3cr3t-pw", batch_size=1
    )
    with caplog.at_level(logging.DEBUG):
        sink.emit(doc())
        sink.close()
    assert "s3cr3t-pw" not in caplog.text
    assert "s3cr3t-pw" not in repr(sink)


def test_elasticsearch_exception_hierarchy_assumptions():
    # The sink relies on these relationships of the elasticsearch 8 client.
    assert elasticsearch.ConnectionError is elastic_transport.ConnectionError
    assert issubclass(elasticsearch.ConnectionError, elasticsearch.TransportError)
    assert issubclass(elasticsearch.ConnectionTimeout, elasticsearch.TransportError)
    assert issubclass(elasticsearch.SerializationError, elasticsearch.TransportError)
    assert issubclass(elasticsearch.BadRequestError, elasticsearch.ApiError)
    assert not issubclass(elasticsearch.ApiError, elasticsearch.TransportError)


# --- ConsoleSink ----------------------------------------------------------------------


def test_console_sink_log_line(caplog):
    sink = ConsoleSink()
    with caplog.at_level(logging.WARNING, logger="neuralguard.alerts"):
        sink.emit(alert_doc(threat_score=0.9731))
    (record,) = caplog.records
    assert record.name == "neuralguard.alerts"
    assert record.levelno == logging.WARNING
    assert record.getMessage() == (
        "ALERT [high] port_scan score=0.97 203.0.113.5:40000 -> 192.168.1.20:22 TCP S "
        "(+12 suppressed)"
    )
    sink.flush()
    sink.flush_if_due()
    sink.close()
    assert sink.emitted == 1


def test_console_line_without_ports_flags_or_suppression():
    line = format_alert_line(
        alert_doc(
            severity="critical",
            attack_type="icmp_flood",
            threat_score=0.995,
            source_ip="10.0.0.1",
            destination_ip="10.0.0.2",
            protocol="ICMP",
            source_port=0,
            destination_port=0,
            tcp_flags="",
            suppressed_count=0,
        )
    )
    assert line == "ALERT [critical] icmp_flood score=0.99 10.0.0.1 -> 10.0.0.2 ICMP"


def test_console_line_ipv6_and_unknown_source():
    line = format_alert_line(
        alert_doc(source_ip=None, destination_ip="2001:db8::1", suppressed_count=0)
    )
    assert "?:40000 -> [2001:db8::1]:22 TCP S" in line


def test_console_sink_with_stream(caplog):
    stream = io.StringIO()
    sink = ConsoleSink(stream=stream)
    with caplog.at_level(logging.WARNING, logger="neuralguard.alerts"):
        sink.emit(alert_doc())
        sink.emit(alert_doc(suppressed_count=0))
    sink.close()
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("ALERT [high] port_scan score=0.97")
    assert lines[0].endswith("(+12 suppressed)")
    assert caplog.records == []


def test_console_sink_survives_closed_stream(caplog):
    stream = io.StringIO()
    stream.close()
    sink = ConsoleSink(stream=stream)
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        sink.emit(alert_doc())
        sink.flush()
    assert "cannot write alert" in caplog.text


# --- JsonlSink ------------------------------------------------------------------------


def test_jsonl_sink_writes_one_document_per_line(tmp_path):
    path = tmp_path / "out" / "alerts.jsonl"
    sink = JsonlSink(path)
    first, second = alert_doc(), alert_doc(source_ip="198.51.100.1", simulated_label="normal")
    sink.emit(first)
    sink.emit(second)
    sink.flush()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]
    sink.close()
    sink.close()  # idempotent
    assert sink.written == 2


def test_jsonl_sink_flush_makes_lines_visible(tmp_path):
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path)
    sink.emit(doc(1))
    sink.flush_if_due()
    assert json.loads(path.read_text(encoding="utf-8")) == doc(1)
    sink.close()


def test_jsonl_sink_appends(tmp_path):
    path = tmp_path / "alerts.jsonl"
    for n in range(2):
        sink = JsonlSink(path)
        sink.emit(doc(n))
        sink.close()
    assert [json.loads(line)["n"] for line in path.read_text().splitlines()] == [0, 1]


def test_jsonl_sink_skips_unserialisable_and_closed(tmp_path, caplog):
    path = tmp_path / "alerts.jsonl"
    sink = JsonlSink(path)
    with caplog.at_level(logging.ERROR, logger="neuralguard.sinks"):
        sink.emit({"bad": object()})
        sink.emit(doc())
        sink.close()
        sink.emit(doc(2))  # after close: logged, not raised
    assert [json.loads(line) for line in path.read_text().splitlines()] == [doc()]
    assert "cannot serialise" in caplog.text
    assert "closed" in caplog.text
