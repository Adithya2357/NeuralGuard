import json
import logging

import pytest

from neuralguard.logutil import JsonFormatter, configure_logging

THIRD_PARTY = ("kafka", "elastic_transport", "elasticsearch", "scapy.runtime")


@pytest.fixture(autouse=True)
def restore_logging():
    """configure_logging changes process-wide state: put it back after each test."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    levels = {name: logging.getLogger(name).level for name in THIRD_PARTY}
    yield
    root.handlers[:], root.level = saved[0], saved[1]
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def test_configure_logging_sets_level_and_quiets_third_party_clients():
    configure_logging("debug", "text")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert logging.getLogger("kafka").level == logging.WARNING
    assert logging.getLogger("scapy.runtime").level == logging.WARNING
    # Its per-retry tracebacks would bypass ElasticsearchSink's rate-limited outage log.
    assert logging.getLogger("elastic_transport").level == logging.ERROR


def test_unknown_level_falls_back_to_info():
    configure_logging("chatty", "text")
    assert logging.getLogger().level == logging.INFO


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord(
        "neuralguard.x", logging.INFO, __file__, 1, "hi %s", ("there",), None
    )
    record.processed = 3
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hi there"
    assert payload["level"] == "INFO" and payload["logger"] == "neuralguard.x"
    assert payload["processed"] == 3
    assert payload["time"].endswith("+00:00")
