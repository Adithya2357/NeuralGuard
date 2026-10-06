"""Small kafka-python helpers shared by the producer and the detection service.

kafka-python is imported lazily (inside the functions), so importing this module stays
cheap for commands that never talk to Kafka.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

# How long one connection attempt waits for the first cluster metadata. kafka-python 3.x
# waits 30 s by default before raising KafkaTimeoutError, which stretched the retry loops
# of create_kafka_consumer / create_kafka_producer (11 attempts) to about six minutes;
# 2.x gave up after about 2 s and has no such setting.
BOOTSTRAP_TIMEOUT_MS = 5000

# The errors kafka-python raises while no bootstrap server answers yet (as is usual for a
# few seconds under ``docker compose up``). 2.x raises NoBrokersAvailable; 3.x raises
# KafkaTimeoutError ("Unable to bootstrap from ...") or KafkaConnectionError instead. They
# are resolved by name because no single version defines all of them.
RETRYABLE_KAFKA_ERROR_NAMES = ("NoBrokersAvailable", "KafkaTimeoutError", "KafkaConnectionError")


def retryable_kafka_errors() -> tuple[type[Exception], ...]:
    """The ``kafka.errors`` classes that mean "no broker reachable yet, try again later".

    Only the classes the installed kafka-python version defines are returned, so the
    tuple can be used directly in an ``except`` clause.
    """
    from kafka import errors

    found = (getattr(errors, name, None) for name in RETRYABLE_KAFKA_ERROR_NAMES)
    return tuple(cls for cls in found if isinstance(cls, type) and issubclass(cls, Exception))


def supported_options(client_class: Any, options: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of ``options`` that ``client_class`` knows.

    ``client_class`` is ``kafka.KafkaConsumer`` / ``kafka.KafkaProducer`` (or a test
    double). kafka-python rejects settings it does not know, and some exist only in 3.x,
    so only settings the class's ``DEFAULT_CONFIG`` declares are kept.
    """
    defaults = getattr(client_class, "DEFAULT_CONFIG", None)
    if not isinstance(defaults, Mapping):
        return {}
    return {name: value for name, value in options.items() if name in defaults}


def bootstrap_timeout_options(
    client_class: Any, timeout_ms: int = BOOTSTRAP_TIMEOUT_MS
) -> dict[str, int]:
    """``{"bootstrap_timeout_ms": timeout_ms}`` if ``client_class`` supports that setting."""
    return supported_options(client_class, {"bootstrap_timeout_ms": int(timeout_ms)})


def wait_before_retry(
    seconds: float, stop_event: threading.Event | None, sleep: Callable[[float], None]
) -> bool:
    """Wait ``seconds`` before the next connection attempt; ``False`` (give up) when
    ``stop_event`` is set, also during the wait, so a stop request is not ignored for the
    minute or so that the connection retries can take."""
    if stop_event is None:
        sleep(seconds)
        return True
    return not stop_event.wait(seconds)
