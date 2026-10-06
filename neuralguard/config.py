"""Runtime configuration, loaded from ``NEURALGUARD_*`` environment variables.

Every setting has a sensible local-development default, so the whole stack runs
out of the box against ``docker compose up``. CLI flags override these values.

Elasticsearch credentials belong in ``NEURALGUARD_ES_USERNAME`` /
``NEURALGUARD_ES_PASSWORD`` or ``NEURALGUARD_ES_API_KEY``, not in a host URL
(``https://user:password@host``); a password in a URL is masked wherever NeuralGuard
shows the URL (see :func:`redact_credentials`), but other tools may print it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

ENV_PREFIX = "NEURALGUARD_"
_URL_CREDENTIALS = re.compile(r"(?<=://)\S+@")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    """Raised when a setting has an invalid value."""


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    # Kafka
    kafka_bootstrap_servers: tuple[str, ...] = ("localhost:9092",)
    kafka_topic: str = "network-traffic"
    kafka_group_id: str = "neuralguard-detector"
    # Elasticsearch
    es_hosts: tuple[str, ...] = ("http://localhost:9200",)
    es_index: str = "threat-detection"
    es_username: str | None = None
    es_password: str | None = field(default=None, repr=False)
    es_api_key: str | None = field(default=None, repr=False)
    es_verify_certs: bool = True
    es_ca_certs: str | None = None
    # Detection
    model_path: Path = Path("models/threat_model.joblib")
    threat_threshold: float = 0.5
    window_seconds: float = 10.0
    alert_cooldown_seconds: float = 5.0
    # Alert only once a (attack type, target) has this many threat detections within
    # alert_corroboration_seconds; 1 = alert on the first detection.
    alert_min_hits: int = 3
    alert_corroboration_seconds: float = 30.0
    # Live detection rejects records dated further ahead of this host's clock; 0 = no check.
    max_clock_skew_seconds: float = 300.0
    # Logging
    log_level: str = "INFO"
    log_format: str = "text"

    def __post_init__(self) -> None:
        if not self.kafka_bootstrap_servers:
            raise ConfigError("kafka_bootstrap_servers must not be empty")
        if not self.kafka_topic:
            raise ConfigError("kafka_topic must not be empty")
        if not self.es_hosts:
            raise ConfigError("es_hosts must not be empty")
        if not self.es_index:
            raise ConfigError("es_index must not be empty")
        if not 0.0 < self.threat_threshold <= 1.0:
            raise ConfigError(
                f"threat_threshold must be in (0, 1], got {self.threat_threshold}"
            )
        if self.window_seconds <= 0:
            raise ConfigError(f"window_seconds must be > 0, got {self.window_seconds}")
        if self.alert_cooldown_seconds < 0:
            raise ConfigError(
                f"alert_cooldown_seconds must be >= 0, got {self.alert_cooldown_seconds}"
            )
        if isinstance(self.alert_min_hits, bool) or not isinstance(self.alert_min_hits, int):
            raise ConfigError(f"alert_min_hits must be an integer, got {self.alert_min_hits!r}")
        if self.alert_min_hits < 1:
            raise ConfigError(f"alert_min_hits must be >= 1, got {self.alert_min_hits}")
        if not 0 < self.alert_corroboration_seconds < float("inf"):  # also rejects NaN
            raise ConfigError(
                "alert_corroboration_seconds must be a number > 0, "
                f"got {self.alert_corroboration_seconds}"
            )
        if not 0 <= self.max_clock_skew_seconds < float("inf"):  # also rejects NaN
            raise ConfigError(
                f"max_clock_skew_seconds must be a number >= 0, got {self.max_clock_skew_seconds}"
            )
        if self.log_format not in ("text", "json"):
            raise ConfigError(f"log_format must be 'text' or 'json', got {self.log_format!r}")
        if self.es_api_key and (self.es_username or self.es_password):
            raise ConfigError("set either es_api_key or es_username/es_password, not both")
        if bool(self.es_username) != bool(self.es_password):
            raise ConfigError("es_username and es_password must be set together")

    @property
    def kafka_endpoints(self) -> tuple[tuple[str, int], ...]:
        """``(host, port)`` of each Kafka bootstrap server (port 9092 when not given)."""
        return _endpoints(self.kafka_bootstrap_servers, lambda scheme: 9092)

    @property
    def es_endpoints(self) -> tuple[tuple[str, int], ...]:
        """``(host, port)`` of each Elasticsearch host. Without a port: 443 for https (as
        the client does), else 9200."""
        return _endpoints(self.es_hosts, lambda scheme: 443 if scheme == "https" else 9200)

    @property
    def kafka_ports(self) -> tuple[int, ...]:
        """TCP ports of the Kafka bootstrap servers, each once."""
        return tuple(dict.fromkeys(port for _, port in self.kafka_endpoints))

    @property
    def es_ports(self) -> tuple[int, ...]:
        """TCP ports of the Elasticsearch hosts, each once."""
        return tuple(dict.fromkeys(port for _, port in self.es_endpoints))

    def __repr__(self) -> str:
        # The dataclass repr, except that credentials in an Elasticsearch URL are masked.
        values = {f.name: getattr(self, f.name) for f in fields(self) if f.repr}
        values["es_hosts"] = tuple(redact_credentials(host) for host in self.es_hosts)
        shown = ", ".join(f"{name}={value!r}" for name, value in values.items())
        return f"{type(self).__name__}({shown})"

    def with_overrides(self, **overrides: object) -> Settings:
        """Return a copy with every non-``None`` override applied (CLI flags use this)."""
        changes = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **changes) if changes else self

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def get(name: str) -> str | None:
            value = env.get(ENV_PREFIX + name)
            return value.strip() if value is not None and value.strip() else None

        def get_float(name: str) -> float | None:
            raw = get(name)
            if raw is None:
                return None
            try:
                return float(raw)
            except ValueError:
                raise ConfigError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from None

        def get_int(name: str) -> int | None:
            raw = get(name)
            if raw is None:
                return None
            try:
                return int(raw)
            except ValueError:
                raise ConfigError(f"{ENV_PREFIX}{name} must be an integer, got {raw!r}") from None

        def get_bool(name: str) -> bool | None:
            raw = get(name)
            if raw is None:
                return None
            if raw.lower() in _TRUE:
                return True
            if raw.lower() in _FALSE:
                return False
            raise ConfigError(f"{ENV_PREFIX}{name} must be true/false, got {raw!r}")

        values: dict[str, object] = {}
        if (raw := get("KAFKA_BOOTSTRAP_SERVERS")) is not None:
            values["kafka_bootstrap_servers"] = _split_csv(raw)
        if (raw := get("KAFKA_TOPIC")) is not None:
            values["kafka_topic"] = raw
        if (raw := get("KAFKA_GROUP_ID")) is not None:
            values["kafka_group_id"] = raw
        if (raw := get("ES_HOSTS")) is not None:
            values["es_hosts"] = _split_csv(raw)
        if (raw := get("ES_INDEX")) is not None:
            values["es_index"] = raw
        values["es_username"] = get("ES_USERNAME")
        values["es_password"] = get("ES_PASSWORD")
        values["es_api_key"] = get("ES_API_KEY")
        values["es_verify_certs"] = get_bool("ES_VERIFY_CERTS")
        values["es_ca_certs"] = get("ES_CA_CERTS")
        if (raw := get("MODEL_PATH")) is not None:
            values["model_path"] = Path(raw)
        values["threat_threshold"] = get_float("THREAT_THRESHOLD")
        values["window_seconds"] = get_float("WINDOW_SECONDS")
        values["alert_cooldown_seconds"] = get_float("ALERT_COOLDOWN_SECONDS")
        values["alert_min_hits"] = get_int("ALERT_MIN_HITS")
        values["alert_corroboration_seconds"] = get_float("ALERT_CORROBORATION_SECONDS")
        values["max_clock_skew_seconds"] = get_float("MAX_CLOCK_SKEW_SECONDS")
        if (raw := get("LOG_LEVEL")) is not None:
            values["log_level"] = raw.upper()
        if (raw := get("LOG_FORMAT")) is not None:
            values["log_format"] = raw.lower()

        return cls().with_overrides(**values)


def redact_credentials(text: str) -> str:
    """``text`` with the ``user:password@`` part of every URL in it masked as ``***@``."""
    return _URL_CREDENTIALS.sub("***@", text)


def _endpoints(
    addresses: tuple[str, ...], default_port: Callable[[str], int]
) -> tuple[tuple[str, int], ...]:
    """``(host, port)`` of ``[scheme://][user:password@]host[:port][/path]`` addresses, each
    once; ``default_port(scheme)`` gives the port of an address without one."""
    endpoints: list[tuple[str, int]] = []
    for address in addresses:
        scheme, separator, rest = address.partition("://")
        if not separator:
            scheme, rest = "", address
        host = rest.split("/", 1)[0].rsplit("@", 1)[-1]
        port = default_port(scheme.lower())
        if host.startswith("["):  # [IPv6]:port
            host, _, tail = host[1:].partition("]")
            if tail.startswith(":") and tail[1:].isdigit():
                port = int(tail[1:])
        elif host.count(":") == 1:
            host, _, raw_port = host.partition(":")
            if raw_port.isdigit():
                port = int(raw_port)
        if (host, port) not in endpoints:
            endpoints.append((host, port))
    return tuple(endpoints)
