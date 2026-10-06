"""Runtime configuration, loaded from ``NEURALGUARD_*`` environment variables.

Every setting has a sensible local-development default, so the whole stack runs
out of the box against ``docker compose up``. CLI flags override these values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

ENV_PREFIX = "NEURALGUARD_"

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
        if self.log_format not in ("text", "json"):
            raise ConfigError(f"log_format must be 'text' or 'json', got {self.log_format!r}")
        if self.es_api_key and (self.es_username or self.es_password):
            raise ConfigError("set either es_api_key or es_username/es_password, not both")
        if bool(self.es_username) != bool(self.es_password):
            raise ConfigError("es_username and es_password must be set together")

    @property
    def kafka_ports(self) -> tuple[int, ...]:
        """TCP ports of the Kafka bootstrap servers (used to keep the sniffer off them)."""
        return _ports(self.kafka_bootstrap_servers, default=9092)

    @property
    def es_ports(self) -> tuple[int, ...]:
        """TCP ports of the Elasticsearch hosts."""
        return _ports(self.es_hosts, default=9200)

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
        if (raw := get("LOG_LEVEL")) is not None:
            values["log_level"] = raw.upper()
        if (raw := get("LOG_FORMAT")) is not None:
            values["log_format"] = raw.lower()

        return cls().with_overrides(**values)


def _ports(addresses: tuple[str, ...], default: int) -> tuple[int, ...]:
    ports: list[int] = []
    for address in addresses:
        rest = address.split("://", 1)[-1].split("/", 1)[0]
        host_port = rest.rsplit("@", 1)[-1]
        port = default
        if host_port.startswith("["):  # [IPv6]:port
            _, _, tail = host_port.partition("]")
            if tail.startswith(":") and tail[1:].isdigit():
                port = int(tail[1:])
        elif host_port.count(":") == 1:
            _, _, raw_port = host_port.partition(":")
            if raw_port.isdigit():
                port = int(raw_port)
        if port not in ports:
            ports.append(port)
    return tuple(ports)
