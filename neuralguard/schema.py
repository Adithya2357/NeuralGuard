"""The traffic-record schema shared by the producer, the trainer and the detector.

A *traffic record* is one observed packet, as a JSON-serialisable dict::

    {
        "timestamp": 1730647800.123,      # epoch seconds (float, packet capture time)
        "source_ip": "192.168.1.10",      # IPv4/IPv6 string, or None when unknown
        "destination_ip": "10.0.0.3",
        "protocol": "TCP",                # one of PROTOCOLS
        "source_port": 51514,             # 0 when the protocol has no ports
        "destination_port": 443,
        "length": 60,                     # bytes on the wire
        "ttl": 64,                        # IPv4 TTL / IPv6 hop limit, 0 when unknown
        "tcp_flags": "S",                 # scapy-style flag letters, "" for non-TCP
        "label": "port_scan",             # OPTIONAL ground truth (simulator / datasets)
    }

``normalize_record`` is the single gate every record passes through before feature
extraction, so the producer, the trainer and the detector can never disagree about
what a field means.
"""

from __future__ import annotations

import ipaddress
import math
import time
from collections.abc import Mapping
from typing import Any

PROTOCOLS = ("TCP", "UDP", "ICMP", "ARP", "OTHER")

NORMAL_LABEL = "normal"
ATTACK_TYPES = ("port_scan", "stealth_scan", "syn_flood", "udp_flood", "icmp_flood")
LABELS = (NORMAL_LABEL, *ATTACK_TYPES)

# Scapy's TCP flag letters: FIN SYN RST PSH ACK URG ECE CWR NS
TCP_FLAG_LETTERS = "FSRPAUECN"

RECORD_FIELDS = (
    "timestamp",
    "source_ip",
    "destination_ip",
    "protocol",
    "source_port",
    "destination_port",
    "length",
    "ttl",
    "tcp_flags",
)

_UNKNOWN_IPS = {"", "unknown", "none", "null", "-"}
_MAX_LENGTH = 1 << 20  # 1 MiB; anything bigger is not a single packet

# Latest plausible capture time: 3000-01-01T00:00:00Z. A later value is corrupt or
# mis-scaled (e.g. milliseconds instead of seconds); accepting it would jump the feature
# extractor's clock, which only ever moves forward, far into the future for good.
MAX_TIMESTAMP = 32_503_680_000.0


class InvalidRecordError(ValueError):
    """Raised when a traffic record is malformed."""


def normalize_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and coerce a raw traffic record into its canonical form.

    Missing optional fields get defaults (ports/ttl/length 0, flags "", IPs None,
    timestamp = now). Unknown protocols map to "OTHER". Anything that cannot be
    coerced raises :class:`InvalidRecordError`, including a timestamp after
    ``MAX_TIMESTAMP`` (the year 3000), which cannot be a real capture time.
    """
    if not isinstance(raw, Mapping):
        raise InvalidRecordError(f"record must be a JSON object, got {type(raw).__name__}")

    protocol = _protocol(raw.get("protocol"))
    record: dict[str, Any] = {
        "timestamp": _timestamp(raw.get("timestamp")),
        "source_ip": _ip(raw.get("source_ip"), "source_ip"),
        "destination_ip": _ip(raw.get("destination_ip"), "destination_ip"),
        "protocol": protocol,
        "source_port": _int(raw.get("source_port"), "source_port", 0, 65535),
        "destination_port": _int(raw.get("destination_port"), "destination_port", 0, 65535),
        "length": _int(raw.get("length"), "length", 0, _MAX_LENGTH),
        "ttl": _int(raw.get("ttl"), "ttl", 0, 255),
        "tcp_flags": _flags(raw.get("tcp_flags")) if protocol == "TCP" else "",
    }
    label = raw.get("label")
    if label is not None:
        if label not in LABELS:
            raise InvalidRecordError(f"label must be one of {LABELS}, got {label!r}")
        record["label"] = label
    return record


def _timestamp(value: Any) -> float:
    if value is None:
        return time.time()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRecordError(f"timestamp must be epoch seconds, got {value!r}")
    try:
        value = float(value)
    except OverflowError:  # JSON integers can have thousands of digits; never echo them
        raise InvalidRecordError(
            "timestamp is not a plausible capture time (an integer too large for a float)"
        ) from None
    if not math.isfinite(value) or value < 0:
        raise InvalidRecordError(f"timestamp must be a finite, non-negative number, got {value}")
    if value > MAX_TIMESTAMP:
        hint = " - milliseconds instead of seconds?" if value / 1000 <= MAX_TIMESTAMP else ""
        raise InvalidRecordError(
            f"timestamp {value:g} is not a plausible capture time (after the year 3000){hint}"
        )
    return value


def _ip(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidRecordError(f"{name} must be a string, got {value!r}")
    if value.strip().lower() in _UNKNOWN_IPS:
        return None
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        raise InvalidRecordError(f"{name} is not a valid IP address: {value!r}") from None
    if getattr(address, "scope_id", None) is not None:
        # A zone id ("fe80::1%eth0") only means something on the host that wrote it and is
        # never part of a captured packet, yet it may hold anything: newlines and terminal
        # escapes (forged log lines) or megabytes of text (memory pinned in every window).
        raise InvalidRecordError(
            f"{name} must not have an IPv6 zone id (after '%'): {value[:60]!r}"
        )
    return str(address)


def _protocol(value: Any) -> str:
    if value is None:
        return "OTHER"
    if not isinstance(value, str):
        raise InvalidRecordError(f"protocol must be a string, got {value!r}")
    upper = value.strip().upper()
    return upper if upper in PROTOCOLS else "OTHER"


def _int(value: Any, name: str, low: int, high: int) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise InvalidRecordError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise InvalidRecordError(f"{name} must be an integer, got {value!r}")
        value = int(value)
    if not isinstance(value, int):
        raise InvalidRecordError(f"{name} must be an integer, got {value!r}")
    if not low <= value <= high:
        raise InvalidRecordError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _flags(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InvalidRecordError(f"tcp_flags must be a string like 'SA', got {value!r}")
    upper = value.strip().upper()
    bad = set(upper) - set(TCP_FLAG_LETTERS)
    if bad:
        raise InvalidRecordError(f"tcp_flags has unknown flag letters {sorted(bad)}: {value!r}")
    # Canonical form: no duplicates, scapy's display order (so SYN+ACK is always "SA").
    return "".join(letter for letter in TCP_FLAG_LETTERS if letter in upper)
