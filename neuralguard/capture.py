"""Packet capture: converts scapy packets into traffic records (see ``schema.py``).

CONTRACT (implementation TODO). Import scapy lazily inside functions, never at module
import time (it is slow, noisy, and not needed by the detector or the trainer).

* ``parse_packet(packet) -> dict | None``
  - IPv4 (``IP``) and IPv6 (``IPv6``) packets: src/dst IP, ttl / hop limit, protocol
    TCP/UDP/ICMP (ICMPv6 counts as ICMP) else OTHER, ports for TCP/UDP, canonical TCP
    flag string via ``str(packet[TCP].flags)`` then canonicalised (see ``schema``),
    ``length = len(packet)``, ``timestamp = float(packet.time)``.
  - ARP: protocol "ARP", ``psrc``/``pdst`` as IPs, ports/ttl 0.
  - Anything else (no IP/ARP layer): return None (skip it - it carries no useful signal).
  - The result must pass ``normalize_record`` unchanged; run it through
    ``normalize_record`` before returning and return None (log at DEBUG) if invalid.
* ``default_bpf_filter(settings) -> str``: excludes NeuralGuard's own traffic so the
  sniffer never captures its own Kafka/Elasticsearch connections (a feedback loop):
  ``"not (tcp port 9092 or tcp port 9200)"`` built from ``settings.kafka_ports`` and
  ``settings.es_ports``.
* ``live_records(interface=None, bpf_filter=None, count=0, stop_event=None)``
  -> iterator of records from a live interface. Use ``scapy.sendrecv.AsyncSniffer``
  feeding a ``queue.Queue`` (``store=False``); yield parsed records; stop when
  ``count`` records were yielded (0 = unlimited) or ``stop_event`` is set; always stop
  the sniffer in a ``finally``. Raise ``CaptureError`` with a helpful message (needs
  root / CAP_NET_RAW, e.g. ``sudo``) on PermissionError / OSError from scapy.
* ``pcap_records(path)`` -> iterator of records read with ``scapy.utils.PcapReader``
  (supports .pcap and .pcapng); ``CaptureError`` if the file is missing/unreadable.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from neuralguard.config import Settings


class CaptureError(RuntimeError):
    """Raised when packets cannot be captured or read."""


def parse_packet(packet: Any) -> dict[str, Any] | None:
    raise NotImplementedError


def default_bpf_filter(settings: Settings) -> str:
    raise NotImplementedError


def live_records(
    interface: str | None = None,
    bpf_filter: str | None = None,
    count: int = 0,
    stop_event: threading.Event | None = None,
) -> Iterator[dict[str, Any]]:
    raise NotImplementedError


def pcap_records(path: str | Path) -> Iterator[dict[str, Any]]:
    raise NotImplementedError
