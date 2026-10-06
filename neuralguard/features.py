"""Feature extraction: turns canonical traffic records into model input vectors.

There are two kinds of features:

* **Packet features** describe a single packet (protocol, size, ports, TCP flags).
* **Window features** describe recent behaviour around it, over a sliding time window:
  how many packets this source sent, how many distinct ports/hosts it touched, how
  many distinct sources are hitting this destination, and so on. These are what make
  port scans and floods visible - a single scan packet looks perfectly innocent.

The same :class:`FeatureExtractor` is used for training and for live detection, so
the model always sees features computed exactly the same way (no train/serve skew).
Records must be fed in time order; the extractor is stateful.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_MAX_TRACKED_HOSTS = 50_000

TCP_FLAG_FEATURES = (
    ("S", "flag_syn"),
    ("A", "flag_ack"),
    ("F", "flag_fin"),
    ("R", "flag_rst"),
    ("P", "flag_psh"),
    ("U", "flag_urg"),
)

PACKET_FEATURES = (
    "is_tcp",
    "is_udp",
    "is_icmp",
    "is_arp",
    "length",
    "ttl",
    "source_port",
    "destination_port",
    "dst_port_well_known",
    *(name for _, name in TCP_FLAG_FEATURES),
)

WINDOW_FEATURES = (
    "src_packet_count",
    "src_unique_dst_ports",
    "src_unique_dst_ips",
    "src_syn_ratio",
    "dst_packet_count",
    "dst_unique_src_ips",
)

FEATURE_NAMES: tuple[str, ...] = (*PACKET_FEATURES, *WINDOW_FEATURES)


def is_pure_syn(record: Mapping[str, Any]) -> bool:
    """A connection-opening SYN (SYN set, ACK not set) - the building block of scans/floods."""
    flags = record.get("tcp_flags") or ""
    return record.get("protocol") == "TCP" and "S" in flags and "A" not in flags


def packet_features(record: Mapping[str, Any]) -> list[float]:
    """Stateless per-packet features, in ``PACKET_FEATURES`` order."""
    protocol = record["protocol"]
    flags = record["tcp_flags"]
    dst_port = record["destination_port"]
    return [
        float(protocol == "TCP"),
        float(protocol == "UDP"),
        float(protocol == "ICMP"),
        float(protocol == "ARP"),
        float(record["length"]),
        float(record["ttl"]),
        float(record["source_port"]),
        float(dst_port),
        float(0 < dst_port < 1024),
        *(float(letter in flags) for letter, _ in TCP_FLAG_FEATURES),
    ]


class _SourceWindow:
    """Recent packets sent *by* one host."""

    __slots__ = ("dst_ips", "dst_ports", "events", "syn_count")

    def __init__(self) -> None:
        self.events: deque[tuple[float, int, str | None, bool]] = deque()
        self.dst_ports: Counter[int] = Counter()
        self.dst_ips: Counter[str | None] = Counter()
        self.syn_count = 0

    def add(self, ts: float, dst_port: int, dst_ip: str | None, syn: bool) -> None:
        self.events.append((ts, dst_port, dst_ip, syn))
        self.dst_ports[dst_port] += 1
        self.dst_ips[dst_ip] += 1
        self.syn_count += syn

    def evict(self, cutoff: float) -> None:
        events = self.events
        while events and events[0][0] <= cutoff:
            _, dst_port, dst_ip, syn = events.popleft()
            _decrement(self.dst_ports, dst_port)
            _decrement(self.dst_ips, dst_ip)
            self.syn_count -= syn


class _DestinationWindow:
    """Recent packets sent *to* one host."""

    __slots__ = ("events", "src_ips")

    def __init__(self) -> None:
        self.events: deque[tuple[float, str | None]] = deque()
        self.src_ips: Counter[str | None] = Counter()

    def add(self, ts: float, src_ip: str | None) -> None:
        self.events.append((ts, src_ip))
        self.src_ips[src_ip] += 1

    def evict(self, cutoff: float) -> None:
        events = self.events
        while events and events[0][0] <= cutoff:
            _, src_ip = events.popleft()
            _decrement(self.src_ips, src_ip)


def _decrement(counter: Counter, key: Any) -> None:
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]


class FeatureExtractor:
    """Stateful sliding-window feature extractor.

    ``window_seconds`` is the look-back window. ``max_tracked_hosts`` bounds memory:
    with spoofed-source floods every packet can come from a new IP, so the least
    recently seen hosts are forgotten once the limit is hit.

    Time is driven by record timestamps (not the wall clock), so results are
    deterministic and identical between training and live detection. A record that
    arrives out of order is treated as if it happened at the latest time seen so far.
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_tracked_hosts: int = DEFAULT_MAX_TRACKED_HOSTS,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if max_tracked_hosts < 1:
            raise ValueError(f"max_tracked_hosts must be >= 1, got {max_tracked_hosts}")
        self.window_seconds = float(window_seconds)
        self.max_tracked_hosts = int(max_tracked_hosts)
        self.reset()

    def reset(self) -> None:
        self._clock = float("-inf")
        self._sources: OrderedDict[str | None, _SourceWindow] = OrderedDict()
        self._destinations: OrderedDict[str | None, _DestinationWindow] = OrderedDict()

    @property
    def tracked_hosts(self) -> int:
        return len(self._sources) + len(self._destinations)

    def transform_one(self, record: Mapping[str, Any]) -> np.ndarray:
        """Feature vector for one canonical record (see ``schema.normalize_record``).

        Updates the window state *before* computing window features, so every count
        includes the current packet.
        """
        ts = max(float(record["timestamp"]), self._clock)
        self._clock = ts
        cutoff = ts - self.window_seconds
        src_ip = record["source_ip"]
        dst_ip = record["destination_ip"]

        source = self._touch(self._sources, src_ip, _SourceWindow)
        source.evict(cutoff)
        source.add(ts, record["destination_port"], dst_ip, is_pure_syn(record))

        destination = self._touch(self._destinations, dst_ip, _DestinationWindow)
        destination.evict(cutoff)
        destination.add(ts, src_ip)

        src_count = len(source.events)
        window = [
            float(src_count),
            float(len(source.dst_ports)),
            float(len(source.dst_ips)),
            source.syn_count / src_count,
            float(len(destination.events)),
            float(len(destination.src_ips)),
        ]
        return np.asarray(packet_features(record) + window, dtype=np.float64)

    def transform(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        """Feature matrix for records in time order, shape ``(n, len(FEATURE_NAMES))``."""
        rows = [self.transform_one(record) for record in records]
        if not rows:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        return np.vstack(rows)

    def _touch(self, table: OrderedDict, key: str | None, factory: type) -> Any:
        window = table.get(key)
        if window is None:
            window = factory()
            table[key] = window
            if len(table) > self.max_tracked_hosts:
                table.popitem(last=False)
        else:
            table.move_to_end(key)
        return window


def features_as_dict(vector: np.ndarray) -> dict[str, float]:
    """Name each value of a feature vector (handy for logging and alert documents)."""
    return {name: float(value) for name, value in zip(FEATURE_NAMES, vector, strict=True)}
