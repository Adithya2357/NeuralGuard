"""Synthetic, labelled network traffic: realistic normal activity plus attack episodes.

Used to (a) train and evaluate the model and (b) demo the full pipeline without root
privileges or a live network. Every record follows the schema in ``schema.py``, is
already canonical (``normalize_record(record) == record``) and carries a ``label``
(``"normal"`` or one of ``ATTACK_TYPES``).

The simulated network is seen from a mirror port on a small office LAN (192.168.1.0/24):
a gateway that also resolves DNS, a web server reachable from the internet, a NAS, a
monitoring box, IoT devices (printer, camera, smart TV) and ~15 Linux / Windows / macOS
workstations, talking to a few dozen external servers. LAN hosts show their stack's
default TTL (64/128), external hosts show the default minus their hop count, and lengths
are Ethernet frame sizes (what ``len(packet)`` gives in ``capture.py``).

**Normal traffic** is a mix of independent Poisson activities whose intensity drifts
over time: web page loads (DNS lookups, then one to six parallel TCP connections, each a
SYN / SYN-ACK / ACK handshake, request/response rounds and a FIN or RST teardown), API
calls, bulk downloads and uploads, inbound connections and occasional flash crowds on the
LAN web server, SSH sessions, NAS file transfers, video streaming, DNS, NTP, mDNS/SSDP/
LLMNR discovery, QUIC, video calls (RTP over UDP), ICMP pings (some with large payloads),
ARP, IGMP, traceroutes, failed connection attempts, a client talking to several ports of
one server (mail, development, Windows services), peer-to-peer fan-out and a monitoring
host that pings every LAN host and checks a few service ports. So legitimate traffic is
full of pure SYNs, pings, UDP to high ports and bursts - no single packet field gives an
attack away; the *behaviour* over a few seconds does.

**Attack episodes** (each one's rate, duration, attacker, target, ports, sizes and TTLs
are drawn from the seeded RNG) are interleaved with normal traffic, which keeps flowing
during attacks. Only the attacker's packets carry the attack label.

* ``port_scan``: one attacker, SYN probes to many distinct ports of one target
  (sequential, shuffled well-known, popular-first or full-range order; raw nmap-style
  probes or connect() scans), from ~4 probes/s up to ~300/s.
* ``stealth_scan``: like ``port_scan`` but NULL (no flags), FIN or XMAS (FPU) probes -
  plus nmap's ACK and Maimon (FIN/ACK) scans, whose probes look like ordinary packets.
* ``syn_flood``: pure SYNs to one target port (80 or 443) at 80-600 pps, from spoofed
  random source IPs or from a handful of real sources.
* ``udp_flood``: UDP to random high ports of one target at 80-600 pps, varied payload
  sizes; from one source, a small botnet, or reflectors (DNS/NTP/SSDP/memcached
  amplification, so the source port looks like an ordinary service reply).
* ``icmp_flood``: ICMP echo to one target at 60-500 pps, small or large payloads, from
  one, a few or spoofed sources.

``attack_ratio`` is held approximately by scheduling the gap before each episode from
the records emitted so far. Above ~0.4 the normal activity is thinned so that high
ratios stay reachable - more strongly while the running attack fraction lags behind (for
example while slow scans hold both attack slots); at 1.0 there is no normal traffic at
all.

Usage::

    simulator = TrafficSimulator(seed=7, attack_ratio=0.3)
    for record in simulator.records(1000):
        ...

``records()`` draws from one continuous stream per simulator, so two calls of
``records(100)`` return the first and the second hundred records.
"""

from __future__ import annotations

import ipaddress
import math
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import product
from typing import Any

from neuralguard.schema import ATTACK_TYPES, MAX_TIMESTAMP, NORMAL_LABEL

__all__ = ["TrafficSimulator", "generate_records"]

# Simulated time generated per scheduling step. Every packet of a session or an attack
# episode is queued when it starts, so each step can emit exactly the queued records
# that fall inside it, in time order.
_STEP_SECONDS = 1.0

# Normal-traffic thinning for high attack ratios: intensity = min(1, K * (1 - r) / r).
_THINNING_K = 0.6
# Extra thinning while the attack fraction lags a ratio that needs thinning at all: each
# normal activity starts with probability (achieved / attack_ratio) ** _LAG_POWER, but at
# least _MIN_KEEP, once _LAG_MIN_RECORDS records have been emitted.
_LAG_POWER = 3.0
_MIN_KEEP = 0.05
_LAG_MIN_RECORDS = 200
_MEAN_EPISODE_RECORDS = 200  # rough average, only used to size the initial warm-up
# At most this many attack episodes run at once (each on its own target). Overlap only
# happens when normal traffic gets ahead of ``attack_ratio`` - e.g. during a slow scan.
_MAX_CONCURRENT_ATTACKS = 2
# A host stays reserved for this long after its episode ends, so a new episode never
# shares an attacker or a victim with one still visible in a 10 s feature window.
_COOLDOWN_SECONDS = 12.0

_FLAG_SYN = "S"

# A pending packet: (timestamp, seq, src, dst, protocol, sport, dport, length, ttl,
# flags, label). ``seq`` makes ordering total and deterministic.
_Pending = tuple[float, int, Any, Any, str, int, int, int, int, str, str]


# --------------------------------------------------------------------------- hosts


@dataclass(frozen=True)
class _Stack:
    """A TCP/IP stack fingerprint: default TTL, typical frame sizes, ephemeral ports."""

    ttl: int
    syn_len: int
    synack_len: int
    ack_len: int
    ping_len: int
    eph_low: int
    eph_high: int


_LINUX = _Stack(64, 74, 74, 66, 98, 32768, 60999)
_WINDOWS = _Stack(128, 66, 66, 54, 74, 49152, 65535)
_MACOS = _Stack(64, 78, 74, 66, 98, 49152, 65535)
_EMBEDDED = _Stack(64, 58, 58, 60, 60, 1024, 4999)  # IoT: MSS option only, padded frames
_ROUTER = _Stack(255, 60, 58, 60, 74, 1024, 65535)
# Internet clients usually sit behind NAT, which maps source ports anywhere in 1024-65535.
_NAT_STACKS = tuple(
    (replace(stack, eph_low=1024, eph_high=65535), share)
    for stack, share in ((_LINUX, 0.3), (_WINDOWS, 0.5), (_MACOS, 0.2))
)


@dataclass(frozen=True)
class _Host:
    ip: str
    stack: _Stack
    ttl: int  # as seen at the capture point: default TTL minus hops
    rtt: float  # round-trip time from the capture point, seconds
    hops: int = 0
    ip6: str | None = None


_LAN = "192.168.1."
_GATEWAY_IP = _LAN + "1"
_MONITOR_IP = _LAN + "5"
_WEB_SERVER_IP = _LAN + "10"
_NAS_IP = _LAN + "20"
_TV_IP = _LAN + "32"

# (host number, stack, has IPv6, browsing weight)
_LAN_CLIENTS = (
    (100, _WINDOWS, True, 3.0),
    (101, _WINDOWS, True, 2.0),
    (102, _WINDOWS, False, 2.0),
    (103, _WINDOWS, True, 1.5),
    (104, _WINDOWS, False, 1.0),
    (105, _WINDOWS, True, 1.0),
    (106, _LINUX, True, 2.5),
    (107, _LINUX, True, 1.5),
    (108, _LINUX, False, 1.0),
    (109, _LINUX, True, 0.7),
    (110, _MACOS, True, 2.0),
    (111, _MACOS, True, 1.5),
    (112, _MACOS, False, 1.0),
    (113, _MACOS, True, 0.8),
    (114, _LINUX, False, 0.5),
)

_TOP_PORTS = (
    80, 443, 22, 21, 23, 25, 53, 110, 111, 135, 139, 143, 445, 993, 995, 1723, 3306,
    3389, 5900, 8080, 8443, 1433, 1521, 5432, 6379, 27017, 9200, 5000, 8000, 8888,
    10000, 49152, 587, 465, 389, 636, 2049, 5060, 6000, 9090, 11211, 179, 161, 162,
)  # fmt: skip
_MAIL_PORTS = (993, 587, 465, 143, 25, 995, 110)
_DEV_PORTS = (22, 80, 443, 3000, 5000, 5432, 6379, 8000, 8080, 8443, 9000, 9200)
_WINDOWS_PORTS = (135, 139, 445, 3389, 5985)
_SCAN_FLAGS = (("", 0.25), ("F", 0.25), ("FPU", 0.2), ("A", 0.15), ("FA", 0.15))
# Reflection/amplification floods: reflector service port -> reply frame sizes.
_REFLECTION = {53: (512, 1514), 123: (482, 490), 1900: (300, 400), 11211: (1400, 1514)}
_UDP_FLOOD_SIZES = {"small": (60, 250), "large": (500, 1514), "random": (60, 1514)}
# (variant, reflector port) combinations, see TrafficSimulator._udp_flood
_UDP_FLOOD_VARIANTS = (
    ("single", ""),
    ("botnet", ""),
    ("botnet", ""),
    *(("reflection", str(port)) for port in _REFLECTION),
)
_PORT_ORDERS = ("sequential", "well_known", "popular", "full")
# First octets used for "public" addresses (no private, loopback, CGNAT, link-local,
# documentation or multicast ranges).
_PUBLIC_FIRST_OCTETS = tuple(
    octet for octet in range(1, 224) if octet not in (10, 100, 127, 169, 172, 192, 198, 203)
)


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _weighted(rng: random.Random, options: Sequence[tuple[Any, float]]) -> Any:
    pick = rng.random() * sum(weight for _, weight in options)
    for value, weight in options:
        pick -= weight
        if pick < 0:
            return value
    return options[-1][0]


# --------------------------------------------------------------------------- simulator


class TrafficSimulator:
    """Seeded generator of labelled traffic records (see the module docstring).

    ``seed``: same seed + same arguments => identical record stream (``None`` = random).
    ``attack_ratio``: approximate fraction of attack records, in ``[0, 1]``.
    ``start_time``: epoch seconds of the first record (default: now).
    ``attack_types``: which of ``ATTACK_TYPES`` to generate.
    """

    def __init__(
        self,
        seed: int | None = None,
        attack_ratio: float = 0.2,
        start_time: float | None = None,
        attack_types: Sequence[str] = ATTACK_TYPES,
    ) -> None:
        self.seed = seed
        self.attack_ratio = _validate_ratio(attack_ratio)
        self.attack_types = _validate_attack_types(attack_types)
        if self.attack_ratio > 0 and not self.attack_types:
            raise ValueError("attack_types must not be empty when attack_ratio > 0")
        self.start_time = _validate_start_time(start_time)

        # Simulation only - this RNG never protects anything.
        self._rng = random.Random(seed)  # noqa: S311  # nosec B311
        self._pending: list[_Pending] = []
        self._seq = 0
        self._normal_count = 0
        self._attack_count = 0

        self._build_network()
        self._normal_scale = _normal_scale(self.attack_ratio)
        self._processes = self._build_processes()
        self._level = 1.0
        self._level_until = self.start_time

        self._attack_bag: list[str] = []
        self._variant_bags: dict[str, list[tuple[str, ...]]] = {}
        self._attack_generated = 0  # attack records queued so far (emitted or pending)
        self._attack_offset = self._warmup_records()
        self._attack_not_before = self.start_time
        self._active_attacks: list[float] = []  # end times of running episodes
        self._reserved: dict[str, float] = {}  # attacker/victim IP -> reserved until
        self._episode_hosts: list[str] = []  # attacker IPs drawn for the current episode
        self._max_concurrent = 1 if self._normal_scale <= 0 else _MAX_CONCURRENT_ATTACKS
        self._attackers = {
            "port_scan": self._port_scan,
            "stealth_scan": self._stealth_scan,
            "syn_flood": self._syn_flood,
            "udp_flood": self._udp_flood,
            "icmp_flood": self._icmp_flood,
        }
        self._stream = self._generate()

    # ------------------------------------------------------------------ public API

    def records(self, count: int | None = None) -> Iterator[dict[str, Any]]:
        """The next ``count`` records of this simulator's stream (infinite if ``None``).

        Records come in non-decreasing ``timestamp`` order and are already canonical.
        """
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise ValueError(f"count must be a non-negative integer or None, got {count!r}")
        return self._take(count)

    def _take(self, count: int | None) -> Iterator[dict[str, Any]]:
        stream = self._stream
        if count is None:
            while True:
                yield next(stream)
        for _ in range(count):
            yield next(stream)

    # ------------------------------------------------------------------ main loop

    def _generate(self) -> Iterator[dict[str, Any]]:
        pending = self._pending
        step_start = self.start_time
        while True:
            step_end = step_start + _STEP_SECONDS
            self._schedule_attack(step_start)
            self._spawn_normal(step_start, step_end)
            while pending and pending[0][0] < step_end:
                ts, _, src, dst, proto, sport, dport, length, ttl, flags, label = heappop(pending)
                if label == NORMAL_LABEL:
                    self._normal_count += 1
                else:
                    self._attack_count += 1
                yield {
                    "timestamp": round(ts, 6),
                    "source_ip": src,
                    "destination_ip": dst,
                    "protocol": proto,
                    "source_port": sport,
                    "destination_port": dport,
                    "length": length,
                    "ttl": ttl,
                    "tcp_flags": flags,
                    "label": label,
                }
            step_start = step_end

    def _emit(
        self,
        ts: float,
        src: str | None,
        dst: str | None,
        proto: str,
        sport: int,
        dport: int,
        length: int,
        ttl: int,
        flags: str = "",
        label: str = NORMAL_LABEL,
    ) -> None:
        self._seq += 1
        heappush(
            self._pending,
            (ts, self._seq, src, dst, proto, sport, dport, length, ttl, flags, label),
        )

    # ------------------------------------------------------------------ attack scheduling

    def _warmup_records(self) -> float:
        """Normal records to let through before the first attack episode."""
        ratio = self.attack_ratio
        if not 0 < ratio < 1:
            return 0.0
        return _MEAN_EPISODE_RECORDS * (1 - ratio) / ratio * self._rng.uniform(0.2, 0.8)

    def _schedule_attack(self, now: float) -> None:
        """Start the next episode once normal traffic has "paid for" the attacks so far.

        Every queued attack record must be matched by ``(1 - r) / r`` emitted normal
        records (plus a random offset per episode, so gaps vary). That keeps the running
        attack fraction near ``attack_ratio`` whatever the episodes' sizes and rates.
        """
        ratio = self.attack_ratio
        if ratio <= 0:
            return
        self._active_attacks = [end for end in self._active_attacks if end > now]
        if len(self._active_attacks) >= self._max_concurrent or now < self._attack_not_before:
            return
        owed = self._attack_generated * (1 - ratio) / ratio + self._attack_offset
        if self._normal_count < owed:
            return
        rng = self._rng
        if not self._attack_bag:
            self._attack_bag = list(self.attack_types)
            rng.shuffle(self._attack_bag)
        kind = self._attack_bag.pop()
        start = now + rng.random() * _STEP_SECONDS
        self._reserved = {ip: until for ip, until in self._reserved.items() if until > now}
        target = self._target(kind)
        self._episode_hosts = [target.ip]
        queued_before = self._seq
        end = self._attackers[kind](start, target.ip)
        size = self._seq - queued_before
        self._attack_generated += size
        self._active_attacks.append(end)
        for ip in self._episode_hosts:
            self._reserved[ip] = end + _COOLDOWN_SECONDS
        self._attack_offset = rng.uniform(-0.4, 0.4) * size * (1 - ratio) / ratio
        self._attack_not_before = start + rng.uniform(1.0, 4.0)

    def _variant(self, kind: str, *dimensions: Sequence[str]) -> tuple[str, ...]:
        """Rotate through every combination of an attack type's variant dimensions."""
        return self._combination(kind, list(product(*dimensions)))

    def _combination(self, kind: str, combinations: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
        """Draw from a shuffled bag of ``combinations``, refilled when empty.

        Drawing without replacement makes every variant appear regularly, so a model
        trained on a few dozen episodes has seen each of them.
        """
        bag = self._variant_bags.setdefault(kind, [])
        if not bag:
            bag.extend(combinations)
            self._rng.shuffle(bag)
        return bag.pop()

    def _attacker(self) -> tuple[str, _Stack, int]:
        """IP, stack and observed TTL of an attacking host: external or a rogue LAN device.

        Never a host another recent episode used (see ``_COOLDOWN_SECONDS``).
        """
        rng = self._rng
        stack = _LINUX if rng.random() < 0.75 else _WINDOWS
        external = rng.random() < 0.7
        for _ in range(50):
            ip = self._public_ip() if external else _LAN + str(rng.randint(200, 254))
            if ip not in self._reserved and ip not in self._episode_hosts:
                break
        self._episode_hosts.append(ip)
        return ip, stack, stack.ttl - rng.randint(5, 25) if external else stack.ttl

    def _target(self, kind: str) -> _Host:
        """A LAN victim that no running episode is already attacking."""
        rng = self._rng
        busy = self._reserved
        web_share = 0.6 if kind == "syn_flood" else 0.25
        host = self._web_server
        for _ in range(20):
            pick = rng.random()
            if pick < web_share:
                host = self._web_server
            elif pick < web_share + 0.15:
                host = self._nas
            elif pick < web_share + 0.25:
                host = self._gateway
            else:
                host = rng.choice(self._lan_hosts)
            if host.ip not in busy:
                break
        return host

    # ------------------------------------------------------------------ attacks

    def _episode_records(
        self, rate: float, low: float, high: float, min_seconds: float, max_seconds: float
    ) -> int:
        """Size of an episode at ``rate``: ~``low``..``high`` records, duration clamped."""
        duration = _log_uniform(self._rng, low, high) / rate
        return max(15, int(rate * min(max_seconds, max(min_seconds, duration))))

    def _scan_ports(self, n: int, order: str) -> list[int]:
        rng = self._rng
        if order == "sequential":  # like nmap -r or a naive scanner
            first = rng.choice((1, 1, 20, 1000, rng.randint(1, 60000)))
            return [(first - 1 + i) % 65535 + 1 for i in range(n)]
        if order == "well_known":  # the well-known range, shuffled
            pool = list(range(1, 1025))
            rng.shuffle(pool)
        elif order == "popular":  # popular ports first (nmap's default), then anything
            pool = list(_TOP_PORTS)
            rng.shuffle(pool)
            pool += rng.sample(range(1025, 65536), max(0, min(n, 5000) - len(pool)))
        else:  # the full range in random order
            pool = rng.sample(range(1, 65536), min(n, 65535))
        while len(pool) < n:  # more probes than ports: a second pass (retries)
            pool += pool[: n - len(pool)]
        return pool[:n]

    def _probe_gaps(self, n: int, rate: float) -> Iterator[float]:
        rng = self._rng
        mode = rng.random()
        if mode < 0.4:  # steady
            for _ in range(n):
                yield rng.uniform(0.7, 1.3) / rate
        elif mode < 0.75:  # Poisson
            for _ in range(n):
                yield rng.expovariate(rate)
        else:  # bursts of probes, then a pause (same average rate)
            burst = rng.randint(5, 40)
            for i in range(n):
                if i % burst == burst - 1:
                    yield burst * 0.8 / rate
                else:
                    yield 0.2 / rate * rng.uniform(0.5, 1.5)

    def _scan(self, start: float, target: str, label: str) -> float:
        rng = self._rng
        attacker, stack, ttl = self._attacker()
        rate = _log_uniform(rng, 4.0, 300.0)
        n = self._episode_records(rate, 50.0, 400.0, 3.0, 30.0)
        if label == "stealth_scan":  # NULL, FIN, XMAS, ACK (nmap -sA), Maimon (nmap -sM)
            (flags,) = self._variant(label, ("", "F", "FPU", "A", "FA"))
            order = rng.choice(_PORT_ORDERS)
            length = rng.choice((54, 60))
            technique = "raw"
        else:
            flags = _FLAG_SYN
            technique, order = self._variant(label, ("raw", "connect"), _PORT_ORDERS)
            # connect() scan: the OS's own SYN; raw SYN scan (nmap -sS, masscan): minimal
            # options (just MSS, or none)
            length = stack.syn_len if technique == "connect" else rng.choice((58, 58, 54, 60))
        sports = self._scan_source_ports(stack, technique)
        emit = self._emit
        ts = start
        ports = self._scan_ports(n, order)
        for port, gap in zip(ports, self._probe_gaps(n, rate), strict=True):
            emit(ts, attacker, target, "TCP", next(sports), port, length, ttl, flags, label)
            ts += gap
        return ts

    def _scan_source_ports(self, stack: _Stack, technique: str) -> Iterator[int]:
        """Probe source ports: consecutive ephemeral ports for connect() scans, random
        per probe for raw scans."""
        rng = self._rng
        if technique == "connect":
            port = rng.randint(stack.eph_low, stack.eph_high)
            while True:
                yield port
                port = port + 1 if port < stack.eph_high else stack.eph_low
        while True:
            yield rng.randint(1024, 65535)

    def _port_scan(self, start: float, target: str) -> float:
        return self._scan(start, target, "port_scan")

    def _stealth_scan(self, start: float, target: str) -> float:
        return self._scan(start, target, "stealth_scan")

    def _flood_records(self, rate: float) -> int:
        return self._episode_records(rate, 70.0, 400.0, 1.5, 10.0)

    def _source_ports(self, stack: _Stack) -> Iterator[int]:
        """Random source ports per packet: anywhere, or in the tool's OS ephemeral range."""
        rng = self._rng
        low, high = (1024, 65535) if rng.random() < 0.6 else (stack.eph_low, stack.eph_high)
        while True:
            yield rng.randint(low, high)

    def _senders(self, count: int) -> list[tuple[int, int]]:
        """``(syn_length, ttl)`` of the machines sending a flood.

        Each runs its own OS (or a packet-crafting tool such as hping3, whose SYNs carry
        no or minimal options) behind its own number of hops.
        """
        rng = self._rng
        senders = []
        for _ in range(count):
            fingerprint = rng.random()
            if fingerprint < 0.35:
                length, ttl = rng.choice((54, 60, 58)), 64
            else:
                stack = _LINUX if fingerprint < 0.7 else _WINDOWS
                length, ttl = stack.syn_len, stack.ttl
            senders.append((length, ttl - rng.randint(5, 25)))
        return senders

    def _botnet_size(self) -> int:
        rng = self._rng
        return 1 if rng.random() < 0.2 else int(_log_uniform(rng, 2, 25))

    def _syn_flood(self, start: float, target: str) -> float:
        rng = self._rng
        sources_mode, dport_text = self._variant("syn_flood", ("spoofed", "few"), ("80", "443"))
        dport = int(dport_text)
        rate = _log_uniform(rng, 80.0, 600.0)
        n = self._flood_records(rate)
        if sources_mode == "spoofed":  # random source IPs, sent by one or more machines
            senders = self._senders(self._botnet_size())
            sources = None
        else:  # a handful of real hosts, each with its own fingerprint
            hosts = [self._attacker()[0] for _ in range(rng.randint(2, 6))]
            senders = self._senders(len(hosts))
            sources = [
                (ip, length, ttl + (rng.randint(5, 25) if ip.startswith(_LAN) else 0))
                for ip, (length, ttl) in zip(hosts, senders, strict=True)
            ]
        sports = self._source_ports(_LINUX if rng.random() < 0.6 else _WINDOWS)
        emit = self._emit
        ts = start
        for _ in range(n):
            if sources is None:
                src = self._public_ip()
                length, ttl = rng.choice(senders)
            else:
                src, length, ttl = rng.choice(sources)
            emit(ts, src, target, "TCP", next(sports), dport, length, ttl, _FLAG_SYN,
                 "syn_flood")  # fmt: skip
            ts += rng.expovariate(rate)
        return ts

    def _udp_flood(self, start: float, target: str) -> float:
        rng = self._rng
        rate = _log_uniform(rng, 80.0, 600.0)
        n = self._flood_records(rate)
        variant, detail = self._combination("udp_flood", _UDP_FLOOD_VARIANTS)
        if variant == "single":
            sources = [self._attacker()[::2]]
        elif variant == "botnet":  # compromised machines: assorted OSes and distances
            sources = [self._attacker()[::2] for _ in range(int(_log_uniform(rng, 3, 40)))]
        else:  # reflection / amplification off open servers
            sources = [
                (self._public_ip(), rng.choice((64, 128)) - rng.randint(5, 25))
                for _ in range(rng.randint(20, 250))
            ]
        if rng.random() < 0.7:
            port_low, port_high = 1024, 65535
        else:
            port_low = rng.randint(1024, 60000)
            port_high = min(65535, port_low + rng.randint(50, 5000))
        if variant == "reflection":
            reflect_port = int(detail)
            size_low, size_high = _REFLECTION[reflect_port]
        else:  # payload sizes vary per packet (a fixed size would be a trivial signature)
            size_low, size_high = _UDP_FLOOD_SIZES[rng.choice(tuple(_UDP_FLOOD_SIZES))]
        emit = self._emit
        ts = start
        for _ in range(n):
            src, ttl = rng.choice(sources)
            sport = reflect_port if variant == "reflection" else rng.randint(1024, 65535)
            size = rng.randint(size_low, size_high)
            dport = rng.randint(port_low, port_high)
            emit(ts, src, target, "UDP", sport, dport, size, ttl, "", "udp_flood")
            ts += rng.expovariate(rate)
        return ts

    def _icmp_flood(self, start: float, target: str) -> float:
        rng = self._rng
        rate = _log_uniform(rng, 60.0, 500.0)
        n = self._flood_records(rate)
        (variant,) = self._variant("icmp_flood", ("single", "few", "few", "spoofed", "spoofed"))
        size_mode = rng.choice(("default", "large", "random"))
        if variant == "single":
            count = 1
        elif variant == "few":
            count = rng.randint(2, 12)
        else:  # random source IPs, sent by one or more machines
            count = self._botnet_size()
        # (source IP or None for spoofed, ttl, default echo size) per sending machine
        sources = []
        for _ in range(count):
            ip, stack, ttl = self._attacker()
            sources.append((None if variant == "spoofed" else ip, ttl, stack.ping_len))
        large = rng.choice((1514, 1042, 1500))
        emit = self._emit
        ts = start
        for _ in range(n):
            src, ttl, ping_len = rng.choice(sources)
            if size_mode == "default":  # plain "ping -f": the OS's default echo size
                size = ping_len
            elif size_mode == "large":
                size = large
            else:
                size = rng.randint(60, 1514)
            emit(ts, src or self._public_ip(), target, "ICMP", 0, 0, size, ttl, "",
                 "icmp_flood")  # fmt: skip
            ts += rng.expovariate(rate)
        return ts

    # ------------------------------------------------------------------ network

    def _public_ip(self) -> str:
        rng = self._rng
        bits = rng.getrandbits(24)
        return (
            f"{rng.choice(_PUBLIC_FIRST_OCTETS)}.{bits >> 16}.{(bits >> 8) & 255}."
            f"{(bits & 255) or 1}"
        )

    def _public_ip6(self) -> str:
        return str(ipaddress.IPv6Address((0x20010DB8 << 96) | self._rng.getrandbits(96)))

    def _external_host(self, *, ipv6: bool = False) -> _Host:
        rng = self._rng
        stack = _weighted(rng, ((_LINUX, 0.7), (_WINDOWS, 0.2), (_ROUTER, 0.1)))
        hops = rng.randint(6, 24)
        return _Host(
            ip=self._public_ip(),
            stack=stack,
            ttl=stack.ttl - hops,
            rtt=rng.uniform(0.008, 0.15),
            hops=hops,
            ip6=self._public_ip6() if ipv6 else None,
        )

    def _lan_host(self, number: int, stack: _Stack, ipv6: bool = False) -> _Host:
        ip6 = str(ipaddress.IPv6Address(f"2001:db8:1:1::{number:x}")) if ipv6 else None
        return _Host(_LAN + str(number), stack, stack.ttl, self._rng.uniform(2e-4, 1e-3), 0, ip6)

    def _build_network(self) -> None:
        rng = self._rng
        lan = self._lan_host
        self._gateway = lan(1, _LINUX)
        self._monitor = lan(5, _LINUX)
        self._web_server = lan(10, _LINUX, True)
        self._nas = lan(20, _LINUX)
        self._printer = lan(30, _EMBEDDED)
        self._camera = lan(31, _EMBEDDED)
        self._tv = lan(32, _EMBEDDED)
        self._clients = [lan(number, stack, v6) for number, stack, v6, _ in _LAN_CLIENTS]
        self._client_weights = [weight for *_, weight in _LAN_CLIENTS]
        self._lan_hosts = [
            self._gateway,
            self._monitor,
            self._web_server,
            self._nas,
            self._printer,
            self._camera,
            self._tv,
            *self._clients,
        ]
        self._web_servers = [self._external_host(ipv6=rng.random() < 0.4) for _ in range(30)]
        self._web_weights = [_log_uniform(rng, 0.2, 5.0) for _ in self._web_servers]
        self._resolvers = [
            _Host(ip, _LINUX, 64 - rng.randint(8, 14), rng.uniform(0.008, 0.03), 11)
            for ip in ("8.8.8.8", "1.1.1.1", "9.9.9.9")
        ]
        self._ntp_servers = [self._external_host() for _ in range(3)]
        self._ssh_servers = [self._external_host() for _ in range(3)]
        self._mail_server = self._external_host()
        self._dev_server = self._external_host()
        self._media_servers = [self._external_host() for _ in range(2)]
        self._cdn = self._external_host(ipv6=True)

    def _client(self) -> _Host:
        return self._rng.choices(self._clients, self._client_weights)[0]

    def _web_server_pick(self) -> _Host:
        return self._rng.choices(self._web_servers, self._web_weights)[0]

    def _resolver(self) -> _Host:
        if self._rng.random() < 0.85:
            return self._gateway
        return self._rng.choice(self._resolvers)

    # ------------------------------------------------------------------ normal traffic

    def _build_processes(self) -> list[list[Any]]:
        """``[next_time, rate_per_second, activity, follows_activity_level]`` rows."""
        table = (
            (0.3, self._page_load, True),
            (0.5, self._api_call, True),
            (1 / 30, self._bulk_transfer, True),
            (0.5, self._inbound_web, True),
            (1 / 90, self._flash_crowd, True),
            (1 / 40, self._ssh_session, True),
            (1 / 40, self._lan_file, True),
            (1 / 90, self._video_stream, True),
            (1.5, self._dns_background, True),
            (22 / 64, self._ntp, False),
            (0.6, self._discovery, True),
            (0.3, self._quic, True),
            (1 / 45, self._video_call, True),
            (1 / 8, self._ping, True),
            (1.0, self._arp, False),
            (1 / 30, self._monitor_sweep, False),
            (1 / 20, self._multi_port, True),
            (1 / 60, self._peer_to_peer, True),
            (1 / 60, self._traceroute, True),
            (1 / 10, self._failed_connect, True),
            (1 / 30, self._igmp, False),
        )
        rng = self._rng
        scale = self._normal_scale
        rows: list[list[Any]] = []
        if scale <= 0:
            return rows
        for rate, activity, follows_level in table:
            effective = rate * scale
            rows.append(
                [self.start_time + rng.expovariate(effective), effective, activity, follows_level]
            )
        return rows

    def _spawn_normal(self, step_start: float, step_end: float) -> None:
        rng = self._rng
        if step_start >= self._level_until:
            # Office activity drifts: quieter and busier stretches.
            self._level = rng.uniform(0.6, 1.5)
            self._level_until = step_start + rng.uniform(20.0, 90.0)
        level = self._level
        keep = self._normal_keep()
        for row in self._processes:
            next_time, rate, activity, follows_level = row
            while next_time < step_end:
                # Thinning a Poisson process by a coin flip per event keeps it Poisson and
                # takes effect at once (a lower rate would only apply after the next gap).
                if keep >= 1.0 or rng.random() < keep:
                    activity(next_time)
                next_time += rng.expovariate(rate * level if follows_level else rate)
            row[0] = next_time

    def _normal_keep(self) -> float:
        """Share of normal activities to start now (1.0 unless a high ratio is lagging).

        Only ratios whose normal activity is thinned at all (``_normal_scale < 1``) are
        affected, so the streams of lower ratios do not depend on it.
        """
        if self._normal_scale >= 1.0:
            return 1.0
        total = self._normal_count + self._attack_count
        if total < _LAG_MIN_RECORDS:
            return 1.0
        achieved = self._attack_count / total
        if achieved >= self.attack_ratio:
            return 1.0
        return max(_MIN_KEEP, (achieved / self.attack_ratio) ** _LAG_POWER)

    # -- TCP building blocks

    def _tcp_session(
        self,
        t: float,
        client: _Host,
        server: _Host,
        dport: int,
        rounds: Sequence[tuple[float, int, int, int]],
        *,
        v6: bool = False,
        sport: int | None = None,
        pps: float | None = None,
        close: str | None = None,
    ) -> float:
        """A complete TCP connection; returns the time of its last packet.

        ``rounds`` are ``(think_time, request_length, response_packets, response_length)``
        tuples; ``response_length`` 0 means full-size segments with a shorter last one.
        """
        rng = self._rng
        emit = self._emit
        cip, sip = (client.ip6, server.ip6) if v6 else (client.ip, server.ip)
        sport = sport or rng.randint(client.stack.eph_low, client.stack.eph_high)
        rtt = max(client.rtt, server.rtt) * rng.uniform(0.9, 1.3)
        c_ttl, s_ttl = client.ttl, server.ttl
        emit(t, cip, sip, "TCP", sport, dport, client.stack.syn_len, c_ttl, "S")
        t += rtt
        emit(t, sip, cip, "TCP", dport, sport, server.stack.synack_len, s_ttl, "SA")
        t += rng.uniform(2e-4, 2e-3)
        emit(t, cip, sip, "TCP", sport, dport, client.stack.ack_len, c_ttl, "A")
        pps = pps or rng.uniform(500.0, 3000.0)
        for think, request_len, response_packets, response_len in rounds:
            t += think
            emit(t, cip, sip, "TCP", sport, dport, request_len, c_ttl, "PA")
            t += rtt
            t = self._tcp_data(
                t, server, sip, dport, client, cip, sport, response_packets, pps, response_len
            )
        return self._tcp_close(t, client, cip, sport, server, sip, dport, rtt, close)

    def _tcp_data(
        self,
        t: float,
        sender: _Host,
        sender_ip: str | None,
        sender_port: int,
        receiver: _Host,
        receiver_ip: str | None,
        receiver_port: int,
        packets: int,
        pps: float,
        length: int = 0,
    ) -> float:
        """``packets`` data segments; the receiver ACKs every second one."""
        rng = self._rng
        emit = self._emit
        s_ttl, r_ttl, ack_len = sender.ttl, receiver.ttl, receiver.stack.ack_len
        gap = 1.0 / pps
        last = packets - 1
        for i in range(packets):
            if length:
                size = max(60, int(length * rng.uniform(0.6, 1.4)))
            elif i < last:
                size = 1514
            else:
                size = rng.randint(80, 1514)
            flags = "PA" if i == last or rng.random() < 0.08 else "A"
            emit(t, sender_ip, receiver_ip, "TCP", sender_port, receiver_port, size, s_ttl, flags)
            if i % 2 == 1 or i == last:
                emit(
                    t + rng.uniform(1e-4, 2e-3),
                    receiver_ip,
                    sender_ip,
                    "TCP",
                    receiver_port,
                    sender_port,
                    ack_len,
                    r_ttl,
                    "A",
                )
            t += gap * rng.uniform(0.5, 1.5)
        return t

    def _tcp_close(
        self,
        t: float,
        client: _Host,
        cip: str | None,
        sport: int,
        server: _Host,
        sip: str | None,
        dport: int,
        rtt: float,
        close: str | None,
    ) -> float:
        rng = self._rng
        emit = self._emit
        if close is None:
            close = _weighted(rng, (("fin", 0.6), ("server_fin", 0.2), ("rst", 0.12), ("", 0.08)))
        if close == "fin" or close == "server_fin":
            first, first_ip, first_port, second, second_ip, second_port = (
                (client, cip, sport, server, sip, dport)
                if close == "fin"
                else (server, sip, dport, client, cip, sport)
            )
            t += rng.uniform(0.01, 0.5)
            emit(t, first_ip, second_ip, "TCP", first_port, second_port,
                 first.stack.ack_len, first.ttl, "FA")  # fmt: skip
            t += rtt
            emit(t, second_ip, first_ip, "TCP", second_port, first_port,
                 second.stack.ack_len, second.ttl, "FA")  # fmt: skip
            t += rng.uniform(1e-4, 2e-3)
            emit(t, first_ip, second_ip, "TCP", first_port, second_port,
                 first.stack.ack_len, first.ttl, "A")  # fmt: skip
        elif close == "rst":
            t += rng.uniform(0.01, 1.0)
            emit(t, cip, sip, "TCP", sport, dport, client.stack.ack_len, client.ttl,
                 "RA" if rng.random() < 0.7 else "R")  # fmt: skip
        return t

    def _rounds(
        self, n: int, *, first_think: float = 0.005, think: tuple[float, float] = (0.05, 3.0)
    ) -> list[tuple[float, int, int, int]]:
        """Web-like request/response rounds: TLS handshake first, then HTTP exchanges."""
        rng = self._rng
        rounds = [(first_think, rng.randint(517, 700), rng.randint(2, 4), 0)]
        for _ in range(n - 1):
            responses = min(300, int(rng.lognormvariate(1.2, 1.1)) + 1)
            rounds.append((rng.uniform(*think), rng.randint(300, 1200), responses, 0))
        return rounds

    def _dns(self, t: float, client: _Host) -> float:
        """One DNS lookup; returns when the answer arrived."""
        rng = self._rng
        emit = self._emit
        resolver = self._resolver()
        sport = rng.randint(client.stack.eph_low, client.stack.eph_high)
        query = rng.randint(70, 110)
        emit(t, client.ip, resolver.ip, "UDP", sport, 53, query, client.ttl)
        rtt = rng.uniform(5e-4, 3e-3) if resolver is self._gateway else resolver.rtt
        if resolver is self._gateway and rng.random() < 0.3:  # cache miss: ask upstream
            upstream = rng.choice(self._resolvers)
            port = rng.randint(1024, 65535)
            emit(t + 2e-4, resolver.ip, upstream.ip, "UDP", port, 53, query, resolver.ttl)
            rtt += upstream.rtt
            answer = rng.randint(query + 16, 500)
            emit(t + rtt - 2e-4, upstream.ip, resolver.ip, "UDP", 53, port, answer, upstream.ttl)
        emit(t + rtt, resolver.ip, client.ip, "UDP", 53, sport, rng.randint(query + 16, 500),
             resolver.ttl)  # fmt: skip
        return t + rtt

    # -- activities (each spawns one session starting at ``t``)

    def _page_load(self, t: float) -> None:
        rng = self._rng
        client = self._client()
        server = self._web_server_pick()
        ready = t
        for _ in range(rng.randint(1, 3)):
            ready = max(ready, self._dns(t + rng.uniform(0, 0.01), client))
        v6 = bool(client.ip6 and server.ip6) and rng.random() < 0.5
        port = 443 if rng.random() < 0.85 else 80
        connections = 1 if rng.random() < 0.45 else rng.randint(2, 6)
        for k in range(connections):
            start = ready + (rng.uniform(0.001, 0.3) if k else rng.uniform(1e-4, 2e-3))
            self._tcp_session(start, client, server, port, self._rounds(rng.randint(1, 5)), v6=v6)

    def _api_call(self, t: float) -> None:
        rng = self._rng
        client = self._client() if rng.random() < 0.85 else rng.choice(self._lan_hosts[4:7])
        server = self._web_server_pick()
        rounds = self._rounds(rng.randint(1, 3), think=(0.02, 1.0))
        self._tcp_session(t, client, server, 443, rounds)

    def _bulk_transfer(self, t: float) -> None:
        rng = self._rng
        client = self._client()
        server = self._cdn if rng.random() < 0.3 else self._web_server_pick()
        v6 = bool(client.ip6 and server.ip6) and rng.random() < 0.5
        packets = int(_log_uniform(rng, 150, 1200))
        pps = rng.uniform(300.0, 1500.0)
        if rng.random() < 0.75:  # download
            rounds = [*self._rounds(1), (0.01, rng.randint(300, 600), packets, 0)]
            self._tcp_session(t, client, server, 443, rounds, v6=v6, pps=pps, close="fin")
            return
        # upload (cloud backup / file sync): the LAN host streams data out
        sport = rng.randint(client.stack.eph_low, client.stack.eph_high)
        cip, sip = (client.ip6, server.ip6) if v6 else (client.ip, server.ip)
        end = self._tcp_session(t, client, server, 443, self._rounds(1), v6=v6, sport=sport,
                                close="")  # fmt: skip
        end = self._tcp_data(end + 0.01, client, cip, sport, server, sip, 443, packets, pps)
        self._tcp_close(end, client, cip, sport, server, sip, 443, server.rtt, "fin")

    def _internet_client(self) -> _Host:
        rng = self._rng
        stack = _weighted(rng, _NAT_STACKS)
        hops = rng.randint(6, 24)
        return _Host(self._public_ip(), stack, stack.ttl - hops, rng.uniform(0.01, 0.2), hops)

    def _inbound_web(self, t: float, client: _Host | None = None) -> None:
        rng = self._rng
        client = client or self._internet_client()
        # The web server's port forwards: HTTPS, HTTP, SSH, an admin UI, a mail relay.
        port = _weighted(rng, ((443, 0.7), (80, 0.15), (22, 0.06), (8443, 0.05), (25, 0.04)))
        rounds = self._rounds(rng.randint(1, 4), think=(0.05, 2.0))
        self._tcp_session(t, client, self._web_server, port, rounds)

    def _flash_crowd(self, t: float) -> None:
        """A link goes viral: many new visitors hit the LAN web server for a while."""
        rng = self._rng
        duration = rng.uniform(8.0, 20.0)
        for _ in range(rng.randint(10, 40)):
            self._inbound_web(t + rng.uniform(0.0, duration))

    def _ssh_session(self, t: float) -> None:
        rng = self._rng
        client = self._client()
        server = rng.choice((*self._ssh_servers, self._nas, self._web_server, self._dev_server))
        rounds = [(0.01, rng.randint(80, 120), rng.randint(1, 3), 0)]  # version exchange
        rounds += [(0.01, rng.randint(200, 1200), rng.randint(1, 4), 0) for _ in range(3)]
        for _ in range(rng.randint(10, 80)):  # keystrokes and their echo/output
            rounds.append((rng.expovariate(1 / 0.6), rng.randint(90, 150), rng.randint(1, 3),
                           rng.randint(90, 400)))  # fmt: skip
        self._tcp_session(t, client, server, 22, rounds, close="fin")

    def _lan_file(self, t: float) -> None:
        rng = self._rng
        client = self._client()
        port = 2049 if client.stack is _LINUX and rng.random() < 0.5 else 445
        rounds = [
            (rng.uniform(0.001, 0.5), rng.randint(150, 600), rng.randint(1, 20), 0)
            for _ in range(rng.randint(2, 15))
        ]
        if rng.random() < 0.3:  # copy a big file at LAN speed
            rounds.append((0.01, rng.randint(150, 300), int(_log_uniform(rng, 200, 2000)), 0))
        self._tcp_session(t, client, self._nas, port, rounds, pps=rng.uniform(800.0, 3000.0))

    def _video_stream(self, t: float) -> None:
        """The smart TV (or a laptop) fetching video segments every few seconds."""
        rng = self._rng
        client = self._tv if rng.random() < 0.6 else self._client()
        rounds = [(0.005, 600, 3, 0)]
        for _ in range(rng.randint(3, 10)):
            rounds.append((rng.uniform(2.0, 6.0), rng.randint(300, 500),
                           rng.randint(30, 200), 0))  # fmt: skip
        self._tcp_session(t, client, self._cdn, 443, rounds, pps=rng.uniform(400.0, 1500.0))

    def _dns_background(self, t: float) -> None:
        self._dns(t, self._rng.choice(self._lan_hosts[1:]))

    def _ntp(self, t: float) -> None:
        rng = self._rng
        host = rng.choice(self._lan_hosts)
        server = rng.choice(self._ntp_servers)
        sport = 123 if rng.random() < 0.5 else rng.randint(host.stack.eph_low, host.stack.eph_high)
        self._emit(t, host.ip, server.ip, "UDP", sport, 123, 90, host.ttl)
        if rng.random() < 0.97:
            self._emit(t + server.rtt, server.ip, host.ip, "UDP", 123, sport, 90, server.ttl)

    def _discovery(self, t: float) -> None:
        """mDNS / SSDP / LLMNR / NetBIOS chatter to multicast and broadcast addresses."""
        rng = self._rng
        emit = self._emit
        host = rng.choice(self._lan_hosts[1:])
        kind = rng.random()
        if kind < 0.5:
            for i in range(rng.randint(1, 3)):
                emit(t + i * rng.uniform(0.02, 1.0), host.ip, "224.0.0.251", "UDP", 5353, 5353,
                     rng.randint(80, 450), 255)  # fmt: skip
            if rng.random() < 0.5:  # someone answers
                peer = rng.choice(self._lan_hosts[1:])
                emit(t + rng.uniform(0.01, 0.2), peer.ip, "224.0.0.251", "UDP", 5353, 5353,
                     rng.randint(100, 600), 255)  # fmt: skip
        elif kind < 0.75:
            sport = rng.randint(host.stack.eph_low, host.stack.eph_high)
            for i in range(rng.randint(2, 6)):
                emit(t + i * rng.uniform(0.05, 1.0), host.ip, "239.255.255.250", "UDP", sport,
                     1900, rng.randint(160, 400), rng.choice((1, 2, 4)))  # fmt: skip
            for device in (self._tv, self._printer, self._nas):
                if rng.random() < 0.6:
                    emit(t + rng.uniform(0.01, 1.5), device.ip, host.ip, "UDP", 1900, sport,
                         rng.randint(250, 420), device.ttl)  # fmt: skip
        elif kind < 0.85:
            sport = rng.randint(host.stack.eph_low, host.stack.eph_high)
            emit(t, host.ip, "224.0.0.252", "UDP", sport, 5355, rng.randint(64, 90), 1)
        else:
            emit(t, host.ip, "192.168.1.255", "UDP", 137, 137, 92, host.ttl)

    def _quic(self, t: float) -> None:
        rng = self._rng
        emit = self._emit
        client = self._client()
        server = self._web_server_pick()
        v6 = bool(client.ip6 and server.ip6) and rng.random() < 0.5
        cip, sip = (client.ip6, server.ip6) if v6 else (client.ip, server.ip)
        sport = rng.randint(client.stack.eph_low, client.stack.eph_high)
        for i in range(rng.randint(1, 2)):
            emit(t + i * 1e-4, cip, sip, "UDP", sport, 443, rng.randint(1292, 1392), client.ttl)
        t += server.rtt
        packets = min(2000, int(rng.lognormvariate(2.5, 1.3)) + 2)
        gap = 1.0 / rng.uniform(200.0, 1200.0)
        ack_every = rng.randint(2, 10)
        for i in range(packets):
            size = rng.randint(1292, 1392) if i < packets - 1 else rng.randint(80, 1392)
            emit(t, sip, cip, "UDP", 443, sport, size, server.ttl)
            if i % ack_every == 0:
                emit(t + 2e-4, cip, sip, "UDP", sport, 443, rng.randint(70, 100), client.ttl)
            t += gap * rng.uniform(0.5, 1.5)

    def _video_call(self, t: float) -> None:
        """RTP audio/video over UDP to a media server, both directions, for a while."""
        rng = self._rng
        emit = self._emit
        client = self._client()
        server = rng.choice(self._media_servers)
        sport = rng.randint(client.stack.eph_low, client.stack.eph_high)
        dport = rng.choice((3478, 3479, 19302, 8801, rng.randint(10000, 60000)))
        emit(t, client.ip, server.ip, "UDP", sport, dport, rng.randint(62, 100), client.ttl)
        emit(t + server.rtt, server.ip, client.ip, "UDP", dport, sport, rng.randint(62, 100),
             server.ttl)  # fmt: skip
        end = t + rng.uniform(6.0, 20.0)
        gap = 1.0 / rng.uniform(10.0, 30.0)
        ts = t + server.rtt + 0.05
        while ts < end:
            size = rng.randint(100, 220) if rng.random() < 0.4 else rng.randint(400, 1200)
            emit(ts, client.ip, server.ip, "UDP", sport, dport, size, client.ttl)
            size = rng.randint(100, 220) if rng.random() < 0.4 else rng.randint(400, 1200)
            emit(ts + rng.uniform(0, gap), server.ip, client.ip, "UDP", dport, sport, size,
                 server.ttl)  # fmt: skip
            ts += gap * rng.uniform(0.7, 1.3)

    def _ping(self, t: float) -> None:
        rng = self._rng
        client = self._client() if rng.random() < 0.9 else self._nas
        pick = rng.random()
        if pick < 0.3:
            target = self._gateway
        elif pick < 0.8:
            target = self._web_server_pick() if rng.random() < 0.7 else rng.choice(self._resolvers)
        else:
            target = rng.choice(self._lan_hosts)
        if rng.random() < 0.12:  # MTU probing / "ping -s 1472"
            length = rng.choice((1514, 1042, 1500, 578, 1242))
        else:
            length = client.stack.ping_len
        interval = 1.0 if rng.random() < 0.8 else 0.2
        rtt = max(target.rtt, client.rtt) * rng.uniform(0.9, 1.2)
        for i in range(rng.randint(3, 12)):
            ts = t + i * interval * rng.uniform(0.98, 1.02)
            self._emit(ts, client.ip, target.ip, "ICMP", 0, 0, length, client.ttl)
            if rng.random() < 0.95:
                self._emit(ts + rtt, target.ip, client.ip, "ICMP", 0, 0, length, target.ttl)

    def _arp(self, t: float) -> None:
        rng = self._rng
        asker = rng.choice(self._lan_hosts)
        if rng.random() < 0.1:  # gratuitous ARP announcement
            self._emit(t, asker.ip, asker.ip, "ARP", 0, 0, 42, 0)
            return
        other = self._gateway if rng.random() < 0.4 else rng.choice(self._lan_hosts)
        if other is asker:
            other = self._nas if asker is not self._nas else self._gateway
        self._emit(t, asker.ip, other.ip, "ARP", 0, 0, rng.choice((42, 60)), 0)
        if rng.random() < 0.9:
            self._emit(t + rng.uniform(1e-4, 5e-3), other.ip, asker.ip, "ARP", 0, 0,
                       rng.choice((42, 60)), 0)  # fmt: skip

    def _monitor_sweep(self, t: float) -> None:
        """The monitoring box pings every LAN host and checks a few service ports."""
        rng = self._rng
        emit = self._emit
        monitor = self._monitor
        hosts = [host for host in self._lan_hosts if host is not monitor]
        rng.shuffle(hosts)
        ts = t
        for host in hosts:
            emit(ts, monitor.ip, host.ip, "ICMP", 0, 0, 98, monitor.ttl)
            if rng.random() < 0.97:
                emit(ts + host.rtt, host.ip, monitor.ip, "ICMP", 0, 0, 98, host.ttl)
            ts += rng.uniform(0.005, 0.05)
        checks = (
            (self._web_server, (80, 443, 22)),
            (self._nas, (445, 22, 2049, 5000)),
            (self._gateway, (80, 443, 22)),
        )
        for server, ports in checks:
            for port in ports:
                ts += rng.uniform(0.005, 0.1)
                self._tcp_check(ts, monitor, server, port, open_port=rng.random() < 0.9)
        self._dns(ts + 0.01, monitor)

    def _tcp_check(
        self, t: float, client: _Host, server: _Host, port: int, *, open_port: bool
    ) -> None:
        rng = self._rng
        emit = self._emit
        sport = rng.randint(client.stack.eph_low, client.stack.eph_high)
        emit(t, client.ip, server.ip, "TCP", sport, port, client.stack.syn_len, client.ttl, "S")
        rtt = max(client.rtt, server.rtt)
        if not open_port:
            emit(t + rtt, server.ip, client.ip, "TCP", port, sport, 54, server.ttl, "RA")
            return
        emit(t + rtt, server.ip, client.ip, "TCP", port, sport, server.stack.synack_len,
             server.ttl, "SA")  # fmt: skip
        if rng.random() < 0.5:  # half-open check
            emit(t + rtt + 1e-4, client.ip, server.ip, "TCP", sport, port, 54, client.ttl, "R")
            return
        emit(t + rtt + 1e-4, client.ip, server.ip, "TCP", sport, port, client.stack.ack_len,
             client.ttl, "A")  # fmt: skip
        self._tcp_close(t + rtt + 2e-4, client, client.ip, sport, server, server.ip, port,
                        rtt, "fin")  # fmt: skip

    def _multi_port(self, t: float) -> None:
        """One client talks to several ports of one server (mail, dev box, Windows host)."""
        rng = self._rng
        client = self._client()
        kind = rng.random()
        if kind < 0.35:
            server, ports = self._mail_server, rng.sample(_MAIL_PORTS, rng.randint(2, 5))
        elif kind < 0.7:
            server, ports = self._dev_server, rng.sample(_DEV_PORTS, rng.randint(3, 8))
        else:
            server = self._nas if rng.random() < 0.5 else rng.choice(self._clients)
            if server is client:
                server = self._nas
            ports = rng.sample(_WINDOWS_PORTS, rng.randint(2, 5))
        for port in ports:
            start = t + rng.uniform(0.0, 0.8)
            if rng.random() < 0.8:
                rounds = self._rounds(rng.randint(1, 3), think=(0.02, 0.5))
                self._tcp_session(start, client, server, port, rounds)
            else:
                self._tcp_check(start, client, server, port, open_port=False)

    def _peer_to_peer(self, t: float) -> None:
        """A BitTorrent-style client: DHT over UDP and TCP connections to many peers."""
        rng = self._rng
        emit = self._emit
        client = self._clients[rng.choice((6, 7, 10))]
        sport = rng.randint(10000, 60000)
        duration = rng.uniform(5.0, 20.0)
        ts = t
        rate = rng.uniform(2.0, 10.0)
        while ts < t + duration:
            peer = self._external_host()
            pport = rng.randint(1024, 65535)
            emit(ts, client.ip, peer.ip, "UDP", sport, pport, rng.randint(100, 350), client.ttl)
            if rng.random() < 0.7:
                emit(ts + peer.rtt, peer.ip, client.ip, "UDP", pport, sport,
                     rng.randint(100, 450), peer.ttl)  # fmt: skip
            ts += rng.expovariate(rate)
        for _ in range(rng.randint(2, 8)):
            peer = self._external_host()
            port = rng.choice((6881, 6889, 51413, rng.randint(1024, 65535)))
            start = t + rng.uniform(0.0, duration)
            outcome = rng.random()
            if outcome < 0.5:
                rounds = [(0.01, 68, 1, 68), (0.05, rng.randint(100, 300),
                                              int(_log_uniform(rng, 10, 400)), 0)]  # fmt: skip
                self._tcp_session(start, client, peer, port, rounds, pps=rng.uniform(100, 800))
            elif outcome < 0.8:
                self._tcp_check(start, client, peer, port, open_port=False)
            else:
                self._syn_retries(start, client, peer, port)

    def _syn_retries(self, t: float, client: _Host, server: _Host, port: int) -> None:
        sport = self._rng.randint(client.stack.eph_low, client.stack.eph_high)
        for delay in (0.0, 1.0, 3.0, 7.0)[: self._rng.randint(1, 4)]:
            self._emit(t + delay, client.ip, server.ip, "TCP", sport, port, client.stack.syn_len,
                       client.ttl, "S")  # fmt: skip

    def _traceroute(self, t: float) -> None:
        rng = self._rng
        emit = self._emit
        client = self._client()
        target = self._web_server_pick()
        icmp = client.stack is _WINDOWS  # tracert uses ICMP echo, traceroute uses UDP
        sport = rng.randint(client.stack.eph_low, client.stack.eph_high)
        routers = [self._gateway.ip] + [self._public_ip() for _ in range(target.hops)]
        probe = 0
        ts = t
        for hop in range(1, target.hops + 2):
            for _ in range(3):
                if icmp:
                    emit(ts, client.ip, target.ip, "ICMP", 0, 0, 106, hop)
                else:
                    emit(ts, client.ip, target.ip, "UDP", sport + probe, 33434 + probe, 74, hop)
                probe += 1
                rtt = target.rtt * hop / (target.hops + 1)
                if hop <= target.hops:
                    if rng.random() < 0.9:  # "time exceeded" from the router at that hop
                        emit(ts + rtt, routers[hop - 1], client.ip, "ICMP", 0, 0, 102,
                             max(1, 255 - hop if rng.random() < 0.5 else 64 - hop))  # fmt: skip
                else:  # the destination: port unreachable / echo reply
                    emit(ts + rtt, target.ip, client.ip, "ICMP", 0, 0, 102, target.ttl)
                ts += rng.uniform(0.001, 0.05)
            ts += rng.uniform(0.01, 0.2)

    def _failed_connect(self, t: float) -> None:
        rng = self._rng
        client = self._client()
        server = self._web_server_pick() if rng.random() < 0.6 else rng.choice(self._lan_hosts)
        port = rng.choice((*_TOP_PORTS[:20], rng.randint(1024, 65535)))
        if rng.random() < 0.5:
            self._tcp_check(t, client, server, port, open_port=False)
        else:
            self._syn_retries(t, client, server, port)

    def _igmp(self, t: float) -> None:
        host = self._rng.choice(self._lan_hosts)
        self._emit(t, host.ip, "224.0.0.22", "OTHER", 0, 0, 60, 1)


# --------------------------------------------------------------------------- helpers


def _normal_scale(attack_ratio: float) -> float:
    """Intensity of normal activity; thinned at high ratios so they stay reachable."""
    if attack_ratio >= 1:
        return 0.0
    if attack_ratio <= 0:
        return 1.0
    return min(1.0, _THINNING_K * (1 - attack_ratio) / attack_ratio)


def _validate_ratio(attack_ratio: Any) -> float:
    if isinstance(attack_ratio, bool) or not isinstance(attack_ratio, (int, float)):
        raise ValueError(f"attack_ratio must be a number in [0, 1], got {attack_ratio!r}")
    value = float(attack_ratio)
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise ValueError(f"attack_ratio must be in [0, 1], got {attack_ratio!r}")
    return value


def _validate_attack_types(attack_types: Any) -> tuple[str, ...]:
    if isinstance(attack_types, str):
        attack_types = (attack_types,)
    chosen = tuple(dict.fromkeys(attack_types))  # de-duplicate, keep order
    unknown = [name for name in chosen if name not in ATTACK_TYPES]
    if unknown:
        raise ValueError(f"unknown attack types {unknown}; choose from {ATTACK_TYPES}")
    return chosen


def _validate_start_time(start_time: Any) -> float:
    if start_time is None:
        return time.time()
    if isinstance(start_time, bool) or not isinstance(start_time, (int, float)):
        raise ValueError(f"start_time must be epoch seconds, got {start_time!r}")
    value = float(start_time)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"start_time must be a finite, non-negative number, got {start_time}")
    if value > MAX_TIMESTAMP:  # its records would not be valid (see schema.normalize_record)
        raise ValueError(f"start_time must be before the year 3000, got {start_time}")
    return value


def generate_records(
    count: int,
    seed: int | None = None,
    attack_ratio: float = 0.2,
    start_time: float | None = None,
) -> list[dict[str, Any]]:
    """The first ``count`` records of ``TrafficSimulator(seed, attack_ratio, start_time)``."""
    simulator = TrafficSimulator(seed=seed, attack_ratio=attack_ratio, start_time=start_time)
    return list(simulator.records(count))
