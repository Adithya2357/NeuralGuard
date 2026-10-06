"""Packet capture: converts scapy packets into traffic records (see ``schema.py``).

* :func:`parse_packet` turns one scapy packet into a canonical record: IPv4 and IPv6
  packets (TCP, UDP, ICMP / ICMPv6, anything else as ``OTHER``) and ARP. Frames without
  an IP or ARP layer carry no useful signal and are skipped.
* :func:`live_records` sniffs a live interface (needs root or ``CAP_NET_RAW``).
* :func:`pcap_records` reads a ``.pcap`` / ``.pcapng`` file.
* :func:`default_exclusions` / :func:`default_bpf_filter` keep the sniffer off
  NeuralGuard's own Kafka and Elasticsearch connections, which would otherwise form a
  feedback loop (every published record would generate more captured packets).
  ``live_records(exclude_tcp_endpoints=..., exclude_tcp_ports=...)`` drops that traffic
  in the kernel when libpcap is installed (scapy needs it to compile BPF filters) and in
  Python otherwise.

scapy is imported lazily inside the functions, never at module import time: it is slow
to import, noisy, and not needed by the detector or the trainer.
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import queue
import socket
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

from neuralguard.config import Settings
from neuralguard.schema import InvalidRecordError, normalize_record

logger = logging.getLogger(__name__)

# Packets waiting between the sniffer thread and the consumer of live_records(). When the
# consumer falls this far behind, new packets are dropped (and counted) instead of
# letting memory grow without bound.
_MAX_QUEUED_PACKETS = 10_000
_POLL_SECONDS = 0.25  # how often live_records() checks stop_event and the sniffer
_PRIVILEGE_HINT = (
    "live capture needs root or the CAP_NET_RAW capability (e.g. run it with sudo); in a "
    "container, run it as root (docker run --user root --network host ...): --cap-add "
    "NET_RAW alone does not give a non-root user the capability"
)
_LIBPCAP_HINT = "scapy needs libpcap to compile BPF filters (e.g. apt install libpcap0.8)"
_PACKET_OUTGOING = getattr(socket, "PACKET_OUTGOING", 4)  # linux/if_packet.h
# Record lengths are untagged Ethernet II frame sizes: header + IP packet; ARP over
# Ethernet is always 14 + 28 bytes.
_ETHERNET_HEADER = 14
_ARP_FRAME = 42


class CaptureError(RuntimeError):
    """Raised when packets cannot be captured or read."""


class CaptureStoppedError(CaptureError):
    """Raised when a running live capture stops by itself, e.g. because its interface
    went down or was removed: a failure at run time, not a configuration problem."""


# --------------------------------------------------------------------------- parsing


def parse_packet(packet: Any) -> dict[str, Any] | None:
    """The canonical traffic record for one scapy packet, or ``None`` to skip it.

    ``timestamp`` is the capture time (``packet.time``). ``length`` is the size of the
    untagged Ethernet II frame that carries the packet, computed from the IP header's own
    length: the convention of the simulator and so of the model, whatever the capture's
    link layer (VLAN tags, Linux cooked capture, raw IP) or Ethernet padding. IPv4 TTL /
    IPv6 hop limit become ``ttl``; ports and TCP flags come from the first TCP or UDP
    header. ARP records carry the protocol addresses (``psrc`` / ``pdst``) and no ports
    or TTL. A packet that does not make a valid record is logged at DEBUG and skipped.
    """
    from scapy.layers.inet import IP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.l2 import ARP

    if packet.haslayer(IP):
        ip = packet[IP]
        raw = {"source_ip": ip.src, "destination_ip": ip.dst, "ttl": ip.ttl}
        raw.update(_transport(ip))
    elif packet.haslayer(IPv6):
        ip6 = packet[IPv6]
        raw = {"source_ip": ip6.src, "destination_ip": ip6.dst, "ttl": ip6.hlim}
        raw.update(_transport(ip6))
    elif packet.haslayer(ARP):
        arp = packet[ARP]
        raw = {"source_ip": arp.psrc, "destination_ip": arp.pdst, "protocol": "ARP", "ttl": 0}
    else:
        return None
    raw.setdefault("source_port", 0)
    raw.setdefault("destination_port", 0)
    raw.setdefault("tcp_flags", "")
    try:
        raw["timestamp"] = float(packet.time)
        raw["length"] = _frame_length(packet)
        return normalize_record(raw)
    except (InvalidRecordError, TypeError, ValueError) as exc:
        logger.debug("skipping packet that does not make a valid record: %s", exc)
        return None


def _frame_length(packet: Any) -> int:
    """Size of the untagged Ethernet II frame carrying the packet's IP (or ARP) layer.

    It comes from the IP header's own length, so a VLAN tag, a Linux cooked or raw-IP
    link layer, or Ethernet padding does not change it.
    """
    from scapy.layers.inet import IP
    from scapy.layers.inet6 import IPv6

    if packet.haslayer(IP):
        ip = packet[IP]
        # 0 or None: segmentation offload, or a packet built in Python - measure it.
        return _ETHERNET_HEADER + (ip.len or len(ip))
    if packet.haslayer(IPv6):
        ip6 = packet[IPv6]
        return _ETHERNET_HEADER + (40 + ip6.plen if ip6.plen else len(ip6))
    return _ARP_FRAME


def _transport(ip_layer: Any) -> dict[str, Any]:
    """Protocol, ports and TCP flags from the first transport header after ``ip_layer``."""
    from scapy.layers.inet import ICMP, TCP, UDP
    from scapy.layers.inet6 import _ICMPv6

    for layer in ip_layer.iterpayloads():
        if isinstance(layer, TCP):
            return {
                "protocol": "TCP",
                "source_port": int(layer.sport),
                "destination_port": int(layer.dport),
                "tcp_flags": str(layer.flags),
            }
        if isinstance(layer, UDP):
            return {
                "protocol": "UDP",
                "source_port": int(layer.sport),
                "destination_port": int(layer.dport),
            }
        if isinstance(layer, (ICMP, _ICMPv6)):  # every ICMPv6 message type subclasses _ICMPv6
            return {"protocol": "ICMP"}
    return {"protocol": "OTHER"}


def _load_dissectors() -> None:
    """Register scapy's Ethernet / Linux-cooked / IPv4 / IPv6 / ARP dissectors.

    scapy decodes a link type only once the module defining its layer has been imported,
    and neither ``scapy.utils`` nor ``scapy.sendrecv`` imports them. Without this, a pcap
    file or a live socket opened in a fresh process yields undecoded ``Raw`` packets
    (scapy logs "unknown LL type") and every one of them would be skipped.
    """
    import scapy.layers.inet
    import scapy.layers.inet6
    import scapy.layers.l2  # noqa: F401


class Exclusions(NamedTuple):
    """TCP traffic a live capture leaves out: NeuralGuard's own connections."""

    endpoints: tuple[tuple[str, int], ...] = ()  # (ip, port) of a service it talks to
    ports: tuple[int, ...] = ()  # left out on any address (hosts that did not resolve)

    def describe(self) -> str:
        """E.g. ``"127.0.0.1:9092, [::1]:9092, port 9200"``."""
        parts = [f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}" for ip, port in self.endpoints]
        return ", ".join([*parts, *(f"port {port}" for port in self.ports)]) or "nothing"


def default_exclusions(
    settings: Settings, *, resolve: Callable[..., Any] = socket.getaddrinfo
) -> Exclusions:
    """NeuralGuard's own Kafka and Elasticsearch connections, to keep out of a capture.

    Every configured Kafka bootstrap server and Elasticsearch host is resolved
    (``resolve`` is injectable for tests), and only TCP packets from or to exactly one of
    those ``(ip, port)`` pairs are left out. Traffic that merely uses port 9092 or 9200 -
    say, a scan sent from source port 9092 to hide from the sensor - is still captured. A
    host name that does not resolve falls back to its port on every address, with a
    warning. Kafka brokers missing from the bootstrap list are not excluded: list them all.
    """
    endpoints: dict[tuple[str, int], None] = {}
    ports: dict[int, None] = {}
    for host, port in (*settings.kafka_endpoints, *settings.es_endpoints):
        try:
            infos = resolve(host, port, type=socket.SOCK_STREAM)
        except (OSError, UnicodeError) as exc:  # socket.gaierror is an OSError
            logger.warning(
                "cannot resolve %s (%s): TCP port %d is left out of the capture on every address",
                host,
                exc,
                port,
            )
            ports[port] = None
            continue
        for info in infos:
            try:  # sockaddr[0]; link-local IPv6 may carry a zone id ("fe80::1%eth0")
                ip = str(ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]))
            except ValueError:
                continue
            endpoints[(ip, port)] = None
    return Exclusions(tuple(endpoints), tuple(ports))


def default_bpf_filter(
    settings: Settings, *, resolve: Callable[..., Any] = socket.getaddrinfo
) -> str | None:
    """A BPF filter excluding NeuralGuard's own Kafka and Elasticsearch traffic (see
    :func:`default_exclusions`), e.g. ``"not ((src host 127.0.0.1 and tcp src port 9092)
    or (dst host 127.0.0.1 and tcp dst port 9092) or ...)"``."""
    return _exclusion_filter(default_exclusions(settings, resolve=resolve))


def _exclusion_filter(exclusions: Exclusions) -> str | None:
    terms = [
        f"(src host {ip} and tcp src port {port}) or (dst host {ip} and tcp dst port {port})"
        for ip, port in exclusions.endpoints
    ]
    terms += [f"tcp port {port}" for port in exclusions.ports]
    return "not (" + " or ".join(terms) + ")" if terms else None


def _bpf_supported() -> bool:
    """Whether scapy can compile BPF filters here: it loads libpcap for that."""
    try:
        import scapy.libs.winpcapy  # noqa: F401  (despite the name, the libpcap binding)
    except (ImportError, OSError):  # OSError: "Cannot find libpcap.so library"
        return False
    return True


# --------------------------------------------------------------------------- live capture


def live_records(
    interface: str | None = None,
    bpf_filter: str | None = None,
    count: int = 0,
    stop_event: threading.Event | None = None,
    *,
    exclude_tcp_endpoints: Iterable[tuple[str, int]] = (),
    exclude_tcp_ports: Iterable[int] = (),
    sniffer_factory: Callable[..., Any] | None = None,
    poll_seconds: float = _POLL_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Records captured live on ``interface`` (scapy's default interface if ``None``).

    A ``scapy.sendrecv.AsyncSniffer`` (``store=False``) feeds a queue in its own thread;
    packets that :func:`parse_packet` skips do not count. Iteration ends after ``count``
    records (0 = unlimited) or once ``stop_event`` is set (checked at least every
    ``poll_seconds``); the sniffer is always stopped when the iterator finishes or is
    closed. A sniffer that stops by itself - its socket failed, e.g. the link went down -
    raises :class:`CaptureStoppedError`, so a supervisor sees the capture fail.

    TCP packets from or to one of the ``(ip, port)`` pairs of ``exclude_tcp_endpoints``,
    or from or to any address on ``exclude_tcp_ports`` (see :func:`default_exclusions`),
    are never yielded. Without a ``bpf_filter`` that exclusion also becomes the kernel
    BPF filter when libpcap is installed; without libpcap it is applied in Python only (a
    warning says so). A ``bpf_filter`` given without libpcap raises
    :class:`CaptureError`, as does a capture that is not permitted (needs root /
    ``CAP_NET_RAW``) or fails. ``sniffer_factory`` (default ``AsyncSniffer``) is
    injectable for tests.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    if poll_seconds <= 0:
        raise ValueError(f"poll_seconds must be > 0, got {poll_seconds}")
    excluded = Exclusions(
        tuple(dict.fromkeys((_canonical_ip(ip), int(port)) for ip, port in exclude_tcp_endpoints)),
        tuple(dict.fromkeys(int(port) for port in exclude_tcp_ports)),
    )
    ports = (*excluded.ports, *(port for _, port in excluded.endpoints))
    if any(not 0 < port <= 65535 for port in ports):
        raise ValueError(f"excluded TCP ports must be 1-65535, got {excluded}")
    _load_dissectors()
    kernel_filter = bpf_filter or _exclusion_filter(excluded)
    if kernel_filter and not _bpf_supported():
        if bpf_filter:
            raise CaptureError(f"cannot apply the BPF filter {bpf_filter!r}: {_LIBPCAP_HINT}")
        logger.warning(
            "libpcap is not installed, so NeuralGuard's own traffic (TCP %s) is filtered "
            "out in Python instead of the kernel; %s",
            excluded.describe(),
            _LIBPCAP_HINT,
        )
        kernel_filter = None
    options: dict[str, Any] = {}
    if sniffer_factory is None:
        from scapy.sendrecv import AsyncSniffer

        sniffer_factory = AsyncSniffer
        socket_class = _loopback_listen_socket(interface)
        if socket_class is not None:
            options["L2socket"] = socket_class
    return _live_records(
        interface,
        kernel_filter,
        frozenset(excluded.endpoints),
        frozenset(excluded.ports),
        count,
        stop_event,
        sniffer_factory,
        poll_seconds,
        options,
    )


def _loopback_listen_socket(interface: str | None) -> type | None:
    """The scapy socket class to sniff Linux's loopback interface with, else ``None``.

    A Linux packet socket on ``lo`` receives every packet twice: once going out and once
    coming in (libpcap and tcpdump hide the outgoing copy). Left alone, every loopback
    record would be duplicated, doubling every sliding-window count.
    """
    if not interface or not sys.platform.startswith("linux"):
        return None
    from scapy.config import conf

    if interface != conf.loopback_name:
        return None
    try:
        return _inbound_listen_socket()
    except (ImportError, AttributeError) as exc:  # a scapy without these internals
        logger.debug("cannot de-duplicate loopback packets: %s", exc)
        return None


@functools.lru_cache(maxsize=1)
def _inbound_listen_socket() -> type:
    """scapy's Linux ``L2ListenSocket``, minus the outgoing copy of each packet."""
    from scapy.arch.linux import L2ListenSocket

    if not hasattr(L2ListenSocket, "_recv_raw"):
        raise AttributeError("scapy's L2ListenSocket has no _recv_raw()")

    class InboundListenSocket(L2ListenSocket):  # type: ignore[misc, valid-type]
        desc = "read incoming packets at layer 2 using Linux PF_PACKET sockets"

        def _recv_raw(self, sock: Any, x: int) -> tuple[bytes, Any, Any]:
            data, address, timestamp = super()._recv_raw(sock, x)
            # address is the AF_PACKET tuple (ifname, proto, pkttype, hatype, addr)
            if data and len(address) > 2 and address[2] == _PACKET_OUTGOING:
                return b"", address, timestamp  # an empty frame: scapy's recv() skips it
            return data, address, timestamp

    return InboundListenSocket


def _live_records(
    interface: str | None,
    bpf_filter: str | None,
    excluded_endpoints: frozenset[tuple[str, int]],
    excluded_ports: frozenset[int],
    count: int,
    stop_event: threading.Event | None,
    sniffer_factory: Callable[..., Any],
    poll_seconds: float,
    extra_options: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    packets: queue.Queue[Any] = queue.Queue(maxsize=_MAX_QUEUED_PACKETS)
    started = threading.Event()
    dropped = 0

    def enqueue(packet: Any) -> None:  # runs in the sniffer thread
        nonlocal dropped
        try:
            packets.put_nowait(packet)
        except queue.Full:
            dropped += 1
            if dropped == 1:
                logger.warning("capture queue full: dropping packets until it drains")

    options: dict[str, Any] = {
        "prn": enqueue,
        "store": False,
        "started_callback": started.set,
        **extra_options,
    }
    if interface:
        options["iface"] = interface
    if bpf_filter:
        options["filter"] = bpf_filter
    where = interface or "the default interface"
    sniffer = sniffer_factory(**options)
    yielded = 0
    try:
        try:
            sniffer.start()
        except OSError as exc:  # PermissionError is an OSError
            raise _capture_error(exc, where) from exc
        _wait_until_started(sniffer, started, where, poll_seconds)
        logger.info("capturing on %s (filter: %s)", where, bpf_filter or "none")
        while not _stopped(stop_event):
            try:
                packet = packets.get(timeout=poll_seconds)
            except queue.Empty:
                if not _sniffer_alive(sniffer, where) and not _stopped(stop_event):
                    # It is never asked to end by itself (no count, no timeout): scapy
                    # ends the thread, with only a warning, once its socket has failed.
                    raise CaptureStoppedError(
                        f"packet capture on {where} stopped unexpectedly "
                        "(interface down or removed?)"
                    ) from None
                continue
            record = parse_packet(packet)
            if record is None or _is_excluded(record, excluded_endpoints, excluded_ports):
                continue
            yield record
            yielded += 1
            if count and yielded >= count:
                return
    finally:
        _stop_sniffer(sniffer)
        if dropped:
            logger.warning("dropped %d packets because the capture queue was full", dropped)
        logger.debug("live capture on %s ended after %d records", where, yielded)


def _stopped(stop_event: threading.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _wait_until_started(
    sniffer: Any, started: threading.Event, where: str, poll_seconds: float
) -> None:
    """Wait until the sniffer thread has opened its socket and is capturing.

    Until then ``AsyncSniffer.stop()`` does nothing: a sniffer stopped that early would
    start capturing afterwards and never end. Raises :class:`CaptureError` when the
    capture fails to start.
    """
    while not started.wait(poll_seconds):
        if not _sniffer_alive(sniffer, where):  # raises if it ended with an error
            raise CaptureError(f"packet capture on {where} ended before it started")


def _is_excluded(
    record: dict[str, Any], endpoints: frozenset[tuple[str, int]], ports: frozenset[int]
) -> bool:
    if record["protocol"] != "TCP":
        return False
    sport, dport = record["source_port"], record["destination_port"]
    return (
        (record["source_ip"], sport) in endpoints
        or (record["destination_ip"], dport) in endpoints
        or sport in ports
        or dport in ports
    )


def _canonical_ip(ip: str) -> str:
    """``ip`` the way records spell it (``ValueError`` if it is not an IP address)."""
    return str(ipaddress.ip_address(ip))


def _sniffer_alive(sniffer: Any, where: str) -> bool:
    """``False`` once the sniffer thread has ended; raises if it ended with an error."""
    exc = getattr(sniffer, "exception", None)
    if exc is not None:  # AsyncSniffer stores errors raised in its thread here
        raise _capture_error(exc, where) from exc
    thread = getattr(sniffer, "thread", None)
    return thread is None or thread.is_alive()


def _capture_error(exc: BaseException, where: str) -> CaptureError:
    if isinstance(exc, PermissionError):
        return CaptureError(f"permission denied capturing on {where}: {_PRIVILEGE_HINT}")
    if isinstance(exc, OSError):
        return CaptureError(
            f"cannot capture on {where}: {exc}. Check the interface name; {_PRIVILEGE_HINT}"
        )
    return CaptureError(f"packet capture on {where} failed: {exc}")


def _stop_sniffer(sniffer: Any) -> None:
    if not getattr(sniffer, "running", False):
        return
    try:
        sniffer.stop()
    except Exception as exc:  # already failed or never really started: nothing to stop
        logger.debug("error while stopping the packet sniffer: %s", exc)


# --------------------------------------------------------------------------- pcap files


def pcap_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Records read from a ``.pcap`` or ``.pcapng`` file, in file order.

    The file is opened right away, so a missing or unreadable file raises
    :class:`CaptureError` here rather than on the first ``next()``. Packets that
    :func:`parse_packet` skips are left out.
    """
    from scapy.error import Scapy_Exception
    from scapy.utils import PcapReader

    _load_dissectors()  # before the reader picks a class for the file's link type
    path = Path(path)
    try:
        reader = PcapReader(str(path))
    except FileNotFoundError:
        raise CaptureError(f"pcap file not found: {path}") from None
    except (OSError, Scapy_Exception, EOFError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
        raise CaptureError(f"cannot read pcap file {path}: {reason}") from exc
    return _read_pcap(reader, path)


def _read_pcap(reader: Any, path: Path) -> Iterator[dict[str, Any]]:
    from scapy.error import Scapy_Exception

    count = 0
    try:
        for packet in reader:
            record = parse_packet(packet)
            if record is not None:
                count += 1
                yield record
    except (OSError, Scapy_Exception) as exc:
        raise CaptureError(f"error reading pcap file {path}: {exc}") from exc
    finally:
        reader.close()
    logger.info("read %d records from %s", count, path)
