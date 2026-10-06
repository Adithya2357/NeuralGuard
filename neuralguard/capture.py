"""Packet capture: converts scapy packets into traffic records (see ``schema.py``).

* :func:`parse_packet` turns one scapy packet into a canonical record: IPv4 and IPv6
  packets (TCP, UDP, ICMP / ICMPv6, anything else as ``OTHER``) and ARP. Frames without
  an IP or ARP layer carry no useful signal and are skipped.
* :func:`live_records` sniffs a live interface (needs root or ``CAP_NET_RAW``).
* :func:`pcap_records` reads a ``.pcap`` / ``.pcapng`` file.
* :func:`default_excluded_ports` / :func:`default_bpf_filter` keep the sniffer off
  NeuralGuard's own Kafka and Elasticsearch connections, which would otherwise form a
  feedback loop (every published record would generate more captured packets).
  ``live_records(exclude_tcp_ports=...)`` drops that traffic in the kernel when libpcap
  is installed (scapy needs it to compile BPF filters) and in Python otherwise.

scapy is imported lazily inside the functions, never at module import time: it is slow
to import, noisy, and not needed by the detector or the trainer.
"""

from __future__ import annotations

import functools
import logging
import queue
import socket
import sys
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from neuralguard.config import Settings
from neuralguard.schema import InvalidRecordError, normalize_record

logger = logging.getLogger(__name__)

# Packets waiting between the sniffer thread and the consumer of live_records(). When the
# consumer falls this far behind, new packets are dropped (and counted) instead of
# letting memory grow without bound.
_MAX_QUEUED_PACKETS = 10_000
_POLL_SECONDS = 0.25  # how often live_records() checks stop_event and the sniffer
_PRIVILEGE_HINT = "live capture needs root or the CAP_NET_RAW capability (e.g. run it with sudo)"
_LIBPCAP_HINT = "scapy needs libpcap to compile BPF filters (e.g. apt install libpcap0.8)"
_PACKET_OUTGOING = getattr(socket, "PACKET_OUTGOING", 4)  # linux/if_packet.h


class CaptureError(RuntimeError):
    """Raised when packets cannot be captured or read."""


# --------------------------------------------------------------------------- parsing


def parse_packet(packet: Any) -> dict[str, Any] | None:
    """The canonical traffic record for one scapy packet, or ``None`` to skip it.

    ``timestamp`` is the capture time (``packet.time``) and ``length`` the frame size
    (``len(packet)``). IPv4 TTL / IPv6 hop limit become ``ttl``; ports and TCP flags
    come from the first TCP or UDP header. ARP records carry the protocol addresses
    (``psrc`` / ``pdst``) and no ports or TTL. A packet that does not make a valid
    record is logged at DEBUG and skipped.
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
        raw["length"] = len(packet)
        return normalize_record(raw)
    except (InvalidRecordError, TypeError, ValueError) as exc:
        logger.debug("skipping packet that does not make a valid record: %s", exc)
        return None


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


def default_excluded_ports(settings: Settings) -> tuple[int, ...]:
    """TCP ports of NeuralGuard's own Kafka and Elasticsearch connections, each once."""
    return tuple(dict.fromkeys((*settings.kafka_ports, *settings.es_ports)))


def default_bpf_filter(settings: Settings) -> str:
    """A BPF filter excluding NeuralGuard's own Kafka and Elasticsearch traffic.

    For the defaults that is ``"not (tcp port 9092 or tcp port 9200)"``.
    """
    return _exclusion_filter(default_excluded_ports(settings))


def _exclusion_filter(ports: Iterable[int]) -> str:
    return "not (" + " or ".join(f"tcp port {port}" for port in ports) + ")"


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
    exclude_tcp_ports: Iterable[int] = (),
    sniffer_factory: Callable[..., Any] | None = None,
    poll_seconds: float = _POLL_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Records captured live on ``interface`` (scapy's default interface if ``None``).

    A ``scapy.sendrecv.AsyncSniffer`` (``store=False``) feeds a queue in its own thread;
    packets that :func:`parse_packet` skips do not count. Iteration ends after ``count``
    records (0 = unlimited), once ``stop_event`` is set (checked at least every
    ``poll_seconds``) or when the sniffer stops by itself; the sniffer is always stopped
    when the iterator finishes or is closed.

    TCP packets from or to ``exclude_tcp_ports`` (see :func:`default_excluded_ports`) are
    never yielded. Without a ``bpf_filter`` that exclusion also becomes the kernel BPF
    filter when libpcap is installed; without libpcap it is applied in Python only (a
    warning says so). A ``bpf_filter`` given without libpcap raises
    :class:`CaptureError`, as does a capture that is not permitted (needs root /
    ``CAP_NET_RAW``) or fails. ``sniffer_factory`` (default ``AsyncSniffer``) is
    injectable for tests.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"count must be a non-negative integer, got {count!r}")
    if poll_seconds <= 0:
        raise ValueError(f"poll_seconds must be > 0, got {poll_seconds}")
    excluded = tuple(dict.fromkeys(int(port) for port in exclude_tcp_ports))
    if any(not 0 < port <= 65535 for port in excluded):
        raise ValueError(f"exclude_tcp_ports must be TCP ports (1-65535), got {excluded}")
    _load_dissectors()
    kernel_filter = bpf_filter or (_exclusion_filter(excluded) if excluded else None)
    if kernel_filter and not _bpf_supported():
        if bpf_filter:
            raise CaptureError(f"cannot apply the BPF filter {bpf_filter!r}: {_LIBPCAP_HINT}")
        logger.warning(
            "libpcap is not installed, so NeuralGuard's own traffic (TCP ports %s) is "
            "filtered out in Python instead of the kernel; %s",
            ", ".join(map(str, excluded)),
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
        frozenset(excluded),
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
    excluded: frozenset[int],
    count: int,
    stop_event: threading.Event | None,
    sniffer_factory: Callable[..., Any],
    poll_seconds: float,
    extra_options: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    packets: queue.Queue[Any] = queue.Queue(maxsize=_MAX_QUEUED_PACKETS)
    dropped = 0

    def enqueue(packet: Any) -> None:  # runs in the sniffer thread
        nonlocal dropped
        try:
            packets.put_nowait(packet)
        except queue.Full:
            dropped += 1
            if dropped == 1:
                logger.warning("capture queue full: dropping packets until it drains")

    options: dict[str, Any] = {"prn": enqueue, "store": False, **extra_options}
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
        logger.info("capturing on %s (filter: %s)", where, bpf_filter or "none")
        while not _stopped(stop_event):
            try:
                packet = packets.get(timeout=poll_seconds)
            except queue.Empty:
                if not _sniffer_alive(sniffer, where):
                    logger.info("the packet sniffer on %s stopped", where)
                    return
                continue
            record = parse_packet(packet)
            if record is None or _is_excluded(record, excluded):
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


def _is_excluded(record: dict[str, Any], ports: frozenset[int]) -> bool:
    return (
        bool(ports)
        and record["protocol"] == "TCP"
        and (record["source_port"] in ports or record["destination_port"] in ports)
    )


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
