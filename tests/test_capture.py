"""Tests for packet parsing and capture (in-memory scapy packets, a fake sniffer)."""

from __future__ import annotations

import itertools
import logging
import sys
import threading
from typing import ClassVar

import pytest
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import (
    ICMPv6EchoRequest,
    ICMPv6MLReport2,
    ICMPv6ND_NS,
    IPv6,
    IPv6ExtHdrHopByHop,
)
from scapy.layers.l2 import ARP, Ether
from scapy.packet import Raw
from scapy.utils import wrpcap

from neuralguard.capture import (
    CaptureError,
    default_bpf_filter,
    default_excluded_ports,
    live_records,
    parse_packet,
    pcap_records,
)
from neuralguard.config import Settings
from neuralguard.schema import normalize_record

TS = 1_700_000_000.25


def frame(payload, ts=TS):
    """An Ethernet frame with explicit MACs (no route lookups) and a fixed capture time."""
    packet = Ether(src="00:11:22:33:44:55", dst="66:77:88:99:aa:bb") / payload
    packet.time = ts
    return packet


def tcp_syn(ts=TS, sport=40000, dport=443):
    return frame(
        IP(src="192.168.1.10", dst="10.0.0.3", ttl=63) / TCP(sport=sport, dport=dport, flags="S"),
        ts,
    )


def lldp():
    packet = Ether(src="00:11:22:33:44:55", dst="01:80:c2:00:00:0e", type=0x88CC) / Raw(b"x" * 30)
    packet.time = TS
    return packet


# ------------------------------------------------------------------ parse_packet


def test_scapy_is_not_imported_at_module_import_time():
    import subprocess

    code = "import neuralguard.capture, sys; print('scapy' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_ipv4_tcp_syn():
    packet = tcp_syn()
    record = parse_packet(packet)
    assert record == {
        "timestamp": TS,
        "source_ip": "192.168.1.10",
        "destination_ip": "10.0.0.3",
        "protocol": "TCP",
        "source_port": 40000,
        "destination_port": 443,
        "length": len(packet),
        "ttl": 63,
        "tcp_flags": "S",
    }
    assert record["length"] == 54  # Ethernet 14 + IPv4 20 + TCP 20
    assert normalize_record(record) == record


@pytest.mark.parametrize(
    ("flags", "expected"),
    [("AS", "SA"), (0, ""), ("FPU", "FPU"), ("RA", "RA"), ("FA", "FA"), ("SEC", "SEC")],
)
def test_tcp_flags_are_canonical(flags, expected):
    packet = frame(IP(src="10.0.0.1", dst="10.0.0.2") / TCP(flags=flags))
    assert parse_packet(packet)["tcp_flags"] == expected


def test_ipv6_udp():
    packet = frame(
        IPv6(src="2001:DB8::1", dst="2001:db8::2", hlim=57)
        / UDP(sport=53, dport=5353)
        / Raw(b"q" * 10)
    )
    record = parse_packet(packet)
    assert record["protocol"] == "UDP"
    assert record["source_ip"] == "2001:db8::1"  # canonical (lower-case, compressed)
    assert record["destination_ip"] == "2001:db8::2"
    assert (record["source_port"], record["destination_port"]) == (53, 5353)
    assert record["ttl"] == 57  # the hop limit
    assert record["length"] == len(packet) == 14 + 40 + 8 + 10
    assert record["tcp_flags"] == ""


def test_icmp():
    record = parse_packet(frame(IP(src="10.0.0.1", dst="10.0.0.2", ttl=128) / ICMP()))
    assert record["protocol"] == "ICMP"
    assert (record["source_port"], record["destination_port"], record["ttl"]) == (0, 0, 128)


@pytest.mark.parametrize(
    "payload",
    [
        IPv6(src="2001:db8::1", dst="2001:db8::2") / ICMPv6EchoRequest(),
        IPv6(src="2001:db8::1", dst="ff02::1:ff00:2") / ICMPv6ND_NS(tgt="2001:db8::2"),
        # MLD reports sit behind a hop-by-hop extension header
        IPv6(src="fe80::1", dst="ff02::16") / IPv6ExtHdrHopByHop() / ICMPv6MLReport2(),
    ],
    ids=["echo", "neighbour-solicitation", "mld-behind-extension-header"],
)
def test_icmpv6_counts_as_icmp(payload):
    record = parse_packet(frame(payload))
    assert record["protocol"] == "ICMP"
    assert record["source_port"] == record["destination_port"] == 0


def test_icmp_error_uses_the_outer_header():
    # A "port unreachable" quoting the offending UDP datagram: still ICMP, no ports.
    payload = IP(src="10.0.0.2", dst="10.0.0.1") / ICMP(type=3, code=3) / IP(src="10.0.0.1") / UDP()
    record = parse_packet(frame(payload))
    assert record["protocol"] == "ICMP"
    assert record["source_ip"] == "10.0.0.2"
    assert record["source_port"] == 0


def test_arp():
    packet = frame(ARP(op=1, psrc="192.168.1.20", pdst="192.168.1.1"))
    record = parse_packet(packet)
    assert record == {
        "timestamp": TS,
        "source_ip": "192.168.1.20",
        "destination_ip": "192.168.1.1",
        "protocol": "ARP",
        "source_port": 0,
        "destination_port": 0,
        "length": 42,
        "ttl": 0,
        "tcp_flags": "",
    }


def test_other_ip_protocol():
    record = parse_packet(
        frame(IP(src="192.168.1.30", dst="224.0.0.22", proto=2, ttl=1) / Raw(b"\x22" * 8))
    )
    assert record["protocol"] == "OTHER"
    assert (record["source_port"], record["destination_port"], record["ttl"]) == (0, 0, 1)


def test_non_ip_frame_is_skipped():
    assert parse_packet(lldp()) is None


def test_invalid_record_is_skipped(caplog):
    packet = tcp_syn()
    packet.time = -5.0  # not a valid timestamp
    with caplog.at_level(logging.DEBUG, logger="neuralguard.capture"):
        assert parse_packet(packet) is None
    assert any("skipping packet" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ default_bpf_filter


def test_default_bpf_filter():
    assert default_bpf_filter(Settings()) == "not (tcp port 9092 or tcp port 9200)"


def test_default_bpf_filter_uses_every_configured_port_once():
    settings = Settings(
        kafka_bootstrap_servers=("kafka-1:29092", "kafka-2:29092", "kafka-3"),
        es_hosts=("https://es:9243", "http://es2:29092"),
    )
    assert default_bpf_filter(settings) == (
        "not (tcp port 29092 or tcp port 9092 or tcp port 9243)"
    )


# ------------------------------------------------------------------ pcap_records


def test_pcap_records(tmp_path):
    packets = [
        tcp_syn(TS),
        lldp(),
        frame(IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=1, dport=2), TS + 1),
        frame(ARP(psrc="192.168.1.5", pdst="192.168.1.1"), TS + 2),
    ]
    path = tmp_path / "capture.pcap"
    wrpcap(str(path), packets)
    records = list(pcap_records(path))
    assert [r["protocol"] for r in records] == ["TCP", "UDP", "ARP"]  # the LLDP frame is skipped
    assert [r["timestamp"] for r in records] == pytest.approx([TS, TS + 1, TS + 2])
    assert records[0]["tcp_flags"] == "S"
    assert all(normalize_record(r) == r for r in records)


def run_fresh(code, *args):
    """stdout of ``code`` run in a fresh interpreter. In-process tests have scapy's layers
    loaded already (this file imports them), which would hide a missing registration."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.split()


def test_pcap_records_decode_packets_in_a_fresh_process(tmp_path):
    # Without registering scapy's layers first, every packet of the file comes out as
    # undecoded Raw ("unknown LL type") and is skipped.
    path = tmp_path / "capture.pcap"
    wrpcap(str(path), [tcp_syn(TS), frame(ARP(psrc="10.0.0.5", pdst="10.0.0.1"), TS + 1)])
    code = (
        "import sys\n"
        "from neuralguard.capture import pcap_records\n"
        "print(','.join(r['protocol'] for r in pcap_records(sys.argv[1])))\n"
    )
    assert run_fresh(code, path) == ["TCP,ARP"]


def test_live_records_register_the_ethernet_dissector_in_a_fresh_process():
    code = (
        "from neuralguard.capture import live_records\n"
        "live_records(sniffer_factory=lambda **options: None)\n"
        "from scapy.config import conf\n"
        "print(getattr(conf.l2types.num2layer.get(1), '__name__', None))\n"  # DLT_EN10MB
    )
    assert run_fresh(code) == ["Ether"]


def test_pcapng_records(tmp_path):
    from scapy.utils import PcapNgWriter

    path = tmp_path / "capture.pcapng"
    writer = PcapNgWriter(str(path))
    for packet in (tcp_syn(TS), tcp_syn(TS + 0.5, dport=80)):
        writer.write(packet)
    writer.close()
    records = list(pcap_records(str(path)))
    assert [r["destination_port"] for r in records] == [443, 80]


def test_missing_pcap(tmp_path):
    with pytest.raises(CaptureError, match="not found"):
        pcap_records(tmp_path / "nope.pcap")  # raised at call time, before iterating


@pytest.mark.parametrize("content", [b"", b"this is not a capture file" * 4])
def test_unreadable_pcap(tmp_path, content):
    path = tmp_path / "bad.pcap"
    path.write_bytes(content)
    with pytest.raises(CaptureError, match="cannot read pcap"):
        pcap_records(path)


def test_pcap_directory(tmp_path):
    with pytest.raises(CaptureError):
        pcap_records(tmp_path)


# ------------------------------------------------------------------ live_records


class FakeThread:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive


class FakeSniffer:
    """Stands in for scapy's AsyncSniffer: "captures" ``packets`` as soon as it starts."""

    instances: ClassVar[list[FakeSniffer]] = []
    packets: ClassVar[list] = []
    start_error: BaseException | None = None
    thread_error: BaseException | None = None
    ends_by_itself = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.running = False
        self.stopped = False
        self.exception = None
        self.thread = None
        FakeSniffer.instances.append(self)

    def start(self):
        if self.start_error is not None:
            raise self.start_error
        self.running = True
        if self.thread_error is not None:  # scapy stores errors from its thread
            self.exception = self.thread_error
            return
        for packet in self.packets:
            assert self.kwargs["prn"](packet) is None  # a non-None result would be printed
        self.thread = FakeThread(alive=not self.ends_by_itself)

    def stop(self, join=True):
        if self.exception is not None:
            raise self.exception
        self.running = False
        self.stopped = True


def fake_sniffer(packets=(), **attributes):
    FakeSniffer.instances = []
    return type("Sniffer", (FakeSniffer,), {"packets": list(packets), **attributes})


@pytest.fixture(autouse=True)
def libpcap(monkeypatch):
    """Whether scapy can compile BPF filters, independent of the test machine: "yes"
    unless a test sets ``libpcap.installed = False`` to simulate a host without it."""
    import neuralguard.capture

    state = type("Libpcap", (), {"installed": True})()
    monkeypatch.setattr(neuralguard.capture, "_bpf_supported", lambda: state.installed)
    return state


def live(sniffer, **kwargs):
    kwargs.setdefault("poll_seconds", 0.01)
    return live_records(sniffer_factory=sniffer, **kwargs)


def test_live_records_count_and_options():
    sniffer = fake_sniffer([tcp_syn(TS + i, dport=1000 + i) for i in range(5)])
    records = list(live(sniffer, interface="eth0", bpf_filter="not tcp port 9092", count=3))
    assert [r["destination_port"] for r in records] == [1000, 1001, 1002]
    (instance,) = FakeSniffer.instances
    assert instance.kwargs["iface"] == "eth0"
    assert instance.kwargs["filter"] == "not tcp port 9092"
    assert instance.kwargs["store"] is False
    assert instance.stopped


def test_live_records_defaults_leave_interface_and_filter_to_scapy():
    sniffer = fake_sniffer([tcp_syn()])
    assert len(list(live(sniffer, count=1))) == 1
    assert set(FakeSniffer.instances[0].kwargs) == {"prn", "store"}


def test_live_records_skip_unparseable_packets_without_counting_them():
    sniffer = fake_sniffer([lldp(), tcp_syn(dport=1), lldp(), tcp_syn(dport=2)])
    records = list(live(sniffer, count=2))
    assert [r["destination_port"] for r in records] == [1, 2]


def test_live_records_stop_event():
    stop = threading.Event()
    sniffer = fake_sniffer([tcp_syn(TS + i) for i in range(10)])
    seen = []
    for record in live(sniffer, stop_event=stop):  # count=0: unlimited
        seen.append(record)
        if len(seen) == 2:
            stop.set()
    assert len(seen) == 2
    assert FakeSniffer.instances[0].stopped


def test_live_records_stop_event_while_waiting_for_packets():
    stop = threading.Event()
    timer = threading.Timer(0.05, stop.set)
    timer.start()
    try:
        assert list(live(fake_sniffer([]), stop_event=stop)) == []
    finally:
        timer.cancel()
    assert FakeSniffer.instances[0].stopped


def test_live_records_end_when_the_sniffer_ends():
    sniffer = fake_sniffer([tcp_syn(), tcp_syn()], ends_by_itself=True)
    assert len(list(live(sniffer))) == 2


def test_closing_the_iterator_stops_the_sniffer():
    iterator = live(fake_sniffer([tcp_syn(), tcp_syn()]))
    next(iterator)
    iterator.close()
    assert FakeSniffer.instances[0].stopped


def test_live_records_do_not_sniff_until_iterated():
    iterator = live(fake_sniffer([tcp_syn(), tcp_syn()]))
    assert FakeSniffer.instances == []
    assert len(list(itertools.islice(iterator, 1))) == 1
    assert len(FakeSniffer.instances) == 1
    iterator.close()


def test_permission_error_on_start():
    sniffer = fake_sniffer(start_error=PermissionError(1, "Operation not permitted"))
    with pytest.raises(CaptureError, match=r"root.*CAP_NET_RAW.*sudo"):
        list(live(sniffer, interface="eth0"))


def test_permission_error_in_the_sniffer_thread():
    sniffer = fake_sniffer(thread_error=PermissionError(1, "Operation not permitted"))
    with pytest.raises(CaptureError, match="sudo"):
        list(live(sniffer))


def test_unknown_interface():
    sniffer = fake_sniffer(start_error=OSError(19, "No such device"))
    with pytest.raises(CaptureError, match=r"cannot capture on nope0.*interface name"):
        list(live(sniffer, interface="nope0"))


def test_other_sniffer_failure():
    sniffer = fake_sniffer(thread_error=RuntimeError("Failed to compile filter expression"))
    with pytest.raises(CaptureError, match="compile filter"):
        list(live(sniffer, bpf_filter="not a filter"))


def test_excluded_tcp_ports_become_the_kernel_filter_and_are_dropped_in_python():
    packets = [tcp_syn(dport=9092), tcp_syn(dport=443), tcp_syn(sport=9200, dport=40000)]
    packets.append(frame(IP(src="10.0.0.1", dst="10.0.0.2") / UDP(sport=1, dport=9092)))
    sniffer = fake_sniffer(packets)  # the fake does not filter, so Python must
    records = list(live(sniffer, count=2, exclude_tcp_ports=[9092, 9200, 9092]))
    kept = [(r["protocol"], r["destination_port"]) for r in records]
    assert kept == [("TCP", 443), ("UDP", 9092)]
    assert FakeSniffer.instances[0].kwargs["filter"] == "not (tcp port 9092 or tcp port 9200)"


def test_without_libpcap_the_exclusion_runs_in_python_only(libpcap, caplog):
    libpcap.installed = False
    sniffer = fake_sniffer([tcp_syn(dport=9092), tcp_syn(dport=443)])
    with caplog.at_level(logging.WARNING, logger="neuralguard.capture"):
        records = list(live(sniffer, count=1, exclude_tcp_ports=[9092]))
    assert [r["destination_port"] for r in records] == [443]
    assert "filter" not in FakeSniffer.instances[0].kwargs  # scapy could not compile one
    assert any("libpcap is not installed" in r.getMessage() for r in caplog.records)


def test_an_explicit_bpf_filter_without_libpcap_is_a_capture_error(libpcap):
    libpcap.installed = False
    with pytest.raises(CaptureError, match=r"cannot apply the BPF filter 'udp'.*libpcap"):
        live_records(bpf_filter="udp", sniffer_factory=fake_sniffer())
    assert FakeSniffer.instances == []


def test_an_explicit_bpf_filter_replaces_the_exclusion_filter():
    sniffer = fake_sniffer([tcp_syn(dport=9092), tcp_syn(dport=22)])
    records = list(live(sniffer, count=1, bpf_filter="tcp", exclude_tcp_ports=[9092]))
    assert [r["destination_port"] for r in records] == [22]  # still dropped in Python
    assert FakeSniffer.instances[0].kwargs["filter"] == "tcp"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux packet sockets")
def test_loopback_capture_drops_the_outgoing_copy_of_each_packet(monkeypatch):
    import socket

    from scapy.arch.linux import L2ListenSocket

    from neuralguard.capture import _loopback_listen_socket

    assert _loopback_listen_socket(None) is None
    assert _loopback_listen_socket("eth0") is None
    socket_class = _loopback_listen_socket("lo")
    assert issubclass(socket_class, L2ListenSocket)

    frames = iter(
        [
            (b"frame-1", ("lo", 0x0800, socket.PACKET_OUTGOING, 772, b""), 1.0),
            (b"frame-1", ("lo", 0x0800, socket.PACKET_HOST, 772, b""), 1.0),
        ]
    )
    monkeypatch.setattr(L2ListenSocket, "_recv_raw", lambda self, sock, x: next(frames))
    listener = object.__new__(socket_class)  # no real socket needed
    assert listener._recv_raw(None, 1500)[0] == b""  # scapy's recv() turns this into None
    assert listener._recv_raw(None, 1500)[0] == b"frame-1"


def test_live_capture_on_loopback_uses_the_deduplicating_socket(monkeypatch):
    import scapy.sendrecv

    from neuralguard.capture import _loopback_listen_socket

    sniffer = fake_sniffer([tcp_syn()])
    monkeypatch.setattr(scapy.sendrecv, "AsyncSniffer", sniffer)
    assert len(list(live_records(interface="lo", count=1, poll_seconds=0.01))) == 1
    expected = _loopback_listen_socket("lo")  # None where there is no such socket
    assert FakeSniffer.instances[0].kwargs.get("L2socket") is expected


def test_default_excluded_ports():
    assert default_excluded_ports(Settings()) == (9092, 9200)
    settings = Settings(
        kafka_bootstrap_servers=("k1:9092", "k2:9093"), es_hosts=("https://es:9092",)
    )
    assert default_excluded_ports(settings) == (9092, 9093)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": -1},
        {"count": 1.5},
        {"poll_seconds": 0},
        {"exclude_tcp_ports": [0]},
        {"exclude_tcp_ports": [70000]},
    ],
)
def test_live_records_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        live_records(sniffer_factory=fake_sniffer(), **kwargs)
