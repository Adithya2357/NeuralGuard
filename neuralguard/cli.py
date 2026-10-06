"""Command-line interface: ``neuralguard <command>`` (or ``python -m neuralguard``).

Commands:

* ``train``   - train the threat model on simulated traffic (or a labelled JSONL
  capture), print an evaluation report and save the model.
* ``produce`` - publish traffic records (simulated, live capture or a pcap file) to Kafka,
  or to a JSON Lines file with ``--output``.
* ``detect``  - consume traffic from Kafka, detect threats and send alerts to the
  console, Elasticsearch and optionally a JSON Lines file.
* ``demo``    - a zero-infrastructure end-to-end run: simulated traffic straight through
  the detector, with a summary comparing detections with the ground truth.

Settings come from ``NEURALGUARD_*`` environment variables (see :mod:`neuralguard.config`);
command-line flags override them. Errors in user input print one ``error: ...`` line to
stderr and exit with status 2 (1 for an unavailable service, an I/O error or a live
capture that stopped, 130 for Ctrl+C). Heavy dependencies (scikit-learn, scapy, kafka,
elasticsearch) are imported inside the commands, so ``neuralguard --help`` stays fast.
"""

from __future__ import annotations

import argparse
import errno
import itertools
import json
import logging
import math
import os
import secrets
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuralguard import __version__
from neuralguard.config import Settings
from neuralguard.logutil import configure_logging
from neuralguard.schema import ATTACK_TYPES, LABELS, NORMAL_LABEL

logger = logging.getLogger(__name__)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

# `neuralguard train`'s simulated training records: four independent 60k-record streams
# (neuralguard.train.DEFAULT_SAMPLES, repeated so that --help need not import numpy).
TRAIN_SAMPLES = 240_000
# The demo's default stream: long enough that seed 7 shows an episode of every attack
# type (the simulator cycles through all of them; see tests/test_cli.py).
DEMO_COUNT = 6000
DEMO_SEED = 7
# The fallback in-memory model sees as much simulated traffic as `neuralguard train`
# (with fewer trees): fewer records cover too few episodes to have seen every attack
# variant, and whole unseen variants then go undetected. Its seed is never the demo
# stream's: scored on its own training traffic, a model looks flawless.
DEMO_TRAIN_SAMPLES = TRAIN_SAMPLES
DEMO_TRAIN_TREES = 100
DEMO_TRAIN_SEED = 42
_DEMO_BATCH_SIZE = 500

# Errors from these classes are reported as one line instead of a traceback. They are
# looked up only in modules that are already imported: an exception can only be an
# instance of a loaded class, and importing scikit-learn or kafka just to check would
# slow down every error path.
_USAGE_ERRORS = (("neuralguard.model", "ModelError"), ("neuralguard.capture", "CaptureError"))
# Checked first: a live capture that stopped while running is a CaptureError too, but a
# failure at run time (status 1, so restart-on-failure supervisors restart it).
_SERVICE_ERRORS = (("kafka.errors", "KafkaError"), ("neuralguard.capture", "CaptureStoppedError"))


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """The ``neuralguard`` argument parser (no heavy imports)."""
    parser = argparse.ArgumentParser(
        prog="neuralguard",
        description="AI-powered real-time network intrusion detection.",
        epilog="Settings are read from NEURALGUARD_* environment variables; flags override "
        "them. Run 'neuralguard COMMAND --help' for a command's options.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        help="logging level (default: $NEURALGUARD_LOG_LEVEL or INFO)",
    )
    parser.add_argument(
        "--log-format",
        choices=("text", "json"),
        help="log format on stderr (default: $NEURALGUARD_LOG_FORMAT or text)",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    _add_train(commands)
    _add_produce(commands)
    _add_detect(commands)
    _add_demo(commands)
    return parser


def _add_train(commands: Any) -> None:
    sub = commands.add_parser(
        "train",
        help="train the threat model and print an evaluation report",
        description="Train the threat model on simulated traffic (or a labelled JSON Lines "
        "capture), evaluate it on held-out data, print the report and save the model.",
    )
    sub.add_argument(
        "--samples",
        type=_positive_int,
        default=TRAIN_SAMPLES,
        metavar="N",
        help="simulated training records, in independent streams of up to 60000 "
        "(default: %(default)s)",
    )
    sub.add_argument("--seed", type=int, default=42, help="random seed (default: %(default)s)")
    sub.add_argument(
        "--attack-ratio",
        type=_fraction,
        default=0.3,
        metavar="R",
        help="fraction of simulated attack traffic, 0-1 (default: %(default)s)",
    )
    sub.add_argument(
        "--window",
        type=_finite_float,
        metavar="SECONDS",
        help="feature window (default: $NEURALGUARD_WINDOW_SECONDS or 10)",
    )
    sub.add_argument(
        "--trees",
        type=_positive_int,
        default=200,
        metavar="N",
        help="random-forest trees (default: %(default)s)",
    )
    sub.add_argument(
        "--data",
        type=Path,
        metavar="FILE.jsonl",
        help="train on labelled records from this JSON Lines file instead of simulated traffic",
    )
    sub.add_argument(
        "--output",
        type=Path,
        metavar="PATH",
        help="where to save the model (default: $NEURALGUARD_MODEL_PATH or "
        "models/threat_model.joblib)",
    )
    sub.add_argument(
        "--report-json",
        type=Path,
        metavar="PATH",
        help="also write the evaluation metrics to this JSON file",
    )
    sub.set_defaults(handler=_cmd_train)


def _add_produce(commands: Any) -> None:
    sub = commands.add_parser(
        "produce",
        help="publish traffic records to Kafka (or a JSON Lines file)",
        description="Publish traffic records - simulated, captured live or read from a pcap "
        "file - to Kafka, or to a JSON Lines file with --output.",
    )
    sub.add_argument(
        "--source",
        choices=("simulate", "live", "pcap"),
        default="simulate",
        help="where records come from (default: %(default)s)",
    )
    sub.add_argument(
        "--count",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="stop after N records; 0 = unlimited (default: %(default)s)",
    )
    sim = sub.add_argument_group("simulate options")
    sim.add_argument("--seed", type=int, help="simulator random seed (default: random)")
    sim.add_argument(
        "--attack-ratio",
        type=_fraction,
        default=0.2,
        metavar="R",
        help="fraction of attack traffic, 0-1 (default: %(default)s)",
    )
    sim.add_argument(
        "--no-pace",
        action="store_true",
        help="send as fast as possible instead of in real time (with --count, the records "
        "are dated so that the last one is now)",
    )
    live = sub.add_argument_group("live options (need root or CAP_NET_RAW)")
    live.add_argument("--interface", metavar="IFACE", help="interface to sniff (default: scapy's)")
    live.add_argument(
        "--bpf-filter",
        metavar="FILTER",
        help="BPF capture filter, needs libpcap; '' = capture everything (default: exclude "
        "NeuralGuard's own Kafka and Elasticsearch traffic, which works without libpcap)",
    )
    pcap = sub.add_argument_group("pcap options")
    pcap.add_argument("--pcap", type=Path, metavar="FILE", help="pcap/pcapng file to read")
    out = sub.add_argument_group("destination")
    out.add_argument(
        "--output",
        metavar="FILE|-",
        help="write JSON Lines to FILE ('-' = stdout) instead of Kafka",
    )
    _add_kafka_options(out)
    sub.set_defaults(handler=_cmd_produce)


def _add_detect(commands: Any) -> None:
    sub = commands.add_parser(
        "detect",
        help="consume traffic from Kafka and raise alerts",
        description="Consume traffic records from Kafka, detect threats and send throttled "
        "alerts to the console, Elasticsearch and optionally a JSON Lines file.",
    )
    _add_detection_options(sub)
    sub.add_argument(
        "--no-elasticsearch", action="store_true", help="do not index alerts into Elasticsearch"
    )
    sub.add_argument(
        "--alerts-file",
        type=Path,
        metavar="FILE",
        help="also append alerts to this JSON Lines file",
    )
    sub.add_argument(
        "--max-messages",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help="stop after N messages; 0 = run until stopped (default: %(default)s)",
    )
    sub.add_argument(
        "--max-clock-skew",
        type=_finite_float,
        metavar="SECONDS",
        help="reject records dated more than this far ahead of this host's clock; 0 = no "
        "check (default: $NEURALGUARD_MAX_CLOCK_SKEW_SECONDS or 300)",
    )
    _add_kafka_options(sub)
    sub.add_argument(
        "--es-hosts",
        metavar="URLS",
        help="comma-separated Elasticsearch URLs (default: $NEURALGUARD_ES_HOSTS "
        "or http://localhost:9200)",
    )
    sub.set_defaults(handler=_cmd_detect)


def _add_demo(commands: Any) -> None:
    sub = commands.add_parser(
        "demo",
        help="end-to-end demo on simulated traffic, no infrastructure needed",
        description="Stream simulated traffic through the detector in-process and compare "
        "detections with the ground truth. Trains a small in-memory model first if the "
        "model file does not exist.",
    )
    sub.add_argument(
        "--count",
        type=_positive_int,
        default=DEMO_COUNT,
        metavar="N",
        help="simulated packets (default: %(default)s: every attack type with the default seed)",
    )
    sub.add_argument(
        "--seed", type=int, default=DEMO_SEED, help="simulator seed (default: %(default)s)"
    )
    sub.add_argument(
        "--attack-ratio",
        type=_fraction,
        default=0.3,
        metavar="R",
        help="fraction of attack traffic, 0-1 (default: %(default)s)",
    )
    _add_detection_options(sub)
    # Size of the fallback in-memory model; internal (used by the test-suite).
    sub.add_argument(
        "--train-samples", type=_positive_int, default=DEMO_TRAIN_SAMPLES, help=argparse.SUPPRESS
    )
    sub.add_argument(
        "--train-trees", type=_positive_int, default=DEMO_TRAIN_TREES, help=argparse.SUPPRESS
    )
    sub.set_defaults(handler=_cmd_demo)


def _add_detection_options(sub: Any) -> None:
    sub.add_argument(
        "--model",
        type=Path,
        metavar="PATH",
        help="model file (default: $NEURALGUARD_MODEL_PATH or models/threat_model.joblib)",
    )
    sub.add_argument(
        "--threshold",
        type=_finite_float,
        metavar="T",
        help="threat-score threshold in (0, 1] (default: $NEURALGUARD_THREAT_THRESHOLD or 0.5)",
    )
    sub.add_argument(
        "--cooldown",
        type=_finite_float,
        metavar="SECONDS",
        help="alert cooldown per (attack type, target); 0 disables throttling "
        "(default: $NEURALGUARD_ALERT_COOLDOWN_SECONDS or 5)",
    )
    sub.add_argument(
        "--min-hits",
        type=_positive_int,
        metavar="N",
        help="alert only after N threat detections of the same (attack type, target) "
        "within the corroboration window; 1 = alert on the first "
        "(default: $NEURALGUARD_ALERT_MIN_HITS or 3)",
    )
    sub.add_argument(
        "--corroboration-window",
        type=_finite_float,
        metavar="SECONDS",
        help="time window for --min-hits "
        "(default: $NEURALGUARD_ALERT_CORROBORATION_SECONDS or 30)",
    )


def _add_kafka_options(group: Any) -> None:
    group.add_argument(
        "--bootstrap-servers",
        metavar="HOSTS",
        help="comma-separated Kafka servers (default: "
        "$NEURALGUARD_KAFKA_BOOTSTRAP_SERVERS or localhost:9092)",
    )
    group.add_argument(
        "--topic", help="Kafka topic (default: $NEURALGUARD_KAFKA_TOPIC or network-traffic)"
    )


# ---------------------------------------------------------------------- entry point


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit status."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help, --version and usage errors
        return _exit_status(exc.code)

    try:
        settings = Settings.from_env().with_overrides(
            log_level=args.log_level, log_format=args.log_format
        )
        configure_logging(settings.log_level, settings.log_format)
        return int(args.handler(args, settings) or EXIT_OK)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except BrokenPipeError:
        # Our stdout reader went away (e.g. `neuralguard produce --output - | head`).
        _silence_stdout()
        return EXIT_FAILURE
    except Exception as exc:
        if isinstance(exc, _loaded(_SERVICE_ERRORS)):
            return _fail(exc, EXIT_FAILURE)
        if isinstance(exc, (ValueError, *_loaded(_USAGE_ERRORS))):
            return _fail(exc, EXIT_USAGE)
        if isinstance(exc, OSError):
            return _fail(exc, EXIT_FAILURE)
        raise


def _fail(exc: BaseException, status: int) -> int:
    logger.debug("command failed", exc_info=exc)
    print(f"error: {_describe(exc)}", file=sys.stderr)
    return status


def _describe(exc: BaseException) -> str:
    if isinstance(exc, FileExistsError) and exc.filename and not Path(exc.filename).is_dir():
        # mkdir(parents=True, exist_ok=True) raises this only when a *file* is in the way.
        return f"Not a directory: {exc.filename}"
    if isinstance(exc, OSError) and exc.strerror:
        text = f"{exc.strerror}: {exc.filename}" if exc.filename else exc.strerror
    else:
        text = str(exc) or type(exc).__name__
    return " ".join(text.split())  # always one line


def _loaded(names: Iterable[tuple[str, str]]) -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = []
    for module_name, attribute in names:
        cls = getattr(sys.modules.get(module_name), attribute, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            types.append(cls)
    return tuple(types)


def _exit_status(code: Any) -> int:
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return EXIT_FAILURE


def _silence_stdout() -> None:
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):  # stdout without a real descriptor
        pass


# ------------------------------------------------------------------------- commands


def _cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    settings = settings.with_overrides(window_seconds=args.window, model_path=args.output)
    # Training can take a while: find out now, not afterwards, that a file can't be written.
    _check_output_path(settings.model_path)
    if args.report_json is not None:
        _check_output_path(args.report_json)

    from neuralguard.model import checksum_path
    from neuralguard.train import format_report, train_model

    result = train_model(
        n_samples=args.samples,
        seed=args.seed,
        attack_ratio=args.attack_ratio,
        window_seconds=settings.window_seconds,
        n_estimators=args.trees,
        data_path=args.data,
    )
    print(format_report(result))
    path = result.model.save(settings.model_path)
    print()
    print(f"Model {result.model.version} saved to {path}")
    print(f"Checksum written to {checksum_path(path)}")
    if args.report_json is not None:
        report = {
            "model_path": str(path),
            "model_version": result.model.version,
            "simulated": result.simulated,
            "train_size": result.train_size,
            "test_size": result.test_size,
            "class_counts": result.class_counts,
            "training": result.model.metadata.get("training"),
            "metrics": result.metrics,
        }
        _write_json(args.report_json, report)
        print(f"Metrics written to {args.report_json}")
    return EXIT_OK


def _cmd_produce(args: argparse.Namespace, settings: Settings) -> int:
    from neuralguard.consumer import install_signal_handlers
    from neuralguard.producer import (
        JsonlRecordSink,
        KafkaRecordSink,
        create_kafka_producer,
        publish,
    )

    _check_source_options(args)
    settings = settings.with_overrides(
        kafka_bootstrap_servers=_csv(args.bootstrap_servers), kafka_topic=args.topic
    )
    stop_event = threading.Event()
    # Before anything that can wait (for Kafka, say): docker stop must work at once.
    install_signal_handlers(stop_event)
    # The source first: a bad --pcap must fail before --output is truncated (or before
    # waiting for Kafka). Live capture only starts sniffing once iterated.
    records = _record_source(args, settings, stop_event)
    if args.output is not None:
        sink: Any = JsonlRecordSink(args.output)
        destination = "stdout" if args.output == "-" else args.output
    else:
        producer = create_kafka_producer(settings, stop_event=stop_event)
        if producer is None:  # stopped while waiting for Kafka
            return EXIT_OK
        sink = KafkaRecordSink(producer, settings.kafka_topic)
        destination = f"Kafka topic {settings.kafka_topic!r}"
    aborted = False
    try:
        pace = args.source == "simulate" and not args.no_pace
        logger.info("publishing %s records to %s", args.source, destination)
        sent = publish(records, sink, pace=pace, stop_event=stop_event)
    except KeyboardInterrupt:  # Ctrl+C again, e.g. while flushing to an unreachable Kafka
        aborted = True
        raise
    finally:
        if aborted and isinstance(sink, KafkaRecordSink):
            _close_quietly(sink, timeout=0)  # do not wait for the broker once more
        else:
            _close_quietly(sink)
    logger.info("published %d records to %s", sent, destination)
    return EXIT_OK


def _check_source_options(args: argparse.Namespace) -> None:
    if args.source == "pcap" and args.pcap is None:
        raise ValueError("--source pcap needs --pcap FILE")
    misplaced = {
        "--pcap": args.source != "pcap" and args.pcap is not None,
        "--interface": args.source != "live" and args.interface is not None,
        "--bpf-filter": args.source != "live" and args.bpf_filter is not None,
        "--seed": args.source != "simulate" and args.seed is not None,
    }
    for flag, wrong in misplaced.items():
        if wrong:
            raise ValueError(f"{flag} does not apply to --source {args.source}")


def _record_source(
    args: argparse.Namespace, settings: Settings, stop_event: threading.Event
) -> Iterator[dict[str, Any]]:
    limit = args.count or None
    if args.source == "simulate":
        from neuralguard.simulator import TrafficSimulator

        seed, start_time = args.seed, None  # None: the simulated stream starts now
        if args.no_pace:
            seed, start_time = _unpaced_start(args.seed, args.attack_ratio, limit)
        simulator = TrafficSimulator(
            seed=seed, attack_ratio=args.attack_ratio, start_time=start_time
        )
        return simulator.records(limit)
    from neuralguard import capture

    if args.source == "live":
        if args.bpf_filter is None:  # default: keep NeuralGuard's own traffic out
            bpf_filter, excluded = None, capture.default_exclusions(settings)
            what = f"excluding its own connections (TCP {excluded.describe()})"
        else:  # a filter of the user's own ('' = capture everything)
            bpf_filter, excluded = args.bpf_filter or None, capture.Exclusions()
            what = f"with filter {bpf_filter!r}" if bpf_filter else "without a filter"
        logger.info("capturing on %s %s", args.interface or "the default interface", what)
        return capture.live_records(
            interface=args.interface,
            bpf_filter=bpf_filter,
            count=args.count,
            stop_event=stop_event,
            exclude_tcp_endpoints=excluded.endpoints,
            exclude_tcp_ports=excluded.ports,
        )
    return itertools.islice(capture.pcap_records(args.pcap), limit)


def _unpaced_start(
    seed: int | None, attack_ratio: float, limit: int | None
) -> tuple[int | None, float | None]:
    """Seed and start time for simulated traffic sent as fast as possible.

    Generated much faster than real time, a stream starting now soon runs ahead of the
    clock, and a detector rejects records dated too far in the future. With a record
    limit the stream is started as long ago as it lasts (measured on the very same
    seeded stream, which is cheap), so its last record is dated now.
    """
    from neuralguard.simulator import TrafficSimulator

    if limit is None:
        logger.warning(
            "unpaced simulated traffic without --count runs ahead of the clock: a detector "
            "rejects records dated more than its max clock skew in the future "
            "(NEURALGUARD_MAX_CLOCK_SKEW_SECONDS, 300 s by default; 0 turns the check off)"
        )
        return seed, None
    if seed is None:
        seed = secrets.randbits(32)  # both passes must generate the same stream
    now = time.time()
    last = now
    for record in TrafficSimulator(seed=seed, attack_ratio=attack_ratio, start_time=now).records(
        limit
    ):
        last = record["timestamp"]
    return seed, max(0.0, now - (last - now))


def _cmd_detect(args: argparse.Namespace, settings: Settings) -> int:
    from neuralguard.consumer import (
        DetectionService,
        create_kafka_consumer,
        install_signal_handlers,
    )

    # First of all: docker stop must also work while the model loads and while Kafka
    # cannot be reached yet, not only once the service runs.
    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    from neuralguard.alerts import AlertThrottler
    from neuralguard.detector import Detector
    from neuralguard.model import ThreatModel
    from neuralguard.sinks import ConsoleSink, ElasticsearchSink, JsonlSink

    settings = settings.with_overrides(
        model_path=args.model,
        threat_threshold=args.threshold,
        alert_cooldown_seconds=args.cooldown,
        alert_min_hits=args.min_hits,
        alert_corroboration_seconds=args.corroboration_window,
        kafka_bootstrap_servers=_csv(args.bootstrap_servers),
        kafka_topic=args.topic,
        es_hosts=_csv(args.es_hosts),
        max_clock_skew_seconds=args.max_clock_skew,
    )
    model = ThreatModel.load(settings.model_path)
    detector = Detector(
        model,
        threshold=settings.threat_threshold,
        max_future_skew=settings.max_clock_skew_seconds or None,  # 0 = no check
    )
    throttler = AlertThrottler.from_settings(settings)

    sinks: list[Any] = [ConsoleSink()]
    try:
        if not args.no_elasticsearch:
            sinks.append(ElasticsearchSink.from_settings(settings))
        if args.alerts_file is not None:
            sinks.append(JsonlSink(args.alerts_file))
        consumer = create_kafka_consumer(settings, stop_event=stop_event)
    except BaseException:
        for sink in sinks:
            _close_quietly(sink)
        raise
    if consumer is None:  # stopped while waiting for Kafka
        for sink in sinks:
            _close_quietly(sink)
        return EXIT_OK

    logger.info(
        "detector starting: model %s, threshold %.2f, window %.1fs, cooldown %.1fs, "
        "alert after %d hit(s) in %gs, max clock skew %s, topic %r, alerts to %s",
        model.version,
        detector.threshold,
        model.window_seconds,
        throttler.cooldown_seconds,
        throttler.min_hits,
        throttler.corroboration_seconds,
        "off" if detector.max_future_skew is None else f"{detector.max_future_skew:g}s",
        settings.kafka_topic,
        ", ".join(_sink_name(sink) for sink in sinks),
    )
    service = DetectionService(detector, sinks, throttler=throttler, model_version=model.version)
    stats = service.run(consumer, stop_event=stop_event, max_messages=args.max_messages or None)
    logger.info(
        "detector stopped: %d records processed, %d threats, %d alerts emitted, %d invalid",
        stats.processed,
        stats.threats,
        service.alerts_emitted,
        stats.invalid,
    )
    return EXIT_OK


def _cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    from neuralguard.alerts import AlertThrottler
    from neuralguard.consumer import DetectionService
    from neuralguard.detector import Detector
    from neuralguard.simulator import TrafficSimulator
    from neuralguard.sinks import ConsoleSink

    settings = settings.with_overrides(
        model_path=args.model,
        threat_threshold=args.threshold,
        alert_cooldown_seconds=args.cooldown,
        alert_min_hits=args.min_hits,
        alert_corroboration_seconds=args.corroboration_window,
    )
    model, source = _demo_model(settings, args.train_samples, args.train_trees, args.seed)
    _warn_about_training_traffic(model, args.seed, args.attack_ratio)
    detector = Detector(model, threshold=settings.threat_threshold)
    throttler = AlertThrottler.from_settings(settings)
    console = ConsoleSink()
    service = DetectionService(
        detector, [console], throttler=throttler, model_version=model.version
    )
    logger.info(
        "demo: %d simulated packets (seed %d, attack ratio %.2f), threshold %.2f, model %s",
        args.count,
        args.seed,
        args.attack_ratio,
        detector.threshold,
        model.version,
    )

    summary = DemoSummary()
    simulator = TrafficSimulator(seed=args.seed, attack_ratio=args.attack_ratio)
    try:
        for batch in _chunks(simulator.records(args.count), _DEMO_BATCH_SIZE):
            detections = detector.process_many(batch)
            summary.add(detections)
            service.emit_alerts(detections)
        service.flush_alerts()  # what throttling still holds back at the end
    finally:
        _close_quietly(console)
    summary.alerts_emitted = service.alerts_emitted
    summary.alerts_suppressed = throttler.suppressed_total
    summary.alerts_uncorroborated = throttler.uncorroborated_total + throttler.pending
    summary.invalid = detector.stats.invalid

    print()
    print(
        f"NeuralGuard demo: {summary.total} simulated packets (seed {args.seed}), "
        f"threshold {detector.threshold:.2f}, model {model.version} ({source})"
    )
    print()
    print(summary.format())
    return EXIT_OK


def _demo_model(
    settings: Settings, n_samples: int, n_trees: int, demo_seed: int
) -> tuple[Any, str]:
    """The saved model, or - when the file does not exist - a small one trained now, on
    simulated streams other than the demo's (``demo_seed``)."""
    from neuralguard.model import ThreatModel

    path = settings.model_path
    if path.exists():
        return ThreatModel.load(path), f"loaded from {path}"

    from neuralguard.train import simulated_plan, train_model

    seed = DEMO_TRAIN_SEED
    while any(stream_seed == demo_seed for stream_seed, _ in simulated_plan(n_samples, seed)[0]):
        seed += 1
    logger.info(
        "model file %s not found; training a small in-memory model (%d samples, %d trees, "
        "seed %d) - run 'neuralguard train' to create a proper one",
        path,
        n_samples,
        n_trees,
        seed,
    )
    result = train_model(
        n_samples=n_samples,
        n_estimators=n_trees,
        seed=seed,
        window_seconds=settings.window_seconds,
    )
    return result.model, "trained in memory"


def _warn_about_training_traffic(model: Any, seed: int, attack_ratio: float) -> None:
    """Warn when the demo's stream is one the model was trained on: its numbers would
    say nothing about how the model does on traffic it has not seen."""
    training = getattr(model, "metadata", {}).get("training")
    if not isinstance(training, dict) or training.get("source") != "simulated":
        return
    seeds = training.get("seeds") or [training.get("seed")]
    if seed in seeds and training.get("attack_ratio") == attack_ratio:
        logger.warning(
            "the demo stream (seed %d, attack ratio %g) is the model's own training "
            "traffic, so its detection rate is flattering; use another --seed",
            seed,
            attack_ratio,
        )


# --------------------------------------------------------------------- demo summary


@dataclass
class _LabelCounts:
    packets: int = 0
    flagged: int = 0
    correct_type: int = 0


@dataclass
class DemoSummary:
    """Ground truth versus detections for the ``demo`` command's summary table."""

    by_label: dict[str, _LabelCounts] = field(default_factory=dict)
    alerts_emitted: int = 0
    alerts_suppressed: int = 0
    alerts_uncorroborated: int = 0
    invalid: int = 0

    def add(self, detections: Iterable[Any]) -> None:
        for detection in detections:
            label = detection.record.get("label") or "unlabelled"
            counts = self.by_label.setdefault(label, _LabelCounts())
            counts.packets += 1
            if detection.is_threat:
                counts.flagged += 1
                if detection.attack_type == label:
                    counts.correct_type += 1

    @property
    def total(self) -> int:
        return sum(counts.packets for counts in self.by_label.values())

    def _sum(self, labels: Iterable[str], attribute: str) -> int:
        return sum(
            getattr(self.by_label[label], attribute) for label in labels if label in self.by_label
        )

    def format(self) -> str:
        """The summary as a plain-text table."""
        known = [label for label in LABELS if label in self.by_label]
        others = sorted(set(self.by_label) - set(LABELS))
        rows = [("True label", "Packets", "Flagged as threat", "Correct attack type")]
        for label in (*known, *others):
            counts = self.by_label[label]
            correct = (
                _count_pct(counts.correct_type, counts.flagged) if label in ATTACK_TYPES else "-"
            )
            rows.append(
                (label, str(counts.packets), _count_pct(counts.flagged, counts.packets), correct)
            )
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        rule = "  ".join("-" * width for width in widths)
        lines = [_table_row(rows[0], widths), rule]
        lines += [_table_row(row, widths) for row in rows[1:]]
        lines.append(rule)

        attacks = self._sum(ATTACK_TYPES, "packets")
        attacks_flagged = self._sum(ATTACK_TYPES, "flagged")
        normal = self._sum([NORMAL_LABEL], "packets")
        normal_flagged = self._sum([NORMAL_LABEL], "flagged")
        overall = [
            ("Detection rate (attack packets flagged)", _rate(attacks_flagged, attacks)),
            ("False-positive rate (normal packets flagged)", _rate(normal_flagged, normal)),
            ("Alerts emitted", str(self.alerts_emitted)),
            ("Alerts suppressed by throttling", str(self.alerts_suppressed)),
            ("Lone detections not corroborated", str(self.alerts_uncorroborated)),
        ]
        if self.invalid:
            overall.append(("Invalid records skipped", str(self.invalid)))
        width = max(len(name) for name, _ in overall)
        lines += ["", "Overall:"]
        lines += [f"  {name + ':':<{width + 1}}  {value}" for name, value in overall]
        lines.append("")
        missing = [label for label in ATTACK_TYPES if label not in self.by_label]
        if attacks and missing:  # the simulator cycles through every type over time
            lines.append(
                f"Not in this run: {', '.join(missing)} - a longer run (--count) "
                "includes every attack type."
            )
        lines.append(
            "'Correct attack type' counts flagged packets whose predicted attack type "
            "matches the true label."
        )
        return "\n".join(lines)


def _table_row(cells: Sequence[str], widths: Sequence[int]) -> str:
    first, *rest = cells
    parts = [first.ljust(widths[0])]
    parts += [cell.rjust(width) for cell, width in zip(rest, widths[1:], strict=True)]
    return "  ".join(parts).rstrip()


def _percent(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def _count_pct(part: int, whole: int) -> str:
    return f"{part} ({_percent(part, whole)})"


def _rate(part: int, whole: int) -> str:
    return f"{_percent(part, whole)} ({part} of {whole})" if whole else "n/a (no packets)"


# ------------------------------------------------------------------------- helpers


def _chunks(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    iterator = iter(items)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def _csv(value: str | None) -> tuple[str, ...] | None:
    """A comma-separated flag value as a tuple (``None`` when the flag was not given)."""
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _close_quietly(resource: Any, **kwargs: Any) -> None:
    try:
        resource.close(**kwargs)
    except Exception:
        logger.exception("error while closing %s", type(resource).__name__)


def _sink_name(sink: Any) -> str:
    return type(sink).__name__.removesuffix("Sink").lower() or type(sink).__name__


def _check_output_path(path: Path) -> None:
    """Raise ``OSError`` if ``path`` clearly cannot be written: it is a directory, or its
    nearest existing ancestor is not a writable directory. Nothing is created."""
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), str(ancestor))
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(ancestor))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_plain(payload), indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _plain(value: Any) -> Any:
    """``value`` with numpy scalars/arrays turned into plain Python values and
    non-finite floats into ``None``, so it is strict JSON."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        for converter in ("tolist", "item"):
            method: Callable[[], Any] | None = getattr(value, converter, None)
            if callable(method):
                return _plain(method())
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# -------------------------------------------------------------------- arg types


def _positive_int(text: str) -> int:
    value = _int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = _int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text!r}")
    return value


def _fraction(text: str) -> float:
    value = _finite_float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return value
