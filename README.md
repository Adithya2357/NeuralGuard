# NeuralGuard: AI-Powered Network Intrusion Detection

[![CI](https://github.com/Adithya2357/NeuralGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/Adithya2357/NeuralGuard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

NeuralGuard watches network traffic in real time and flags attacks such as port scans,
stealth scans and SYN/UDP/ICMP floods. Packets are captured with **scapy**, streamed through
**Kafka**, scored by a **scikit-learn** model that looks at *behaviour over time* (not just
single packets), and every alert is indexed into **Elasticsearch** and charted in **Grafana**.

```text
$ neuralguard demo
True label    Packets  Flagged as threat  Correct attack type
------------  -------  -----------------  -------------------
normal           4089           3 (0.1%)                    -
port_scan         129        124 (96.1%)          123 (99.2%)
stealth_scan      671        664 (99.0%)         664 (100.0%)
syn_flood         186        179 (96.2%)         179 (100.0%)
udp_flood         241        233 (96.7%)         233 (100.0%)
icmp_flood        684       684 (100.0%)         684 (100.0%)

Detection rate (attack packets flagged):       98.6% (1884 of 1911)
False-positive rate (normal packets flagged):  0.1% (3 of 4089)
Alerts emitted:                                13
Alerts suppressed by throttling:               1873
Lone detections not corroborated:              1
```

---

## Table of contents

- [How it works](#how-it-works)
- [Features](#features)
- [Quick start: no infrastructure (1 minute)](#quick-start-no-infrastructure-1-minute)
- [Full stack: Kafka + Elasticsearch + Grafana](#full-stack-kafka--elasticsearch--grafana)
- [Capturing real traffic](#capturing-real-traffic)
- [Detection in detail](#detection-in-detail)
- [Model performance](#model-performance)
- [Training on your own data](#training-on-your-own-data)
- [Configuration](#configuration)
- [Command reference](#command-reference)
- [Development](#development)
- [Security notes](#security-notes)
- [Upgrading from v0.1](#upgrading-from-v01)
- [Roadmap](#roadmap)
- [License](#license)

---

## How it works

```mermaid
flowchart LR
    subgraph sources[Traffic sources]
        live[Live capture<br/>scapy sniffer]
        pcap[pcap / pcapng file]
        sim[Traffic simulator<br/>normal LAN + 5 attack types]
    end
    live --> produce[neuralguard produce]
    pcap --> produce
    sim --> produce
    produce -- JSON records<br/>topic network-traffic --> kafka[(Kafka)]
    kafka --> detect
    subgraph detect[neuralguard detect]
        norm[Validate record] --> feat[Sliding-window<br/>feature extractor]
        feat --> model[RandomForest<br/>threat model]
        model --> throttle[Corroboration +<br/>throttling]
    end
    throttle --> es[(Elasticsearch<br/>threat-detection)]
    throttle --> console[Console / JSON Lines]
    es --> grafana[Grafana dashboard]
```

1. **`neuralguard produce`** turns packets (live, from a pcap, or simulated) into small JSON
   *traffic records* and publishes them to Kafka, keyed by source IP.
2. **`neuralguard detect`** consumes the topic, validates each record, computes features over
   a sliding time window, and asks the model for a **threat score** (the probability that the
   packet is part of an attack) and the most likely **attack type**.
3. Threats become **alerts** with a severity. An alert is raised only once the same
   (attack type, target) has been detected **3 times within 30 s** (corroboration). Scans
   and floods are many packets, while a lone misfire of the model is not. During a flood,
   alerts are then throttled to one per (attack type, target) every few seconds, and each
   alert says how many detections it stands for.
4. Alerts go to the console, to Elasticsearch (explicit index mapping), and optionally to a
   JSON Lines file. The provisioned **Grafana** dashboard reads them from Elasticsearch.

## Features

- **Behavioural detection.** Per-source, per-destination and per-pair sliding-window
  features (distinct ports and hosts contacted, ports probed on this target, SYN ratio,
  fan-in of distinct sources) make scans and floods visible, even though each individual
  scan packet looks innocent.
- **Low-noise alerting.** Corroboration (N detections of the same attack on the same target
  before alerting) removes most false alarms without missing attacks, and throttling turns a
  flood of thousands of packets into a handful of alerts. No threat is lost from the
  accounting.
- **One feature pipeline for training and detection.** The trainer and the detector share
  the same record validation and feature extractor, and the model file stores its window
  length, so there is no train/serve skew.
- **Five attack classes plus normal traffic.** `port_scan`, `stealth_scan` (NULL/FIN/XMAS/ACK/
  Maimon), `syn_flood`, `udp_flood` (including DNS/NTP/SSDP/memcached reflection) and
  `icmp_flood`, each with a threat score, a predicted type and a severity
  (`low` / `medium` / `high` / `critical`).
- **Realistic traffic simulator.** It models an office LAN (web, DNS, NTP, SSH, NAS,
  streaming, video calls, pings, ARP, discovery protocols, monitoring) with randomised attack
  episodes. Normal traffic deliberately includes SYNs, pings and bursts, so "SYN = attack"
  cannot be learned. You can train and demo the whole system without root access or a live
  network.
- **Live capture and pcap replay.** IPv4/IPv6, TCP/UDP/ICMP/ICMPv6, ARP. The sniffer excludes
  NeuralGuard's own Kafka/Elasticsearch connections, so it never captures its own traffic.
- **Resilient service.** Retries while Kafka starts. Malformed messages are skipped and
  counted, never fatal. Elasticsearch outages are buffered with backoff and a bounded buffer.
  SIGINT/SIGTERM shut down gracefully, flushing pending alerts.
- **Production-minded packaging.** `pyproject.toml` with a `neuralguard` CLI, `NEURALGUARD_*`
  environment configuration, text or JSON logs, Docker image (non-root, model baked in
  read-only), `docker compose` stack bound to `127.0.0.1`, and GitHub Actions CI (ruff, tests
  on Python 3.10-3.13, bandit, pip-audit, Docker build).
- **Well tested.** 600+ unit and integration tests, about 15 seconds plus about 20 seconds of
  model-quality tests (`-m "not slow"` skips them). No Kafka or Elasticsearch needed.

## Quick start: no infrastructure (1 minute)

Requires **Python 3.10+**.

```bash
git clone https://github.com/Adithya2357/NeuralGuard.git
cd NeuralGuard
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

neuralguard demo        # trains a model in memory if none exists (~25 s), then runs
```

`demo` streams simulated traffic through the real detector in-process and prints the alerts
plus a table comparing detections with the ground truth (see the top of this page).

To train and save a model with a full evaluation report:

```bash
neuralguard train       # -> models/threat_model.joblib (+ .sha256), about 30 s
```

## Full stack: Kafka + Elasticsearch + Grafana

Requires Docker with Compose v2.

**Option A: everything in containers** (detector and simulator included):

```bash
docker compose --profile demo up --build      # or: make docker-demo
```

Open Grafana at <http://127.0.0.1:3000> (user `admin`, password `admin` unless you set
`GRAFANA_ADMIN_PASSWORD`). The **NeuralGuard - Threat Detection** dashboard is the home page
and fills up within a few seconds.

**Option B: infrastructure in containers, NeuralGuard on your machine:**

```bash
make up                        # Kafka (localhost:9092), Elasticsearch (:9200), Grafana (:3000)
neuralguard train              # once
neuralguard detect             # terminal 1: consume, detect, index alerts
neuralguard produce            # terminal 2: publish simulated traffic in real time
make down                      # stop (data volumes are kept)
```

The detector only needs to be started before the traffic. A consumer group with no committed
offset starts at the latest message. After a restart it resumes from where it left off.

The dashboard shows total and critical alerts, suppressed duplicates, alerts over time by
attack type, attack-type and severity breakdowns, the top source IPs, the threat-score trend
and a table of the latest alerts.

## Capturing real traffic

```bash
# Live capture (needs root or CAP_NET_RAW)
sudo .venv/bin/neuralguard produce --source live --interface eth0

# Replay a capture file
neuralguard produce --source pcap --pcap capture.pcapng

# Record records to a file instead of Kafka ('-' = stdout)
neuralguard produce --source pcap --pcap capture.pcap --output traffic.jsonl
```

By default the sniffer leaves out NeuralGuard's **own** connections: TCP to and from the
exact Kafka and Elasticsearch host:port pairs in your settings. Otherwise every published
record would generate more captured packets (a feedback loop). Other traffic on those port
numbers is still captured, so an attacker cannot hide behind them. `--bpf-filter 'tcp or
udp'` sets a custom BPF filter (needs libpcap, e.g. `apt install libpcap0.8`), and
`--bpf-filter ''` captures everything.

Packet lengths are normalised to an Ethernet frame size, so VLAN tags, Linux "cooked"
captures and raw-IP pcaps produce the same `length` feature as the training data. A sniffer
that dies (for example after an interface flap) makes `produce` exit with status 1, so
`restart: on-failure` supervisors restart it. In a container, live capture needs root:
`docker run --user root --network host ...`.

> The bundled model is trained on **simulated** traffic. Expect false positives on a real
> network until you retrain on labelled captures from it (see
> [Training on your own data](#training-on-your-own-data)).

## Detection in detail

**Traffic record**: what the producer sends and the detector validates:

```json
{"timestamp": 1730647800.123, "source_ip": "203.0.113.5", "destination_ip": "192.168.1.20",
 "protocol": "TCP", "source_port": 40000, "destination_port": 22, "length": 60, "ttl": 64,
 "tcp_flags": "S"}
```

**Features**: 22 per packet, computed by one stateful extractor:

| Group | Features |
|---|---|
| Packet | protocol (TCP/UDP/ICMP/ARP one-hot), `length`, `ttl`, `source_port`, `destination_port`, `dst_port_well_known`, TCP flags SYN/ACK/FIN/RST/PSH/URG |
| Source window (last 10 s) | `src_packet_count`, `src_unique_dst_ports`, `src_unique_dst_ips`, `src_syn_ratio` |
| Destination window (last 10 s) | `dst_packet_count`, `dst_unique_src_ips` |
| Source-to-destination pair (last 10 s) | `pair_unique_dst_ports` (ports this source probed on this target) |

Window time is driven by packet timestamps, not the wall clock, so replays behave exactly
like live traffic. Live detection rejects records dated more than 5 minutes ahead of the
host clock, so a forged timestamp cannot freeze the windows. If time jumps backwards (for
example when an older capture is replayed), the windows restart. Memory is bounded even
under spoofed-source floods: hosts whose window has expired are forgotten, and per-host
windows are capped.

**Model**: a multiclass `RandomForestClassifier` (`normal` plus 5 attack types), trained by
default on 240,000 simulated packets from 4 independent streams. The first 10 packets of
each attack episode are left out of training, because they carry no behavioural evidence
yet: a flood's first SYN looks exactly like a new visitor's. `threat_score = 1 - P(normal)`.
A packet is a threat when the score reaches the threshold (default `0.5`). Severity:
`critical` >= 0.9, `high` >= 0.75, `medium` >= 0.6, else `low`.

**Alerts**: two stages, both on packet time.

1. **Corroboration.** The first alert for an (attack type, destination IP) needs
   `--min-hits` detections (default 3) within `--corroboration-window` (default 30 s).
   Detections that never reach that are counted as `uncorroborated` and never alerted.
   Once an attack is corroborated, it stays so while detections keep coming.
2. **Throttling.** At most one alert per (attack type, destination IP) per cooldown
   (default 5 s). `suppressed_count` carries the number of detections an alert stands for,
   and the remaining ones are flushed as a summary when the flood ends or at shutdown.

Every threat is accounted for: alerts + suppressed + uncorroborated = threats. On the
held-out test streams, corroboration cut false alerts by 50-73% while every one of the 50
attack episodes still raised an alert. Scans of up to one probe every 3 s are still
alerted. Slower ones are beyond what the 10 s feature window can see anyway. Use
`--min-hits 1` to alert on the first detection.

**Model file**: a versioned joblib bundle with the feature names, window length, training
parameters and metrics, plus a `sha256sum`-style sidecar that is verified **before**
unpickling.

## Model performance

`neuralguard train` with defaults: 240,000 simulated packets from 4 independent streams
for training, evaluated on **4 separate held-out streams** (60,000 packets, different seeds),
so the hosts, timings and all 50 attack episodes are unseen:

| Metric | Value |
|---|---|
| Detection rate (attack packets flagged) | 98.4% (97.4-98.8% across streams) |
| False-positive rate (normal packets flagged) | 0.12% (0.10-0.18% across streams) |
| Attack episodes detected | **50 of 50** (median 3.5 packets before the first detection) |
| False alerts on attack-free traffic | about 127 per hour (busy simulated LAN, after corroboration and throttling) |
| ROC-AUC (threat vs normal) | 0.9998 |
| Attack-type accuracy (flagged attacks) | 99.4% |
| Per-class recall | port_scan 0.969, stealth_scan 0.965, syn_flood 0.985, udp_flood 0.992, icmp_flood 0.992 |
| Training time | about 30 s (4 cores) |

The packets that are missed are mostly the first few of each episode, before the window
features have any evidence.

**Read these numbers honestly.** They are measured on synthetic traffic from the same
simulator family. They show the pipeline and the behavioural features work, not how the model
will do on your network. For real deployments, retrain on labelled captures from your
environment or a public dataset.

## Training on your own data

Write labelled records (one JSON object per line, schema above, plus a `label` of `normal`,
`port_scan`, `stealth_scan`, `syn_flood`, `udp_flood` or `icmp_flood`) and train on them:

```bash
neuralguard produce --source pcap --pcap benign.pcap --output benign.jsonl  # then add labels
neuralguard train --data labelled.jsonl --output models/my_model.joblib
```

Records are sorted by time, and the data is split **chronologically** (the last 25% is the
test set). A random split would leak, because neighbouring packets share almost identical
window features.

## Configuration

Every setting can be set with an environment variable. Command-line flags override them.

| Variable | Default | Meaning |
|---|---|---|
| `NEURALGUARD_KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | comma-separated Kafka servers |
| `NEURALGUARD_KAFKA_TOPIC` | `network-traffic` | topic for traffic records |
| `NEURALGUARD_KAFKA_GROUP_ID` | `neuralguard-detector` | consumer group of the detector |
| `NEURALGUARD_ES_HOSTS` | `http://localhost:9200` | comma-separated Elasticsearch URLs |
| `NEURALGUARD_ES_INDEX` | `threat-detection` | alert index |
| `NEURALGUARD_ES_USERNAME` / `NEURALGUARD_ES_PASSWORD` | unset | basic auth |
| `NEURALGUARD_ES_API_KEY` | unset | API key auth (instead of basic auth) |
| `NEURALGUARD_ES_VERIFY_CERTS` | `true` | verify TLS certificates |
| `NEURALGUARD_ES_CA_CERTS` | unset | CA bundle for a private CA |
| `NEURALGUARD_MODEL_PATH` | `models/threat_model.joblib` | model file |
| `NEURALGUARD_THREAT_THRESHOLD` | `0.5` | threat-score threshold, in (0, 1] |
| `NEURALGUARD_WINDOW_SECONDS` | `10` | feature window used by `train` |
| `NEURALGUARD_ALERT_COOLDOWN_SECONDS` | `5` | alert throttling per (type, target); `0` disables |
| `NEURALGUARD_ALERT_MIN_HITS` | `3` | detections of the same (type, target) needed before the first alert; `1` = alert at once |
| `NEURALGUARD_ALERT_CORROBORATION_SECONDS` | `30` | time window for `ALERT_MIN_HITS` |
| `NEURALGUARD_MAX_CLOCK_SKEW_SECONDS` | `300` | `detect` rejects records dated further ahead of the host clock; `0` disables |
| `NEURALGUARD_LOG_LEVEL` | `INFO` | `DEBUG` ... `CRITICAL` |
| `NEURALGUARD_LOG_FORMAT` | `text` | `text` or `json` (one object per line) |

Secrets are never logged or shown in `repr()`, including credentials embedded in an
Elasticsearch URL. Prefer the username/password or API key variables.

## Command reference

| Command | What it does | Useful options |
|---|---|---|
| `neuralguard train` | train, evaluate on held-out data, print a report, save the model | `--samples`, `--trees`, `--window`, `--data FILE.jsonl`, `--output`, `--report-json` |
| `neuralguard produce` | publish records to Kafka (or a JSON Lines file) | `--source simulate\|live\|pcap`, `--count`, `--seed`, `--attack-ratio`, `--no-pace`, `--interface`, `--bpf-filter`, `--pcap`, `--output` |
| `neuralguard detect` | consume, detect, alert | `--model`, `--threshold`, `--cooldown`, `--min-hits`, `--corroboration-window`, `--max-clock-skew`, `--no-elasticsearch`, `--alerts-file`, `--max-messages` |
| `neuralguard demo` | the whole pipeline in-process, no infrastructure | `--count`, `--seed`, `--attack-ratio`, `--threshold`, `--cooldown`, `--min-hits`, `--model` |

Global options: `--log-level`, `--log-format text|json`, `--version`. Exit codes: `0` ok,
`1` service or I/O error, `2` bad input or configuration, `130` interrupted.
`python -m neuralguard` works too.

## Development

```bash
make venv && source .venv/bin/activate   # package + dev tools in .venv
make test        # pytest with coverage
make lint        # ruff
make security    # bandit + pip-audit
make help        # every target
```

CI (`.github/workflows/ci.yml`) runs lint, the test suite on Python 3.10-3.13, bandit and
pip-audit, and validates `docker-compose.yml` and the Docker image (build plus a smoke test).
Dependabot keeps pip packages, GitHub Actions and Docker images up to date.

```text
neuralguard/
  schema.py      traffic-record schema + validation (the single gate for every record)
  features.py    sliding-window feature extraction (shared by training and detection)
  simulator.py   synthetic labelled traffic: office LAN + attack episodes
  capture.py     scapy packet -> record, live capture, pcap reading
  producer.py    Kafka / JSON Lines record sinks, real-time pacing
  model.py       ThreatModel: training, prediction, evaluation, verified save/load
  train.py       datasets, held-out evaluation, training report
  detector.py    record -> features -> model -> Detection
  alerts.py      alert documents, throttling
  sinks.py       Elasticsearch (index mapping, bulk, backoff), console, JSON Lines
  consumer.py    Kafka detection service, graceful shutdown
  cli.py         the neuralguard command
deploy/grafana/  provisioned datasource + dashboard
tests/           unit and integration tests (no external services needed)
```

## Security notes

- **The compose stack is for local use.** Kafka and Elasticsearch run without
  authentication, and every port is published on `127.0.0.1` only. Before exposing anything,
  enable Elasticsearch security (instructions in `docker-compose.yml`), set
  `NEURALGUARD_ES_USERNAME`/`PASSWORD` or `NEURALGUARD_ES_API_KEY` and
  `NEURALGUARD_ES_CA_CERTS`, and change the Grafana admin password.
- **Model files are pickles. Only load models you trust.** NeuralGuard verifies the
  `.sha256` sidecar before unpickling. In the Docker image the model is owned by root and is
  read-only for the unprivileged runtime user.
- **Traffic is untrusted input.** Every record is validated: types, ranges, canonical IP
  syntax (IPv6 zone IDs are rejected, since they could smuggle control characters into logs),
  and plausible timestamps (integers too large for a float, and dates far in the future, are
  rejected). One malformed message is skipped and counted. It can never crash the detector.
  Logged payloads are truncated, console alerts escape control characters, and all in-memory
  state (windows, throttling and corroboration keys, buffers, capture queue) is bounded.
- **The sensor cannot be evaded through NeuralGuard's own ports.** Only its exact Kafka and
  Elasticsearch connections are excluded from capture, not every packet on those port
  numbers.

## Upgrading from v0.1

v0.2 is a rewrite. The old scripts had a syntax error, a missing `flags` field, a model
trained on random noise and hardcoded `/home/kali/...` paths. They map to the new CLI like
this:

| v0.1 | v0.2 |
|---|---|
| `python data_ingestion/kafka_producer.py` | `sudo neuralguard produce --source live` (or `--source simulate`) |
| `python model/model_training.py` | `neuralguard train` |
| `python real_time_detection/detection_consumer.py` | `neuralguard detect` |
| `logging/log_to_elasticsearch.py` | built into `detect` (`neuralguard/sinks.py`) |
| Zookeeper + Kafka started by hand | `docker compose up -d` (Kafka in KRaft mode, no Zookeeper) |

## Roadmap

- Sequence models (LSTM / temporal CNN) over per-host packet sequences
- Training and evaluation on public datasets (CIC-IDS2017, UNSW-NB15)
- More attack classes: brute force (SSH/RDP), DNS tunnelling, data exfiltration
- Alerting integrations (Slack, email, webhooks) from Grafana

## License

MIT, see [LICENSE](LICENSE).
