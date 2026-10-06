"""Alert documents and alert throttling.

:func:`build_alert_document` turns a threat :class:`~neuralguard.detector.Detection` into
the JSON document that is indexed into Elasticsearch (and charted by Grafana). It only
contains plain Python types - never numpy scalars - so any JSON encoder accepts it::

    {"@timestamp": "2024-11-03T15:30:00.123Z",   # packet time, UTC
     "detected_at": "2024-11-03T15:30:00.456Z",  # when the detector saw it, UTC
     "source_ip": "203.0.113.5", "destination_ip": "192.168.1.20",
     "protocol": "TCP", "source_port": 40000, "destination_port": 22,
     "length": 60, "ttl": 64, "tcp_flags": "S",
     "threat_score": 0.9731, "attack_type": "port_scan", "severity": "critical",
     "suppressed_count": 12, "model_version": "3f2a9c1b7d4e",
     "features": {"is_tcp": 1.0, ...},
     "simulated_label": "port_scan"}              # only when the record had a label

:class:`AlertThrottler` keeps alert volume sane: during a flood *every* packet is a
detection, but one alert per (attack type, target) every few seconds - carrying the
number of alerts it stands for - is what an analyst (and Elasticsearch) can cope with.
:meth:`AlertThrottler.flush` reports what is still held back once a key goes quiet (and
at shutdown), so every threat is accounted for exactly once: summed over all alerts,
``1 + suppressed_count`` equals the number of threat detections.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from neuralguard.detector import Detection, severity_for
from neuralguard.features import features_as_dict

DEFAULT_MAX_KEYS = 10_000

ThrottleKey = tuple[str | None, str | None]


def build_alert_document(
    detection: Detection,
    *,
    model_version: str,
    suppressed_count: int = 0,
    now: float | None = None,
) -> dict[str, Any]:
    """The alert document for a threat detection.

    ``suppressed_count`` is the number of similar alerts the throttler suppressed since
    the previous one (see :meth:`AlertThrottler.check`). ``now`` (epoch seconds) is the
    detection time, injectable for tests; it defaults to the current time. Raises
    ``ValueError`` when ``detection`` is not a threat.
    """
    if not detection.is_threat or detection.attack_type is None:
        raise ValueError("cannot build an alert document for a detection that is not a threat")
    if suppressed_count < 0:
        raise ValueError(f"suppressed_count must be >= 0, got {suppressed_count}")

    record = detection.record
    detected_at = _iso_utc(time.time() if now is None else now)
    if detected_at is None:
        raise ValueError(f"now is not a representable epoch time: {now!r}")
    doc: dict[str, Any] = {
        # A packet time beyond year 9999 cannot be a real capture time: fall back to now.
        "@timestamp": _iso_utc(record["timestamp"]) or detected_at,
        "detected_at": detected_at,
        "source_ip": _optional_str(record.get("source_ip")),
        "destination_ip": _optional_str(record.get("destination_ip")),
        "protocol": str(record["protocol"]),
        "source_port": int(record["source_port"]),
        "destination_port": int(record["destination_port"]),
        "length": int(record["length"]),
        "ttl": int(record["ttl"]),
        "tcp_flags": str(record.get("tcp_flags") or ""),
        "threat_score": round(float(detection.threat_score), 4),
        "attack_type": str(detection.attack_type),
        "severity": str(detection.severity or severity_for(detection.threat_score)),
        "suppressed_count": int(suppressed_count),
        "model_version": str(model_version),
        "features": features_as_dict(detection.features),
    }
    label = record.get("label")
    if label is not None:
        doc["simulated_label"] = str(label)
    return doc


def throttle_key(detection: Detection) -> ThrottleKey:
    """What makes two alerts "the same" for throttling: (attack type, destination IP)."""
    return detection.attack_type, detection.record.get("destination_ip")


class _KeyState:
    __slots__ = ("last_emitted", "latest_suppressed", "suppressed")

    def __init__(self, last_emitted: float) -> None:
        self.last_emitted = last_emitted
        self.suppressed = 0
        self.latest_suppressed: Detection | None = None


class AlertThrottler:
    """Emits at most one alert per (attack type, destination IP) per cooldown.

    Time is *packet* time (the record's ``timestamp``), not the wall clock, so
    throttling is deterministic and replaying a capture throttles exactly like live
    traffic. ``cooldown_seconds=0`` disables throttling. At most ``max_keys`` keys are
    remembered: beyond that, keys whose cooldown has expired are forgotten first, then
    the ones whose last alert is oldest. Not thread-safe.
    """

    def __init__(self, cooldown_seconds: float, max_keys: int = DEFAULT_MAX_KEYS) -> None:
        cooldown = float(cooldown_seconds)
        if not math.isfinite(cooldown) or cooldown < 0:
            raise ValueError(f"cooldown_seconds must be a finite number >= 0, got {cooldown}")
        if max_keys < 1:
            raise ValueError(f"max_keys must be >= 1, got {max_keys}")
        self.cooldown_seconds = cooldown
        self.max_keys = int(max_keys)
        # Ordered by last emission (oldest first), which is what pruning needs.
        self._keys: OrderedDict[ThrottleKey, _KeyState] = OrderedDict()
        self._suppressed_total = 0
        self._latest = float("-inf")

    @property
    def suppressed_total(self) -> int:
        """Threats suppressed since this throttler was created, not counting those that
        :meth:`flush` later turned into alerts (so alerts + this = threats)."""
        return self._suppressed_total

    @property
    def tracked_keys(self) -> int:
        """Number of (attack type, destination IP) keys currently remembered."""
        return len(self._keys)

    def check(self, detection: Detection) -> int | None:
        """Decide whether ``detection`` should produce an alert.

        Returns ``None`` to suppress it, otherwise the number of alerts suppressed for
        its key since the last emitted one (0 for the first). Throttling disabled
        (cooldown 0) always returns 0. A detection that is not a threat never produces
        an alert: ``None``, without counting as suppressed.
        """
        if not detection.is_threat:
            return None
        if self.cooldown_seconds == 0:
            return 0

        timestamp = float(detection.record["timestamp"])
        self._latest = max(self._latest, timestamp)
        key = throttle_key(detection)
        state = self._keys.get(key)
        if state is not None and timestamp - state.last_emitted < self.cooldown_seconds:
            state.suppressed += 1
            state.latest_suppressed = detection
            self._suppressed_total += 1
            return None

        suppressed = state.suppressed if state is not None else 0
        self._keys[key] = _KeyState(timestamp)
        self._keys.move_to_end(key)
        self._prune()
        return suppressed

    def flush(
        self, now: float | None = None, *, everything: bool = False
    ) -> list[tuple[Detection, int]]:
        """Alerts still held back, for keys whose cooldown is over (or all with
        ``everything``, e.g. at shutdown).

        Without this, the suppressed tail of a flood that has ended would never be
        reported. Returns ``(detection, suppressed_count)`` pairs: the latest suppressed
        detection of a key, to be sent as an alert standing for ``suppressed_count``
        others suppressed before it. Its key then counts as alerted at that detection's
        time. ``now`` is the latest packet time seen (also non-threat packets), which
        decides whose cooldown is over. Expired keys with nothing held back are
        forgotten.
        """
        if now is not None:
            self._latest = max(self._latest, float(now))
        expiry = self._latest - self.cooldown_seconds
        summaries: list[tuple[Detection, int]] = []
        for key in list(self._keys):
            state = self._keys[key]
            if not everything and state.last_emitted > expiry:
                continue
            detection = state.latest_suppressed
            if detection is None:
                del self._keys[key]  # nothing held back, and its cooldown is over
                continue
            summaries.append((detection, state.suppressed - 1))
            self._suppressed_total -= 1  # that one is an alert now
            self._keys[key] = _KeyState(float(detection.record["timestamp"]))
            self._keys.move_to_end(key)
        return summaries

    def _prune(self) -> None:
        keys = self._keys
        if len(keys) <= self.max_keys:
            return
        # Keys are in emission order, so expired ones are at the front (only approximately
        # when packets arrive out of order - good enough for a memory bound).
        expiry = self._latest - self.cooldown_seconds
        while len(keys) > 1:
            state = next(iter(keys.values()))
            if state.last_emitted > expiry:
                break
            keys.popitem(last=False)
        while len(keys) > self.max_keys:
            keys.popitem(last=False)


def _iso_utc(epoch_seconds: float) -> str | None:
    """Epoch seconds as UTC ISO 8601 with millisecond precision and a "Z" suffix.

    ``None`` when the time cannot be represented (outside years 1-9999).
    """
    try:
        moment = datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
