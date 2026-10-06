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
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Any

from neuralguard.config import Settings
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
    the ones whose last alert is oldest; what a forgotten key still held back is reported
    by the next :meth:`flush`. Not thread-safe.

    **Corroboration** (``min_hits`` > 1): a key raises its first alert only once it has
    ``min_hits`` threat detections within ``corroboration_seconds``. Scans and floods are
    many packets by nature, while a lone misfire of the model is not, so this removes most
    false alarms. The detections held back meanwhile are folded into that first alert's
    suppressed count. Those that never reach ``min_hits`` are dropped and counted in
    :attr:`uncorroborated_total`. A key stays corroborated while its detections keep
    coming at most ``corroboration_seconds`` apart, so an ongoing attack is never held
    back again. In total: alerts + :attr:`suppressed_total` + :attr:`uncorroborated_total`
    + :attr:`pending` = threats.
    """

    def __init__(
        self,
        cooldown_seconds: float,
        max_keys: int = DEFAULT_MAX_KEYS,
        *,
        min_hits: int = 1,
        corroboration_seconds: float = 30.0,
    ) -> None:
        cooldown = float(cooldown_seconds)
        if not math.isfinite(cooldown) or cooldown < 0:
            raise ValueError(f"cooldown_seconds must be a finite number >= 0, got {cooldown}")
        if max_keys < 1:
            raise ValueError(f"max_keys must be >= 1, got {max_keys}")
        if isinstance(min_hits, bool) or not isinstance(min_hits, int) or min_hits < 1:
            raise ValueError(f"min_hits must be an integer >= 1, got {min_hits!r}")
        corroboration = float(corroboration_seconds)
        if not math.isfinite(corroboration) or corroboration <= 0:
            raise ValueError(
                f"corroboration_seconds must be a finite number > 0, got {corroboration}"
            )
        self.cooldown_seconds = cooldown
        self.max_keys = int(max_keys)
        self.min_hits = min_hits
        self.corroboration_seconds = corroboration
        # Detection times of keys still waiting for corroboration (least recent first).
        self._pending: OrderedDict[ThrottleKey, deque[float]] = OrderedDict()
        # Corroborated keys and the time of their latest detection (least recent first).
        self._corroborated: OrderedDict[ThrottleKey, float] = OrderedDict()
        self._uncorroborated_total = 0
        # Ordered by last emission (oldest first), which is what pruning needs.
        self._keys: OrderedDict[ThrottleKey, _KeyState] = OrderedDict()
        # Summaries of keys forgotten while they held detections back, for the next flush().
        self._owed: list[tuple[Detection, int]] = []
        self._suppressed_total = 0
        self._latest = float("-inf")

    @classmethod
    def from_settings(cls, settings: Settings) -> AlertThrottler:
        """The throttler configured by ``settings`` (cooldown and corroboration)."""
        return cls(
            settings.alert_cooldown_seconds,
            min_hits=settings.alert_min_hits,
            corroboration_seconds=settings.alert_corroboration_seconds,
        )

    @property
    def suppressed_total(self) -> int:
        """Threats represented by an alert without being one (duplicates during a
        cooldown, and detections held back until their key was corroborated), not
        counting those that :meth:`flush` later turned into alerts."""
        return self._suppressed_total

    @property
    def uncorroborated_total(self) -> int:
        """Threat detections dropped because their key never reached ``min_hits``
        detections within ``corroboration_seconds``."""
        return self._uncorroborated_total

    @property
    def pending(self) -> int:
        """Threat detections currently held back, waiting for corroboration."""
        return sum(len(hits) for hits in self._pending.values())

    @property
    def tracked_keys(self) -> int:
        """Number of (attack type, destination IP) keys currently remembered."""
        return len(self._keys)

    def check(self, detection: Detection) -> int | None:
        """Decide whether ``detection`` should produce an alert.

        Returns ``None`` to suppress it (or hold it back for corroboration), otherwise
        the number of other detections the alert stands for: those suppressed for its key
        since the last emitted alert, plus those held back until it was corroborated (0
        for a first alert without corroboration). Throttling disabled (cooldown 0) emits
        every corroborated detection. A detection that is not a threat never produces an
        alert: ``None``, without counting as suppressed.
        """
        if not detection.is_threat:
            return None
        timestamp = float(detection.record["timestamp"])
        self._latest = max(self._latest, timestamp)
        key = throttle_key(detection)
        held = self._corroborate(key, timestamp)
        if held is None:
            return None
        if self.cooldown_seconds == 0:
            self._suppressed_total += held
            return held

        state = self._keys.get(key)
        if state is not None and timestamp - state.last_emitted < self.cooldown_seconds:
            state.suppressed += 1 + held
            state.latest_suppressed = detection
            self._suppressed_total += 1 + held
            return None

        suppressed = (state.suppressed if state is not None else 0) + held
        self._suppressed_total += held
        self._keys[key] = _KeyState(timestamp)
        self._keys.move_to_end(key)
        self._prune()
        return suppressed

    def _corroborate(self, key: ThrottleKey, timestamp: float) -> int | None:
        """``None`` to hold this detection of ``key`` back; otherwise how many detections
        held back earlier it now stands for (0 when the key was already corroborated)."""
        if self.min_hits == 1:
            return 0
        last = self._corroborated.get(key)
        if last is not None and timestamp - last <= self.corroboration_seconds:
            self._corroborated[key] = max(last, timestamp)
            self._corroborated.move_to_end(key)
            return 0
        if last is not None:  # the attack went quiet: corroborate it again
            del self._corroborated[key]

        hits = self._pending.get(key)
        if hits is None:
            hits = self._pending[key] = deque()
        else:
            self._pending.move_to_end(key)
        cutoff = timestamp - self.corroboration_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
            self._uncorroborated_total += 1
        hits.append(timestamp)
        if len(hits) < self.min_hits:
            while len(self._pending) > self.max_keys:  # bound memory: drop the stalest
                _, dropped = self._pending.popitem(last=False)
                self._uncorroborated_total += len(dropped)
            return None

        del self._pending[key]
        self._corroborated[key] = timestamp
        self._corroborated.move_to_end(key)
        while len(self._corroborated) > self.max_keys:
            self._corroborated.popitem(last=False)  # it will just be corroborated again
        return len(hits) - 1

    def _expire_corroboration(self, everything: bool) -> None:
        """Drop held-back detections too old to be corroborated (all of them with
        ``everything``) and forget keys whose attack went quiet."""
        cutoff = self._latest - self.corroboration_seconds
        for key in list(self._pending):
            hits = self._pending[key]
            while hits and (everything or hits[0] < cutoff):
                hits.popleft()
                self._uncorroborated_total += 1
            if not hits:
                del self._pending[key]
        corroborated = self._corroborated
        while corroborated and next(iter(corroborated.values())) < cutoff:
            corroborated.popitem(last=False)

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
        if self.min_hits > 1:
            self._expire_corroboration(everything)
        expiry = self._latest - self.cooldown_seconds
        summaries, self._owed = self._owed, []
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
            key, state = next(iter(keys.items()))
            if state.last_emitted > expiry:
                break
            self._forget(key)
        while len(keys) > self.max_keys:
            self._forget(next(iter(keys)))

    def _forget(self, key: ThrottleKey) -> None:
        """Drop ``key``; a summary of what it still holds back is owed to :meth:`flush`."""
        state = self._keys.pop(key)
        if state.latest_suppressed is not None:
            self._owed.append((state.latest_suppressed, state.suppressed - 1))
            self._suppressed_total -= 1  # that one is an alert now


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
