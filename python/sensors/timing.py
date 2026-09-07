"""
python/sensors/timing.py

Measurement delivery scheduling and packet ordering.

PROBLEM:
  Real sensors have two distinct times:
    sample_time_s  — the epoch the measurement physically refers to
    delivery_time_s — when the processed data arrives at the filter

  In a deterministic simulation without real-time constraints, both are
  trivially known. However, for a rigorous evaluation we must:
    1. Never process a packet before its delivery_time_s
    2. Always process packets in deterministic chronological order
    3. Handle ties (same delivery_time_s for different sensors) by sequence_num
    4. Reject packets older than the retained history with a diagnostic

DESIGN:
  MeasurementBus collects all packets, sorts them by (delivery_time_s, sequence_num),
  and yields them in order. The filter processes packets in this sorted order.

  For delayed packets (delivery_time > sample_time), a correct filter must:
    - Store a history of (t, x, P, inputs) snapshots
    - When a delayed packet arrives at delivery_t, restore the snapshot at sample_t
    - Replay all subsequent events forward to delivery_t in the correct order

  Phase 2 implements zero-latency and delayed-delivery scheduling.
  Phase 6 adds the full out-of-sequence (OOS) replay handler.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from ..records import MeasurementPacket


@dataclass(order=True)
class _PrioritizedPacket:
    """Wrapper for priority-queue ordering of packets."""
    delivery_time_s: float
    sequence_num: int          # Secondary sort key for tie-breaking
    sensor_id: str = field(compare=False)
    packet: MeasurementPacket = field(compare=False)


class MeasurementBus:
    """
    Priority queue for time-ordered measurement delivery.

    Accepts packets from any sensor and delivers them in order of
    (delivery_time_s, sequence_num). This is the canonical stream
    consumed by the navigation filter.

    Parameters
    ----------
    max_latency_s : float
        Maximum supported latency for delayed insertion.
        Packets older than (current_time - max_latency_s) are rejected.
        Default 10.0 s.
    """

    def __init__(self, max_latency_s: float = 10.0) -> None:
        self._heap: list[_PrioritizedPacket] = []
        self._max_latency_s = max_latency_s
        self._current_delivery_time: float = -np.inf
        self._rejection_log: list[dict] = []

    def add_packet(self, packet: MeasurementPacket) -> bool:
        """
        Add a packet to the delivery queue.

        Returns True if accepted, False if rejected (too old).
        """
        # Check if packet is older than the retained history
        min_acceptable = self._current_delivery_time - self._max_latency_s
        if packet.delivery_time_s < min_acceptable:
            self._rejection_log.append({
                "sensor_id":      packet.sensor_id,
                "sequence_num":   packet.sequence_num,
                "sample_time_s":  packet.sample_time_s,
                "delivery_time_s": packet.delivery_time_s,
                "reason":         "too_old",
                "current_time_s": self._current_delivery_time,
            })
            return False

        item = _PrioritizedPacket(
            delivery_time_s=packet.delivery_time_s,
            sequence_num=packet.sequence_num,
            sensor_id=packet.sensor_id,
            packet=packet,
        )
        heapq.heappush(self._heap, item)
        return True

    def add_packets(self, packets: list[MeasurementPacket]) -> int:
        """Add a list of packets. Returns number accepted."""
        n_accepted = sum(self.add_packet(p) for p in packets)
        return n_accepted

    def pop_next(self) -> Optional[MeasurementPacket]:
        """
        Pop the next packet in delivery order.

        Returns None if queue is empty.
        """
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        self._current_delivery_time = max(
            self._current_delivery_time, item.delivery_time_s
        )
        return item.packet

    def peek_next_time(self) -> Optional[float]:
        """Return the delivery time of the next packet without popping."""
        if not self._heap:
            return None
        return self._heap[0].delivery_time_s

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def rejection_log(self) -> list[dict]:
        return self._rejection_log

    def drain_up_to(self, t: float) -> list[MeasurementPacket]:
        """
        Return all packets with delivery_time_s <= t, in order.

        This is the standard interface for the filter's main loop:
        at each navigation epoch t, call drain_up_to(t) to get all
        measurements that have arrived.
        """
        result = []
        while self._heap and self._heap[0].delivery_time_s <= t + 1e-9:
            result.append(self.pop_next())
        return result

    def as_sorted_list(self) -> list[MeasurementPacket]:
        """
        Return all packets sorted by (delivery_time_s, sequence_num).
        Non-destructive (copies the heap).
        """
        items = sorted(self._heap)
        return [item.packet for item in items]


def build_measurement_bus(
    *packet_lists: list[MeasurementPacket],
    max_latency_s: float = 10.0,
) -> MeasurementBus:
    """
    Convenience function: create a bus from multiple sensor packet lists.

    Parameters
    ----------
    *packet_lists : multiple list[MeasurementPacket]
        One list per sensor.
    max_latency_s : float

    Returns
    -------
    bus : MeasurementBus sorted by delivery time
    """
    bus = MeasurementBus(max_latency_s=max_latency_s)
    for plist in packet_lists:
        bus.add_packets(plist)
    return bus
