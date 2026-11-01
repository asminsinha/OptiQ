"""Core simulation data structures for CPU scheduling."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Process:
    """Represents a discrete schedulable process / job."""
    pid: int
    arrival_time: float
    burst_time: float                 # Actual service/burst time (ground truth)
    predicted_burst: float            # Predicted burst time from ML or estimate
    base_priority: int = 5            # Higher number = higher priority convention (1-10)

    # Dynamic execution tracking
    remaining_time: float = field(init=False)
    predicted_remaining_time: float = field(init=False)
    effective_priority: int = field(init=False)
    start_time: Optional[float] = None
    completion_time: Optional[float] = None

    # Cumulative waiting & execution time
    time_executed: float = 0.0
    ready_queue_wait: float = 0.0     # Time spent waiting since most recent enqueue / CPU service (for starvation & aging)
    cumulative_wait: float = 0.0      # Total cumulative time spent in ready queue across all preemption cycles
    last_queued_time: float = 0.0     # Simulation clock time when entered ready queue

    def __post_init__(self):
        self.remaining_time = round(float(self.burst_time), 4)
        self.predicted_remaining_time = round(float(self.predicted_burst), 4)
        self.effective_priority = int(self.base_priority)
        self.last_queued_time = float(self.arrival_time)
        self.cumulative_wait = 0.0

    def update_wait(self, current_time: float) -> None:
        """Accumulate waiting time while idle in ready queue."""
        if current_time > self.last_queued_time:
            delta = current_time - self.last_queued_time
            self.ready_queue_wait += delta
            self.cumulative_wait += delta
            self.last_queued_time = current_time

    def on_service(self, current_time: float) -> None:
        """Called when process receives CPU service to reset its current idle wait time."""
        self.ready_queue_wait = 0.0
        self.last_queued_time = current_time
        self.effective_priority = self.base_priority

    def apply_aging(self, aging_interval: float) -> None:
        """Update effective priority according to configured aging rule.

        Rule: effective_priority = base_priority + floor(ready_queue_wait / aging_interval)
        Higher integer = higher priority.
        """
        if aging_interval > 0:
            aging_boost = int(self.ready_queue_wait // aging_interval)
            self.effective_priority = self.base_priority + aging_boost
        else:
            self.effective_priority = self.base_priority


@dataclass
class ExecutionSlice:
    """Represents an atomic scheduling interval on the CPU."""
    pid: int
    start_time: float
    end_time: float
    duration: float
    reason: str = ""
    effective_priority: int = 5
    predicted_remaining: float = 0.0


@dataclass
class ScheduleDecisionLog:
    """Detailed trace record for each scheduler decision event."""
    timestamp: float
    scheduler_name: str
    ready_queue_pids: List[int]
    chosen_pid: Optional[int]
    reason: str
    effective_priorities: Dict[int, int]
    predicted_remaining_times: Dict[int, float]
    actual_remaining_times: Dict[int, float]
