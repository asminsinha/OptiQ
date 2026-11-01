"""Base Scheduler class and simulation engine for all CPU schedulers."""

from abc import ABC, abstractmethod
import copy
from typing import Dict, List, Optional, Tuple

import pandas as pd

from optiq.simulator.process import ExecutionSlice, Process, ScheduleDecisionLog
from optiq.simulator.metrics import SchedulerBenchmarkSummary, compute_metrics


class BaseScheduler(ABC):
    """Abstract base class for all CPU scheduling algorithms."""

    def __init__(self, name: str, is_oracle: bool = False, notes: str = ""):
        self.name = name
        self.is_oracle = is_oracle
        self.notes = notes
        self.timeline: List[ExecutionSlice] = []
        self.decision_logs: List[ScheduleDecisionLog] = []

    @abstractmethod
    def run(
        self,
        processes: List[Process],
        starvation_threshold: float = 100.0,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        """Run the simulation on a cloned copy of processes.

        Must populate self.timeline and self.decision_logs, and return
        (benchmark_summary, process_metrics_df).
        """
        pass

    def _clone_processes(self, processes: List[Process]) -> List[Process]:
        """Deep copy processes to guarantee identical initial state across schedulers."""
        return [
            Process(
                pid=p.pid,
                arrival_time=p.arrival_time,
                burst_time=p.burst_time,
                predicted_burst=p.predicted_burst,
                base_priority=p.base_priority,
            )
            for p in processes
        ]
