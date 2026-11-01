"""Standard evaluation metrics for CPU scheduling simulations.

Definitions:
- Turnaround Time = Completion Time - Arrival Time
- Waiting Time = Turnaround Time - Burst Time
- Response Time = First CPU Start Time - Arrival Time
- Makespan = Max(Completion Time) - Min(Arrival Time)
- Throughput = Completed Processes / Makespan
- Context Switches: Count of transitions between different non-idle PIDs
- Starvation-Risk Count: Count of processes whose waiting time >= starvation_threshold
- Waiting-Time Dispersion: Standard deviation of process waiting times
- Jain's Fairness Index: (Sum(Wait)^2) / (N * Sum(Wait^2)) or on turnaround/burst ratio
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from optiq.simulator.process import Process, ExecutionSlice


@dataclass
class ProcessMetricResult:
    """Individual process outcome metrics."""
    pid: int
    arrival_time: float
    burst_time: float
    predicted_burst: float
    base_priority: int
    start_time: float
    completion_time: float
    turnaround_time: float
    waiting_time: float
    response_time: float


@dataclass
class SchedulerBenchmarkSummary:
    """Aggregate benchmark results for a scheduler run."""
    scheduler_name: str
    total_processes: int
    completed_processes: int
    makespan: float
    throughput: float
    avg_turnaround_time: float
    median_turnaround_time: float
    avg_waiting_time: float
    median_waiting_time: float
    avg_response_time: float
    median_response_time: float
    max_waiting_time: float
    wait_time_dispersion: float
    jains_fairness_index: float
    total_context_switches: int
    starvation_risk_count: int
    starvation_threshold: float
    cpu_utilization_pct: float
    is_oracle: bool = False
    notes: str = ""


def compute_metrics(
    processes: List[Process],
    timeline: List[ExecutionSlice],
    scheduler_name: str,
    starvation_threshold: float = 100.0,
    is_oracle: bool = False,
    notes: str = "",
) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
    """Compute per-process and aggregate simulation metrics from actual trace.

    Calculates:
    - per-process turnaround, waiting, response
    - aggregate averages, medians, dispersion, context switches, starvation count, fairness
    """
    if not processes:
        raise ValueError("Cannot compute metrics on empty process list.")

    process_results: List[ProcessMetricResult] = []
    total_burst_time = 0.0

    for p in processes:
        if p.completion_time is None or p.start_time is None:
            raise RuntimeError(f"Process {p.pid} did not complete execution.")

        turnaround = p.completion_time - p.arrival_time
        waiting = turnaround - p.burst_time
        response = p.start_time - p.arrival_time
        total_burst_time += p.burst_time

        # Numerical safety for zero / floating roundoff
        waiting = max(0.0, round(waiting, 4))
        turnaround = max(p.burst_time, round(turnaround, 4))
        response = max(0.0, round(response, 4))

        process_results.append(
            ProcessMetricResult(
                pid=p.pid,
                arrival_time=round(p.arrival_time, 4),
                burst_time=round(p.burst_time, 4),
                predicted_burst=round(p.predicted_burst, 4),
                base_priority=p.base_priority,
                start_time=round(p.start_time, 4),
                completion_time=round(p.completion_time, 4),
                turnaround_time=turnaround,
                waiting_time=waiting,
                response_time=response,
            )
        )

    # Process metrics dataframe
    proc_df = pd.DataFrame([vars(r) for r in process_results])

    # Context switch count: distinct PID transitions (excluding idle)
    # Consecutive execution slices by the same PID are consolidated as 0 context switch
    context_switches = 0
    prev_pid: Optional[int] = None
    for slice_item in timeline:
        if prev_pid is not None and slice_item.pid != prev_pid:
            context_switches += 1
        prev_pid = slice_item.pid

    # Aggregate metrics
    min_arrival = min(p.arrival_time for p in processes)
    max_completion = max(p.completion_time for p in processes)
    makespan = max(0.001, max_completion - min_arrival)
    completed_n = len(processes)

    throughput = completed_n / makespan
    avg_turnaround = float(proc_df["turnaround_time"].mean())
    median_turnaround = float(proc_df["turnaround_time"].median())
    avg_waiting = float(proc_df["waiting_time"].mean())
    median_waiting = float(proc_df["waiting_time"].median())
    avg_response = float(proc_df["response_time"].mean())
    median_response = float(proc_df["response_time"].median())
    max_wait = float(proc_df["waiting_time"].max())
    wait_std = float(proc_df["waiting_time"].std()) if completed_n > 1 else 0.0
    starvation_risk = int((proc_df["waiting_time"] >= starvation_threshold).sum())
    cpu_util = min(100.0, round((total_burst_time / makespan) * 100.0, 2))
    
    # Jain's Fairness Index on slowdown/normalized turnaround: T_i / B_i
    # Slowdown S_i = turnaround / burst (>= 1.0)
    slowdowns = proc_df["turnaround_time"] / np.maximum(proc_df["burst_time"], 0.1)
    sum_s = float(slowdowns.sum())
    sum_s_sq = float((slowdowns ** 2).sum())
    if sum_s_sq > 0 and completed_n > 0:
        jains_fairness = round((sum_s ** 2) / (completed_n * sum_s_sq), 4)
    else:
        jains_fairness = 1.0

    summary = SchedulerBenchmarkSummary(
        scheduler_name=scheduler_name,
        total_processes=completed_n,
        completed_processes=completed_n,
        makespan=round(makespan, 2),
        throughput=round(throughput, 4),
        avg_turnaround_time=round(avg_turnaround, 3),
        median_turnaround_time=round(median_turnaround, 3),
        avg_waiting_time=round(avg_waiting, 3),
        median_waiting_time=round(median_waiting, 3),
        avg_response_time=round(avg_response, 3),
        median_response_time=round(median_response, 3),
        max_waiting_time=round(max_wait, 3),
        wait_time_dispersion=round(wait_std, 3),
        jains_fairness_index=jains_fairness,
        total_context_switches=context_switches,
        starvation_risk_count=starvation_risk,
        starvation_threshold=starvation_threshold,
        cpu_utilization_pct=cpu_util,
        is_oracle=is_oracle,
        notes=notes,
    )

    return summary, proc_df
