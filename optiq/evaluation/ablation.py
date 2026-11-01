"""Policy ablation study harness for OptiQ:
Compares:
1. OptiQ-Baseline: Unbounded aging with Q=8s, starvation=60s
2. OptiQ-PredictionOnly: SRTF using predicted burst, without aging/starvation override
3. OptiQ-AgingOnly: Highest aged priority first, without burst-time prediction
4. OptiQ-Hybrid (Frozen Policy): Adaptive ML burst prediction with bounded aging, starvation threshold=1800s, Q=30s
"""

from typing import Dict, List, Tuple
import pandas as pd

from optiq.simulator.process import Process
from optiq.schedulers import OptiQScheduler, MLSJFScheduler, PriorityScheduler
from optiq.simulator.metrics import SchedulerBenchmarkSummary


def run_ablation_study(
    processes: List[Process],
    quantum: float = 30.0,
    aging_interval: float = 60.0,
    starvation_threshold: float = 300.0,
) -> pd.DataFrame:
    """Execute ablation policies on identical process instances and return comparison DataFrame."""
    # 1. Baseline OptiQ (Unbounded aging, small starvation threshold, small quantum)
    p_baseline = OptiQScheduler(
        quantum=8.0,
        aging_interval=15.0,
        starvation_threshold=60.0,
        max_priority_ceiling=999999,
        preemption_threshold_ratio=1.0,
    )
    p_baseline.name = "OptiQ-Baseline"

    # 2. Prediction-Only (ML burst prediction without aging or starvation override)
    p_pred_only = OptiQScheduler(
        quantum=quantum,
        aging_interval=0.0,
        starvation_threshold=float("inf"),
        max_priority_ceiling=10,
        preemption_threshold_ratio=0.5,
    )
    p_pred_only.name = "OptiQ-PredictionOnly"

    # 3. Aging-Only (Static priority with aging, without burst prediction)
    p_aging_only = OptiQScheduler(
        quantum=quantum,
        aging_interval=aging_interval,
        starvation_threshold=0.0,  # Every job is considered starved immediately, sorting purely by aged priority
        max_priority_ceiling=25,
        preemption_threshold_ratio=0.0,
    )
    p_aging_only.name = "OptiQ-AgingOnly"

    # 4. Final Frozen OptiQ Policy (matches current experiment settings)
    p_final = OptiQScheduler(
        quantum=quantum,
        aging_interval=aging_interval,
        starvation_threshold=starvation_threshold,
        max_priority_ceiling=25,
        preemption_threshold_ratio=0.5,
    )
    p_final.name = "OptiQ-Hybrid (Final)"

    ablation_records = []
    for scheduler in [p_baseline, p_pred_only, p_aging_only, p_final]:
        summary, _ = scheduler.run(processes, starvation_threshold=starvation_threshold)
        ablation_records.append({
            "Policy": scheduler.name,
            "Avg Waiting Time (s)": summary.avg_waiting_time,
            "Median Waiting Time (s)": summary.median_waiting_time,
            "Avg Turnaround Time (s)": summary.avg_turnaround_time,
            "Median Turnaround Time (s)": summary.median_turnaround_time,
            "Context Switches": summary.total_context_switches,
            "Starvation Risk Count": summary.starvation_risk_count,
            "Fairness Index": summary.jains_fairness_index,
        })

    return pd.DataFrame(ablation_records)
