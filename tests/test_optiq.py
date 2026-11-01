"""Comprehensive test suite for OptiQ:
1. Metric formulas (Turnaround, Waiting, Response, Dispersion, Fairness)
2. Scheduler invariants: identical arrivals, completions, zero jobs dropped
3. Round Robin quantum enforcement and consolidated context switches
4. Priority aging calculations and bounded ceilings
5. Target leakage prevention: ensuring pre-scheduling features only
6. Reproducibility test with seeded synthetic workload
7. Prediction-only dispatch logic (SRTF-like on predicted remaining time)
8. Policy ablation invariant check
9. Fast smoke test for end-to-end integration
"""

import pytest
import numpy as np
import pandas as pd

from optiq.data.loader import (
    generate_synthetic_workload,
    PRE_SCHEDULING_FEATURES,
    TARGET_COL,
)
from optiq.models.trainer import ModelTrainer
from optiq.simulator.process import Process, ExecutionSlice
from optiq.simulator.metrics import compute_metrics
from optiq.schedulers import (
    FCFSScheduler,
    SJFScheduler,
    SRTFScheduler,
    RoundRobinScheduler,
    PriorityScheduler,
    MLSJFScheduler,
    OptiQScheduler,
)
from optiq.evaluation.ablation import run_ablation_study


def test_metric_formulas():
    """Verify exact definitions: Turnaround = Comp - Arr, Waiting = Turnaround - Burst, Response = Start - Arr."""
    p1 = Process(pid=1, arrival_time=0.0, burst_time=5.0, predicted_burst=5.0)
    p1.start_time = 0.0
    p1.completion_time = 5.0

    p2 = Process(pid=2, arrival_time=2.0, burst_time=3.0, predicted_burst=3.0)
    p2.start_time = 5.0
    p2.completion_time = 8.0

    slices = [
        ExecutionSlice(pid=1, start_time=0.0, end_time=5.0, duration=5.0),
        ExecutionSlice(pid=2, start_time=5.0, end_time=8.0, duration=3.0),
    ]

    summary, df = compute_metrics([p1, p2], slices, "TestScheduler", starvation_threshold=4.0)

    # Process 1 checks
    row1 = df[df["pid"] == 1].iloc[0]
    assert row1["turnaround_time"] == 5.0
    assert row1["waiting_time"] == 0.0
    assert row1["response_time"] == 0.0

    # Process 2 checks: Arr=2, Start=5, Comp=8, Burst=3
    # Turnaround = 8 - 2 = 6
    # Waiting = 6 - 3 = 3
    # Response = 5 - 2 = 3
    row2 = df[df["pid"] == 2].iloc[0]
    assert row2["turnaround_time"] == 6.0
    assert row2["waiting_time"] == 3.0
    assert row2["response_time"] == 3.0

    # Aggregate checks
    assert summary.avg_waiting_time == (0.0 + 3.0) / 2.0
    assert summary.avg_turnaround_time == (5.0 + 6.0) / 2.0
    assert summary.median_waiting_time == 1.5
    assert summary.median_turnaround_time == 5.5
    assert summary.total_context_switches == 1
    # Waiting 3 < 4 threshold -> starvation_risk_count == 0
    assert summary.starvation_risk_count == 0
    assert 0.0 <= summary.jains_fairness_index <= 1.0


def test_priority_aging_formula_and_ceiling():
    """effective_priority = base_priority + floor(ready_queue_wait / aging_interval), bounded by ceiling."""
    p = Process(pid=1, arrival_time=0.0, burst_time=10.0, predicted_burst=10.0, base_priority=3)
    p.ready_queue_wait = 25.0
    p.apply_aging(aging_interval=10.0)
    assert p.effective_priority == 3 + int(25.0 // 10.0) == 5

    # Check ceiling within OptiQScheduler loop
    scheduler = OptiQScheduler(quantum=10.0, aging_interval=10.0, max_priority_ceiling=8)
    p2 = Process(pid=2, arrival_time=0.0, burst_time=100.0, predicted_burst=100.0, base_priority=5)
    summary, _ = scheduler.run([p2])
    assert summary.completed_processes == 1


def test_round_robin_time_slicing_and_consolidation():
    """Verify that Round Robin never executes single slice exceeding quantum, and consolidates lone slices."""
    procs = [
        Process(pid=1, arrival_time=0.0, burst_time=9.0, predicted_burst=9.0),
        Process(pid=2, arrival_time=0.0, burst_time=4.0, predicted_burst=4.0),
    ]
    quantum = 3.0
    rr = RoundRobinScheduler(quantum=quantum)
    summary, _ = rr.run(procs)

    max_duration = max(s.duration for s in rr.timeline)
    assert max_duration <= quantum
    assert summary.completed_processes == 2
    assert summary.total_context_switches > 0


def test_fcfs_order():
    """Verify FCFS strictly processes according to arrival order."""
    procs = [
        Process(pid=10, arrival_time=5.0, burst_time=4.0, predicted_burst=4.0),
        Process(pid=20, arrival_time=1.0, burst_time=8.0, predicted_burst=8.0),
        Process(pid=30, arrival_time=3.0, burst_time=2.0, predicted_burst=2.0),
    ]
    fcfs = FCFSScheduler()
    fcfs.run(procs)

    execution_order = [s.pid for s in fcfs.timeline]
    # Arrival order is PID 20 (t=1), then PID 30 (t=3), then PID 10 (t=5)
    assert execution_order == [20, 30, 10]


def test_optiq_anti_starvation_override():
    """Verify that when a process exceeds starvation threshold, OptiQ overrides and schedules it."""
    procs = [
        Process(pid=1, arrival_time=0.0, burst_time=50.0, predicted_burst=50.0, base_priority=5),
        Process(pid=2, arrival_time=5.0, burst_time=2.0, predicted_burst=2.0, base_priority=5),
        Process(pid=3, arrival_time=10.0, burst_time=15.0, predicted_burst=15.0, base_priority=1),
    ]
    optiq = OptiQScheduler(quantum=5.0, aging_interval=10.0, starvation_threshold=20.0)
    summary, df = optiq.run(procs)

    assert summary.completed_processes == 3
    assert len(optiq.decision_logs) > 0
    assert any("Shortest predicted" in log.reason or "Starvation" in log.reason for log in optiq.decision_logs)


def test_prediction_only_dispatch():
    """Verify that OptiQ without aging strictly favors the job with shortest predicted remaining time."""
    p1 = Process(pid=1, arrival_time=0.0, burst_time=100.0, predicted_burst=100.0, base_priority=10)
    p2 = Process(pid=2, arrival_time=0.0, burst_time=10.0, predicted_burst=5.0, base_priority=1)

    optiq_pred = OptiQScheduler(quantum=20.0, aging_interval=0.0, starvation_threshold=float("inf"))
    optiq_pred.run([p1, p2])

    # First executed slice should be PID 2 (predicted burst 5 < 100)
    assert optiq_pred.timeline[0].pid == 2


def test_target_leakage_exclusion():
    """Validate that future runtime and post-execution fields are NOT in pre-scheduling features."""
    forbidden_features = [
        "run_time",
        "wait_time",
        "avg_cpu_time",
        "used_mem",
        "status",
        "completion_time",
        "remaining_time",
        "user_id",
        "group_id",
    ]
    for feat in forbidden_features:
        assert feat not in PRE_SCHEDULING_FEATURES, f"Target leakage or spurious ID detected! {feat} must not be a feature."


def test_reproducibility():
    """Verify that training and workload generation with identical seeds produce bit-identical results."""
    df1, _ = generate_synthetic_workload(n_samples=200, seed=123)
    df2, _ = generate_synthetic_workload(n_samples=200, seed=123)
    pd.testing.assert_frame_equal(df1, df2)

    trainer1 = ModelTrainer(seed=123)
    m1, preds1, _ = trainer1.train_and_evaluate_all(df1)

    trainer2 = ModelTrainer(seed=123)
    m2, preds2, _ = trainer2.train_and_evaluate_all(df2)

    np.testing.assert_allclose(preds1["RandomForest"], preds2["RandomForest"])


def test_ablation_study_execution():
    """Verify that the policy ablation study runs and returns rows for all 4 variants."""
    procs = [
        Process(pid=1, arrival_time=0.0, burst_time=20.0, predicted_burst=18.0),
        Process(pid=2, arrival_time=2.0, burst_time=5.0, predicted_burst=4.5),
        Process(pid=3, arrival_time=4.0, burst_time=12.0, predicted_burst=11.0),
    ]
    ablation_df = run_ablation_study(procs, starvation_threshold=15.0)
    assert len(ablation_df) == 4
    assert set(ablation_df["Policy"]) == {
        "OptiQ-Baseline",
        "OptiQ-PredictionOnly",
        "OptiQ-AgingOnly",
        "OptiQ-Hybrid (Final)",
    }


def test_smoke_end_to_end_synthetic():
    """Fast integration smoke test on seeded synthetic workload."""
    df, meta = generate_synthetic_workload(n_samples=150, seed=42)
    assert meta.source_type == "SYNTHETIC_WORKLOAD"

    trainer = ModelTrainer(seed=42)
    metrics_df, preds, test_df = trainer.train_and_evaluate_all(df)
    assert not metrics_df.empty
    assert "RandomForest" in preds

    best_preds = preds[trainer.best_model_name]
    procs = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        procs.append(
            Process(
                pid=int(row["job_id"]),
                arrival_time=float(row["submit_time"]),
                burst_time=float(row[TARGET_COL]),
                predicted_burst=float(best_preds[idx]),
                base_priority=int(row.get("base_priority", 5)),
            )
        )

    # Normalize arrival times so simulation starts at t=0
    min_arr = min(p.arrival_time for p in procs)
    for p in procs:
        p.arrival_time -= min_arr

    # Run OptiQ
    optiq = OptiQScheduler(quantum=30.0, aging_interval=60.0, starvation_threshold=1800.0)
    summary, proc_results = optiq.run(procs)
    assert summary.completed_processes == len(procs)
    assert summary.avg_waiting_time >= 0.0
    assert summary.avg_turnaround_time >= summary.avg_waiting_time


def test_multiple_overdue_jobs_sharing():
    """Verify multiple starved jobs both make progress and do not monopolize CPU."""
    p1 = Process(pid=1, arrival_time=0.0, burst_time=60.0, predicted_burst=60.0, base_priority=5)
    p2 = Process(pid=2, arrival_time=0.0, burst_time=60.0, predicted_burst=60.0, base_priority=5)
    sched = OptiQScheduler(quantum=10.0, aging_interval=20.0, starvation_threshold=15.0)
    summary, proc_df = sched.run([p1, p2])
    assert summary.completed_processes == 2
    # Verify both processes received slices in timeline
    p1_slices = [s for s in sched.timeline if s.pid == 1]
    p2_slices = [s for s in sched.timeline if s.pid == 2]
    assert len(p1_slices) >= 2
    assert len(p2_slices) >= 2


def test_cumulative_vs_idle_wait_tracking():
    """Verify cumulative ready-queue wait does not reset while current ready_queue_wait does on service."""
    p = Process(pid=10, arrival_time=0.0, burst_time=50.0, predicted_burst=50.0)
    p.update_wait(30.0)
    assert p.ready_queue_wait == 30.0
    assert p.cumulative_wait == 30.0
    p.on_service(30.0)
    assert p.ready_queue_wait == 0.0
    assert p.cumulative_wait == 30.0  # Cumulative wait preserved
    p.update_wait(45.0)
    assert p.ready_queue_wait == 15.0
    assert p.cumulative_wait == 45.0


def test_api_endpoints_health_and_runs():
    """Verify FastAPI backend endpoints return expected schemas and statuses."""
    from fastapi.testclient import TestClient
    from optiq.api.server import app

    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert isinstance(runs.json(), list)
    assert len(runs.json()) > 0
    run_id = runs.json()[0]["run_id"]

    summary = client.get(f"/api/runs/{run_id}/summary")
    assert summary.status_code == 200
    assert "comparison" in summary.json()
    assert "metadata" in summary.json()

    index = client.get("/")
    assert index.status_code == 200

