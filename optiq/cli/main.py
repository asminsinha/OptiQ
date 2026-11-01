"""Full end-to-end experiment orchestration and CLI wizard for OptiQ."""

import argparse
from datetime import datetime
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

from optiq.data.loader import (
    parse_swf_trace,
    generate_synthetic_workload,
    DatasetMetadata,
    TARGET_COL,
)
from optiq.models.trainer import ModelTrainer
from optiq.simulator.process import Process
from optiq.simulator.metrics import SchedulerBenchmarkSummary
from optiq.schedulers import (
    FCFSScheduler,
    SJFScheduler,
    MLSJFScheduler,
    SRTFScheduler,
    RoundRobinScheduler,
    PriorityScheduler,
    OptiQScheduler,
)
from optiq.evaluation.plots import (
    plot_ml_predictions_and_residuals,
    plot_ml_model_comparison,
    plot_feature_importance,
    plot_scheduler_comparison,
    plot_gantt_chart,
)


def run_full_experiment(
    data_source: str = "real",
    trace_path: str = "data/raw/LANL-CM5-1994-4.1-cln.swf.gz",
    n_jobs: int = 500,
    synthetic_profile: str = "mixed",
    quantum: float = 10.0,
    aging_interval: float = 20.0,
    starvation_threshold: float = 100.0,
    seed: int = 42,
    output_base_dir: str = "outputs",
) -> str:
    """Execute complete end-to-end ML training, scheduler benchmarking, and report generation."""
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_base_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    models_dir = os.path.join("saved_models", run_id)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" OPTIQ EXPERIMENT PIPELINE [Run ID: {run_id}]")
    print("=" * 70)

    # 1. DATA LOADING
    print(f"\n[1/5] Loading and Validating Dataset (Source: {data_source.upper()})...")
    if data_source == "real":
        if not os.path.exists(trace_path):
            print(f"ERROR: Trace file not found at {trace_path}.")
            print("Falling back to synthetic workload for reliable demonstration.")
            df, meta = generate_synthetic_workload(n_samples=n_jobs, seed=seed, profile=synthetic_profile)
        else:
            df, meta = parse_swf_trace(trace_path, max_records=max(n_jobs * 3, 2000))
            if len(df) > n_jobs:
                df = df.iloc[:n_jobs].copy()
    else:
        df, meta = generate_synthetic_workload(n_samples=n_jobs, seed=seed, profile=synthetic_profile)

    print(f" -> Source: {meta.source_type} ({meta.filepath_or_name})")
    print(f" -> Loaded Records: {len(df)} | Excluded/Filtered: {meta.excluded_records}")
    print(f" -> Target Service Time: Median={meta.target_median:.2f}s, Mean={meta.target_mean:.2f}s, Max={meta.target_max:.2f}s")
    print(f" -> Features: {', '.join(meta.features)}")

    # 2. MODEL TRAINING & EVALUATION
    print("\n[2/5] Training Regression Models on Pre-Scheduling Features...")
    trainer = ModelTrainer(seed=seed)
    metrics_df, predictions_dict, test_df = trainer.train_and_evaluate_all(df, chronological=True)

    print("\n" + metrics_df.to_string(index=False))
    best_model_name = trainer.best_model_name
    print(f"\n -> Selected Best Model (Lowest Validation MAE): {best_model_name}")

    # Save Models and Scaler
    trainer.save_artifacts(models_dir)
    print(f" -> Model artifacts saved to: {models_dir}")

    # Save ML metrics and test predictions
    metrics_csv = os.path.join(out_dir, "model_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    preds_df = pd.DataFrame({"actual": test_df[TARGET_COL].values})
    for m_name, preds in predictions_dict.items():
        preds_df[f"pred_{m_name}"] = preds
    preds_csv = os.path.join(out_dir, "predictions.csv")
    preds_df.to_csv(preds_csv, index=False)

    # 3. GENERATE ML VISUALIZATIONS
    print("\n[3/5] Generating ML Diagnostic Plots...")
    best_preds = predictions_dict[best_model_name]
    y_test_actual = test_df[TARGET_COL].values

    scatter_png = os.path.join(out_dir, "actual_vs_predicted.png")
    plot_ml_predictions_and_residuals(y_test_actual, best_preds, best_model_name, scatter_png)

    models_comp_png = os.path.join(out_dir, "ml_model_comparison.png")
    plot_ml_model_comparison(metrics_df, models_comp_png)

    fi_df = trainer.get_feature_importances(best_model_name)
    if fi_df is not None:
        fi_png = os.path.join(out_dir, "feature_importance.png")
        plot_feature_importance(fi_df, best_model_name, fi_png)

    # 4. CPU SCHEDULING SIMULATION
    print("\n[4/5] Executing Comparative CPU Scheduling Simulations...")
    test_procs: List[Process] = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        test_procs.append(
            Process(
                pid=int(row["job_id"]),
                arrival_time=float(row["submit_time"]),
                burst_time=float(row[TARGET_COL]),
                predicted_burst=float(best_preds[idx]),
                base_priority=int(row.get("base_priority", 5)),
            )
        )

    # Normalize simulation start time to t=0
    min_arr = min(p.arrival_time for p in test_procs)
    for p in test_procs:
        p.arrival_time = round(p.arrival_time - min_arr, 4)

    schedulers = [
        FCFSScheduler(),
        SJFScheduler(),
        MLSJFScheduler(),
        SRTFScheduler(),
        RoundRobinScheduler(quantum=quantum),
        PriorityScheduler(),
        OptiQScheduler(
            quantum=quantum,
            aging_interval=aging_interval,
            starvation_threshold=starvation_threshold,
        ),
    ]

    summary_records = []
    optiq_timeline = []
    optiq_decision_logs = []

    for s in schedulers:
        summary, proc_results_df = s.run(test_procs, starvation_threshold=starvation_threshold)
        summary_records.append({
            "Scheduler": summary.scheduler_name,
            "Type": "Oracle Baseline" if summary.is_oracle else "Standard/Hybrid",
            "Avg Waiting Time (s)": summary.avg_waiting_time,
            "Median Waiting Time (s)": summary.median_waiting_time,
            "Avg Turnaround Time (s)": summary.avg_turnaround_time,
            "Median Turnaround Time (s)": summary.median_turnaround_time,
            "Avg Response Time (s)": summary.avg_response_time,
            "Context Switches": summary.total_context_switches,
            "Starvation Risk Count": summary.starvation_risk_count,
            "Fairness Index": summary.jains_fairness_index,
            "Makespan (s)": summary.makespan,
            "Throughput (jobs/s)": summary.throughput,
        })

        # Save per-process results
        proc_csv = os.path.join(out_dir, f"process_results_{s.name.split()[0].lower()}.csv")
        proc_results_df.to_csv(proc_csv, index=False)

        if isinstance(s, OptiQScheduler):
            optiq_timeline = s.timeline
            optiq_decision_logs = s.decision_logs

    comparison_df = pd.DataFrame(summary_records)
    comp_csv = os.path.join(out_dir, "algorithm_comparison.csv")
    comparison_df.to_csv(comp_csv, index=False)

    print("\n" + comparison_df.to_string(index=False))

    # Run policy ablation study
    print("\n[4b/5] Running Policy Ablation Study...")
    from optiq.evaluation.ablation import run_ablation_study
    ablation_df = run_ablation_study(
        test_procs,
        quantum=quantum,
        aging_interval=aging_interval,
        starvation_threshold=starvation_threshold,
    )
    ablation_csv = os.path.join(out_dir, "ablation_study.csv")
    ablation_df.to_csv(ablation_csv, index=False)
    print("\n" + ablation_df.to_string(index=False))

    # Save scheduling trace / decision logs for OptiQ
    trace_records = [
        {
            "timestamp": log.timestamp,
            "chosen_pid": log.chosen_pid,
            "reason": log.reason,
            "ready_count": len(log.ready_queue_pids),
        }
        for log in optiq_decision_logs
    ]
    trace_df = pd.DataFrame(trace_records)
    trace_csv = os.path.join(out_dir, "optiq_scheduling_trace.csv")
    trace_df.to_csv(trace_csv, index=False)

    # 5. SCHEDULER CHARTS & EXPERIMENT METADATA
    print("\n[5/5] Generating Scheduling Evaluation Charts & Metadata...")
    sched_comp_png = os.path.join(out_dir, "scheduler_comparison.png")
    plot_scheduler_comparison(comparison_df, sched_comp_png)

    from optiq.evaluation.plots import plot_waiting_time_distribution, plot_ablation_comparison
    # Waiting time distribution plot
    per_proc_map = {}
    for s in schedulers:
        p_path = os.path.join(out_dir, f"process_results_{s.name.split()[0].lower()}.csv")
        if os.path.exists(p_path):
            per_proc_map[s.name] = pd.read_csv(p_path)
    wait_dist_png = os.path.join(out_dir, "waiting_time_distribution.png")
    plot_waiting_time_distribution(per_proc_map, wait_dist_png)

    # Ablation comparison plot
    ablation_png = os.path.join(out_dir, "ablation_comparison.png")
    plot_ablation_comparison(ablation_df, ablation_png)

    gantt_png = os.path.join(out_dir, "optiq_gantt_chart.png")
    plot_gantt_chart(optiq_timeline, "OptiQ Simulation Schedule", gantt_png)

    metadata_path = os.path.join(out_dir, "experiment_metadata.json")
    meta_dict = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": vars(meta),
        "parameters": {
            "n_jobs": n_jobs,
            "quantum": quantum,
            "aging_interval": aging_interval,
            "starvation_threshold": starvation_threshold,
            "seed": seed,
            "best_ml_model": best_model_name,
        },
        "output_files": [
            "model_metrics.csv",
            "predictions.csv",
            "algorithm_comparison.csv",
            "ablation_study.csv",
            "optiq_scheduling_trace.csv",
            "actual_vs_predicted.png",
            "ml_model_comparison.png",
            "scheduler_comparison.png",
            "waiting_time_distribution.png",
            "ablation_comparison.png",
            "optiq_gantt_chart.png",
        ],
    }
    with open(metadata_path, "w") as f:
        json.dump(meta_dict, f, indent=2)

    print("\n" + "=" * 70)
    print(" EXPERIMENT COMPLETE! OUTPUT SUMMARY:")
    print(f" -> Artifacts directory: {os.path.abspath(out_dir)}")
    print(f" -> Algorithm comparison: {comp_csv}")
    print(f" -> Model metrics: {metrics_csv}")
    print(f" -> Gantt chart: {gantt_png}")
    print("=" * 70)

    return out_dir


def cli_wizard():
    """Interactive command-line wizard for demonstration."""
    print("*" * 70)
    print("  OptiQ: Adaptive ML CPU Scheduling Framework (Review 2 Terminal Demo)")
    print("*" * 70)

    while True:
        print("\nMain Menu:")
        print("  [1] Run Full End-to-End Experiment (ML + Schedulers + Plots)")
        print("  [2] Train and Compare ML Burst Prediction Models Only")
        print("  [3] Run Comparative CPU Scheduling Simulation Only")
        print("  [4] Inspect Saved Results and Outputs")
        print("  [5] Exit")

        choice = input("\nEnter choice [1-5] (default 1): ").strip() or "1"

        if choice == "1":
            print("\nConfigure Experiment Parameters (Press Enter to use defaults):")
            ds = input("Data Source ['real' trace / 'synthetic'] (default: real): ").strip().lower() or "real"
            n_val = input("Number of processes / jobs (default: 500): ").strip() or "500"
            q_val = input("Round Robin / OptiQ Quantum (s) (default: 10.0): ").strip() or "10.0"
            a_val = input("Priority Aging Interval (s) (default: 20.0): ").strip() or "20.0"
            s_val = input("Starvation Threshold (s) (default: 100.0): ").strip() or "100.0"

            run_full_experiment(
                data_source=ds,
                n_jobs=int(n_val),
                quantum=float(q_val),
                aging_interval=float(a_val),
                starvation_threshold=float(s_val),
            )

        elif choice == "2":
            ds = input("Data Source ['real' / 'synthetic'] (default: real): ").strip().lower() or "real"
            n_val = int(input("Number of samples (default: 500): ").strip() or "500")
            if ds == "real":
                df, _ = parse_swf_trace("data/raw/LANL-CM5-1994-4.1-cln.swf.gz", max_records=n_val * 2)
                df = df.iloc[:n_val]
            else:
                df, _ = generate_synthetic_workload(n_samples=n_val)

            trainer = ModelTrainer()
            metrics_df, _, _ = trainer.train_and_evaluate_all(df)
            print("\n" + metrics_df.to_string(index=False))

        elif choice == "3":
            print("Running scheduling comparison with synthetic sample...")
            df, _ = generate_synthetic_workload(n_samples=100)
            trainer = ModelTrainer()
            _, preds, test_df = trainer.train_and_evaluate_all(df)
            best_preds = preds[trainer.best_model_name]

            procs = [
                Process(
                    pid=int(r["job_id"]),
                    arrival_time=float(r["submit_time"]),
                    burst_time=float(r[TARGET_COL]),
                    predicted_burst=float(best_preds[i]),
                    base_priority=int(r.get("base_priority", 5)),
                )
                for i, (_, r) in enumerate(test_df.iterrows())
            ]
            min_a = min(p.arrival_time for p in procs)
            for p in procs:
                p.arrival_time -= min_a

            rows = []
            for S in [FCFSScheduler(), SJFScheduler(), SRTFScheduler(), RoundRobinScheduler(10.0), OptiQScheduler(10.0, 20.0, 100.0)]:
                sumry, _ = S.run(procs)
                rows.append({
                    "Scheduler": S.name,
                    "Avg Wait": sumry.avg_waiting_time,
                    "Avg Turnaround": sumry.avg_turnaround_time,
                    "Context Switches": sumry.total_context_switches,
                })
            print("\n" + pd.DataFrame(rows).to_string(index=False))

        elif choice == "4":
            if not os.path.exists("outputs"):
                print("No outputs found yet.")
            else:
                runs = sorted(os.listdir("outputs"))
                print(f"Available runs in outputs/: {runs}")
                if runs:
                    latest = runs[-1]
                    latest_comp = os.path.join("outputs", latest, "algorithm_comparison.csv")
                    if os.path.exists(latest_comp):
                        print(f"\n--- Latest Run ({latest}) Summary ---")
                        print(pd.read_csv(latest_comp).to_string(index=False))

        elif choice == "5":
            print("Exiting OptiQ CLI. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice, please select 1-5.")


def main():
    parser = argparse.ArgumentParser(description="OptiQ CPU Scheduling Framework CLI")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive CLI wizard")
    parser.add_argument("--source", type=str, default="real", choices=["real", "synthetic"], help="Workload source")
    parser.add_argument("--jobs", type=int, default=500, help="Number of jobs to process")
    parser.add_argument("--quantum", type=float, default=10.0, help="Time quantum in seconds")
    parser.add_argument("--aging", type=float, default=20.0, help="Priority aging interval in seconds")
    parser.add_argument("--starvation", type=float, default=100.0, help="Starvation threshold in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        cli_wizard()
    else:
        run_full_experiment(
            data_source=args.source,
            n_jobs=args.jobs,
            quantum=args.quantum,
            aging_interval=args.aging,
            starvation_threshold=args.starvation,
            seed=args.seed,
            output_base_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
