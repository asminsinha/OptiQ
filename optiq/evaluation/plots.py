"""Visualization and telemetry plot generation for OptiQ:
1. Actual vs Predicted Scatter Plot & Residual Distribution Plot
2. ML Models Metric Comparison Bar Chart
3. Feature Importances Chart
4. Schedulers Metric Comparison Bar Chart (Waiting Time, Turnaround, Response)
5. Waiting Time Distributions & Fairness Boxplot
6. Policy Ablation Comparison Plot
7. Simulation Gantt Chart
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optiq.simulator.process import ExecutionSlice


def plot_ml_predictions_and_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str,
) -> str:
    """Generate actual vs predicted scatter plot and residual distribution with units and counts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Scatter Plot
    ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors="none", color="#1f77b4")
    min_val = max(0.1, min(float(y_true.min()), float(y_pred.min())))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.8, label="Ideal (y = x)")
    ax1.set_xlabel("Actual Service Time (s)")
    ax1.set_ylabel("Predicted Service Time (s)")
    ax1.set_title(f"{model_name}: Actual vs Predicted (N={len(y_true)})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Residual Distribution
    residuals = y_pred - y_true
    ax2.hist(residuals, bins=30, color="#2ca02c", edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    ax2.set_xlabel("Residual (Predicted - Actual) (s)")
    ax2.set_ylabel("Frequency (Job Count)")
    ax2.set_title(f"{model_name}: Residuals (Mean: {residuals.mean():.1f}s, Median: {np.median(residuals):.1f}s)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_ml_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Bar chart comparing ML models across MAE, RMSE, and R2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    models = metrics_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    mae_col = "Test MAE (s)" if "Test MAE (s)" in metrics_df.columns else "MAE (s)"
    rmse_col = "Test RMSE (s)" if "Test RMSE (s)" in metrics_df.columns else "RMSE (s)"
    r2_col = "Test R2" if "Test R2" in metrics_df.columns else "R2 Score"

    # MAE & RMSE
    ax1.bar(x - width/2, metrics_df[mae_col], width, label="Test MAE (s)", color="#4c72b0")
    ax1.bar(x + width/2, metrics_df[rmse_col], width, label="Test RMSE (s)", color="#dd8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.set_ylabel("Error in Seconds (Lower is Better)")
    ax1.set_title("Model Error Comparison (MAE & RMSE)")
    ax1.legend()
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # R2 Score
    r2_vals = metrics_df[r2_col].fillna(0.0)
    r2_colors = ["#55a868" if r >= 0 else "#c44e52" for r in r2_vals]
    ax2.bar(x, r2_vals, width=0.5, color=r2_colors)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.set_ylabel("R2 Score (Higher is Better)")
    ax2.set_title("Model Goodness of Fit (R2 Score)")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_feature_importance(
    fi_df: pd.DataFrame,
    model_name: str,
    output_path: str,
) -> str:
    """Horizontal bar chart for model feature importance."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fi_sorted = fi_df.sort_values(by="Importance", ascending=True)

    ax.barh(fi_sorted["Feature"], fi_sorted["Importance"], color="#3470a3", edgecolor="black")
    ax.set_xlabel("Relative Importance Score")
    ax.set_title(f"Feature Importance ({model_name})")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_scheduler_comparison(
    summary_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Grouped bar chart comparing scheduling algorithms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    schedulers = summary_df["Scheduler"].tolist()
    y_pos = np.arange(len(schedulers))

    # Turnaround & Waiting
    width = 0.35
    ax1.barh(y_pos - width/2, summary_df["Avg Waiting Time (s)"], width, label="Avg Wait (s)", color="#4c72b0")
    ax1.barh(y_pos + width/2, summary_df["Avg Turnaround Time (s)"], width, label="Avg Turnaround (s)", color="#dd8452")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(schedulers)
    ax1.set_xlabel("Seconds (Lower is Better)")
    ax1.set_title("Waiting & Turnaround Time Comparison")
    ax1.legend()
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    ax1.invert_yaxis()

    # Context Switches & Starvation Risk Count
    ax2.barh(y_pos - width/2, summary_df["Context Switches"], width, label="Context Switches", color="#8172b3")
    ax2.barh(y_pos + width/2, summary_df["Starvation Risk Count"], width, label="Starvation Count", color="#c44e52")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(schedulers)
    ax2.set_xlabel("Count (Lower is Better)")
    ax2.set_title("Overhead & Starvation Risk")
    ax2.legend()
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_waiting_time_distribution(
    per_process_dfs: Dict[str, pd.DataFrame],
    output_path: str,
) -> str:
    """Boxplot displaying per-algorithm waiting time distribution and dispersion."""
    fig, ax = plt.subplots(figsize=(13, 6))

    labels = []
    data = []
    for s_name, df in per_process_dfs.items():
        if "waiting_time" in df.columns:
            labels.append(s_name)
            data.append(df["waiting_time"].values)

    ax.boxplot(data, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.set_ylabel("Waiting Time (s) [Linear Scale]")
    ax.set_title("Waiting Time Distribution & Dispersion across Schedulers")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_ablation_comparison(
    ablation_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Comparative bar chart showing policy ablation study progression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    variants = ablation_df["Policy"].tolist()
    y_pos = np.arange(len(variants))
    width = 0.35

    ax1.barh(y_pos - width/2, ablation_df["Avg Waiting Time (s)"], width, label="Mean Wait (s)", color="#4c72b0")
    ax1.barh(y_pos + width/2, ablation_df["Median Waiting Time (s)"], width, label="Median Wait (s)", color="#55a868")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(variants)
    ax1.set_xlabel("Seconds (Lower is Better)")
    ax1.set_title("Policy Ablation: Mean vs Median Wait Time")
    ax1.legend()
    ax1.grid(axis="x", linestyle=":", alpha=0.6)
    ax1.invert_yaxis()

    ax2.barh(y_pos, ablation_df["Context Switches"], height=0.5, color="#8172b3")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(variants)
    ax2.set_xlabel("Context Switches (Lower is Better)")
    ax2.set_title("Policy Ablation: Context Switch Overhead")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)
    ax2.invert_yaxis()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_gantt_chart(
    timeline: List[ExecutionSlice],
    title: str,
    output_path: str,
    max_slices: int = 50,
) -> str:
    """Render a clean timeline Gantt chart for simulator execution."""
    fig, ax = plt.subplots(figsize=(14, 4.5))

    slices = timeline[:max_slices]
    pids = sorted(list({s.pid for s in slices}))
    pid_to_y = {pid: i for i, pid in enumerate(pids)}

    cmap = matplotlib.colormaps.get_cmap("tab20")

    for s in slices:
        y_val = pid_to_y[s.pid]
        ax.broken_barh(
            [(s.start_time, s.duration)],
            (y_val - 0.35, 0.7),
            facecolors=cmap(y_val % 20),
            edgecolor="black",
            linewidth=0.8,
        )

    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels([f"PID {p}" for p in pids])
    ax.set_xlabel("Simulation Time (seconds)")
    ax.set_title(f"{title} (First {len(slices)} Slices)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
