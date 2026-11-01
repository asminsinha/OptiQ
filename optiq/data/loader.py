"""Data loading, cleaning, validation, and feature preparation for OptiQ.

Supports:
1. Standard Workload Format (SWF) HPC traces (e.g. LANL-CM5).
2. Seeded synthetic workload generation for deterministic testing.
"""

from dataclasses import dataclass, field
import gzip
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

SWF_COLUMNS = [
    "job_id",
    "submit_time",
    "wait_time",
    "run_time",
    "alloc_cpus",
    "avg_cpu_time",
    "used_mem",
    "req_cpus",
    "req_time",
    "req_mem",
    "status",
    "user_id",
    "group_id",
    "executable",
    "queue_num",
    "partition_num",
    "preceding_job",
    "think_time",
]

# Robust pre-scheduling features without high-cardinality spurious IDs
PRE_SCHEDULING_FEATURES = [
    "req_cpus",
    "req_time",
    "req_mem",
    "queue_num",
    "submit_hour",
    "submit_dayofweek",
    "inter_arrival_time",
]

TARGET_COL = "run_time"


@dataclass
class DatasetMetadata:
    """Provenance and audit metadata for loaded datasets."""
    source_type: str
    filepath_or_name: str
    total_records_raw: int
    records_after_cleaning: int
    excluded_records: int
    features: List[str]
    target: str
    target_mean: float
    target_median: float
    target_std: float
    target_min: float
    target_max: float
    notes: str = ""


def parse_swf_trace(
    filepath: str,
    max_records: Optional[int] = 50000,
    min_runtime: float = 1.0,
    max_runtime: Optional[float] = 86400.0,
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Parse a Standard Workload Format (SWF) trace with robust validation.

    CRITICAL DATA INTEGRITY RULES:
    1. Only pre-scheduling attributes and arrival times are extracted as features.
    2. Zero or negative runtimes are excluded (min_runtime >= 1.0s).
    3. Missing / sentinel values (-1 in SWF) are sanitized properly without leakage.
    4. Categorical ID proxies (user_id, group_id) excluded from features to prevent memorization.
    5. HPC job runtime is treated as a service-time proxy in a single-logical-CPU simulator.

    Args:
        filepath: Path to .swf or .swf.gz file.
        max_records: Maximum rows to parse (to maintain bounded execution).
        min_runtime: Minimum valid runtime in seconds.
        max_runtime: Maximum valid runtime in seconds (None for unlimited).

    Returns:
        Cleaned pandas DataFrame and DatasetMetadata audit record.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trace file not found: {filepath}")

    open_fn = gzip.open if filepath.endswith(".gz") else open
    records: List[List[float]] = []
    raw_count = 0

    with open_fn(filepath, "rt", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith(";"):
                continue
            raw_count += 1
            parts = line_str.split()
            if len(parts) < 18:
                continue
            try:
                row_vals = [float(p) for p in parts[:18]]
                records.append(row_vals)
            except ValueError:
                continue

            if max_records and len(records) >= max_records:
                break

    if not records:
        raise ValueError(f"No valid data records parsed from {filepath}")

    df = pd.DataFrame(records, columns=SWF_COLUMNS)

    # Convert ID and time columns
    df["job_id"] = df["job_id"].astype(int)
    df["submit_time"] = df["submit_time"].clip(lower=0.0)

    # Exclude invalid / cancelled / 0-runtime jobs
    valid_mask = df["run_time"] >= min_runtime
    if max_runtime is not None:
        valid_mask = valid_mask & (df["run_time"] <= max_runtime)

    df_clean = df[valid_mask].copy().reset_index(drop=True)

    # Feature Engineering (Strictly Pre-Scheduling Only)
    df_clean["submit_hour"] = (df_clean["submit_time"] % 86400 // 3600).astype(int)
    df_clean["submit_dayofweek"] = (df_clean["submit_time"] // 86400 % 7).astype(int)

    # Inter-arrival time (submit_time[i] - submit_time[i-1])
    submit_diff = df_clean["submit_time"].diff().fillna(0.0)
    df_clean["inter_arrival_time"] = submit_diff.clip(lower=0.0)

    # Sanitize pre-scheduling fields
    df_clean["req_cpus"] = np.where(
        df_clean["req_cpus"] > 0,
        df_clean["req_cpus"],
        np.where(df_clean["alloc_cpus"] > 0, df_clean["alloc_cpus"], 1.0),
    )

    pos_req_time = df_clean.loc[df_clean["req_time"] > 0, "req_time"]
    median_req_time = pos_req_time.median() if not pos_req_time.empty else 3600.0
    df_clean["req_time"] = np.where(
        df_clean["req_time"] > 0, df_clean["req_time"], median_req_time
    )

    df_clean["req_mem"] = np.where(df_clean["req_mem"] > 0, df_clean["req_mem"], 0.0)

    for col in ["user_id", "group_id", "queue_num"]:
        df_clean[col] = df_clean[col].clip(lower=0).astype(int)

    # Base priority proxy: from requested time (higher priority = shorter requested window, in range [1, 10])
    df_clean["base_priority"] = np.clip(10 - np.log1p(df_clean["req_time"]).astype(int), 1, 10)

    # Ensure chronological order
    df_clean = df_clean.sort_values(by="submit_time").reset_index(drop=True)

    target_series = df_clean[TARGET_COL]
    metadata = DatasetMetadata(
        source_type="SWF_REAL_TRACE",
        filepath_or_name=os.path.basename(filepath),
        total_records_raw=raw_count,
        records_after_cleaning=len(df_clean),
        excluded_records=raw_count - len(df_clean),
        features=PRE_SCHEDULING_FEATURES,
        target=TARGET_COL,
        target_mean=float(target_series.mean()),
        target_median=float(target_series.median()),
        target_std=float(target_series.std()),
        target_min=float(target_series.min()),
        target_max=float(target_series.max()),
        notes="HPC job runtime proxy in a single-logical-CPU simulation. No post-execution features leaked. High-cardinality IDs excluded from features.",
    )

    return df_clean, metadata


def generate_synthetic_workload(
    n_samples: int = 1000,
    seed: int = 42,
    profile: str = "mixed",
) -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Generate a reproducible, seeded synthetic workload trace for simulator testing.

    Explicitly labeled SYNTHETIC_WORKLOAD. Does NOT masquerade as real data.

    Workload Profiles:
        - 'mixed': Balanced mix of short interactive and long batch jobs (bimodal lognormal).
        - 'short_heavy': Predominantly short bursts (web/interactive).
        - 'long_heavy': Heavy compute/batch workloads.

    Returns:
        DataFrame with pre-scheduling features, arrival times, base priorities, and runtimes.
    """
    rng = np.random.default_rng(seed)

    if profile == "short_heavy":
        lambda_rate = 0.5
    elif profile == "long_heavy":
        lambda_rate = 0.05
    else:
        lambda_rate = 0.15

    inter_arrivals = rng.exponential(scale=1.0 / lambda_rate, size=n_samples)
    inter_arrivals[0] = 0.0
    submit_times = np.cumsum(inter_arrivals)

    user_ids = rng.integers(1, 25, size=n_samples)
    group_ids = rng.integers(1, 5, size=n_samples)
    queue_nums = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    req_cpus = rng.choice([1, 2, 4, 8, 16], size=n_samples, p=[0.5, 0.25, 0.15, 0.07, 0.03])

    if profile == "short_heavy":
        raw_runtimes = rng.lognormal(mean=1.5, sigma=0.8, size=n_samples)
    elif profile == "long_heavy":
        raw_runtimes = rng.lognormal(mean=4.0, sigma=1.0, size=n_samples)
    else:
        is_short = rng.random(size=n_samples) < 0.65
        short_bursts = rng.lognormal(mean=1.8, sigma=0.7, size=n_samples)
        long_bursts = rng.lognormal(mean=4.2, sigma=0.9, size=n_samples)
        raw_runtimes = np.where(is_short, short_bursts, long_bursts)

    run_times = np.clip(np.round(raw_runtimes, 1), 1.0, 3600.0)

    estimation_multipliers = rng.uniform(1.1, 2.5, size=n_samples)
    req_times = np.clip(np.round(run_times * estimation_multipliers), 5.0, 7200.0)
    req_mem = req_cpus * rng.choice([512, 1024, 2048, 4096], size=n_samples)
    base_priority = rng.integers(1, 11, size=n_samples)

    submit_hour = (submit_times % 86400 // 3600).astype(int)
    submit_dayofweek = (submit_times // 86400 % 7).astype(int)

    df = pd.DataFrame({
        "job_id": np.arange(1, n_samples + 1),
        "submit_time": submit_times,
        "inter_arrival_time": inter_arrivals,
        "req_cpus": req_cpus.astype(float),
        "req_time": req_times.astype(float),
        "req_mem": req_mem.astype(float),
        "user_id": user_ids,
        "group_id": group_ids,
        "queue_num": queue_nums,
        "submit_hour": submit_hour,
        "submit_dayofweek": submit_dayofweek,
        "base_priority": base_priority,
        "run_time": run_times,
    })

    metadata = DatasetMetadata(
        source_type="SYNTHETIC_WORKLOAD",
        filepath_or_name=f"synthetic_{profile}_seed{seed}_n{n_samples}",
        total_records_raw=n_samples,
        records_after_cleaning=n_samples,
        excluded_records=0,
        features=PRE_SCHEDULING_FEATURES,
        target=TARGET_COL,
        target_mean=float(df[TARGET_COL].mean()),
        target_median=float(df[TARGET_COL].median()),
        target_std=float(df[TARGET_COL].std()),
        target_min=float(df[TARGET_COL].min()),
        target_max=float(df[TARGET_COL].max()),
        notes="Seeded synthetic benchmark trace. Strictly labeled for repeatable simulator verification.",
    )

    return df, metadata
