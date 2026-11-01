# OptiQ: Adaptive Machine Learning-Based Hybrid CPU Scheduling Framework

OptiQ is an adaptive CPU scheduling framework combining machine learning burst prediction, dynamic priority aging, and starvation override policies to balance average waiting time with fair starvation prevention.

---

## 1. Environment & Setup

Ensure Python 3.11+ is installed.

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Core Dependencies:
- `pandas>=2.2.0`
- `numpy>=1.26.0`
- `scikit-learn>=1.4.0`
- `matplotlib>=3.8.0`
- `joblib>=1.3.0`
- `pytest>=8.0.0`
- `xgboost>=2.0.0`
- `lightgbm>=4.0.0`

---

## 2. Dataset Acquisition and Placement

The real trace supported is the Standard Workload Format (SWF) trace from the Parallel Workloads Archive:
- **Default Real Trace**: LANL Thinking Machines CM-5 cleaned trace (`LANL-CM5-1994-4.1-cln.swf.gz`).
- **Placement**: Place trace files inside `data/raw/LANL-CM5-1994-4.1-cln.swf.gz`.
- **Manual Download (if needed)**:
  ```bash
  mkdir -p data/raw
  # Download from Parallel Workloads Archive:
  # URL: http://www.cs.huji.ac.il/labs/parallel/workload/l_lanl_cm5/LANL-CM5-1994-4.1-cln.swf.gz
  ```
- **Synthetic Workload**: A seeded synthetic generator (`--source synthetic`) is included with three profiles: `mixed`, `short_heavy`, and `long_heavy`.

> [!NOTE]
> HPC job runtime acts as a service-time proxy in a simplified single-logical-CPU simulator, not identical to OS kernel process CPU bursts.

---

## 3. Running the Framework

### Interactive Wizard (Recommended)
Launch the interactive terminal menu:
```bash
python -m optiq.cli.main --interactive
```

### Running the Interactive Web Dashboard & API
OptiQ includes a web interface connected to the FastAPI backend with 6 functional pages (Overview, Benchmark, Simulation, ML Diagnostics, Ablation & Fairness, and Experiment Registry):

```powershell
# Launch FastAPI backend server with mounted interactive UI (Port 8000)
python -m uvicorn optiq.api.server:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:800`** in any browser.

### Direct CLI Commands

1. **Full End-to-End Experiment on Real Trace**:
   ```powershell
   python -m optiq.cli.main --source real --jobs 300 --quantum 30.0 --aging 60.0 --starvation 300.0
   ```

2. **Full End-to-End Experiment on Seeded Synthetic Workload**:
   ```powershell
   python -m optiq.cli.main --source synthetic --jobs 250 --quantum 15.0 --aging 45.0 --starvation 100.0
   ```

3. **Running the Unit and Integration Tests**:
   ```powershell
   python -m pytest tests/test_optiq.py -v
   ```

---

## 4. Output Locations

Every run creates an isolated timestamped directory under `outputs/run_YYYYMMDD_HHMMSS/`:
- `algorithm_comparison.csv`: Comparative benchmark across FCFS, SJF (Oracle), ML-SJF, SRTF (Oracle), Round Robin, Static Priority, and OptiQ.
- `ablation_study.csv`: Controlled ablation evaluating OptiQ-Baseline, OptiQ-PredictionOnly, OptiQ-AgingOnly, and OptiQ-Hybrid (Final).
- `model_metrics.csv`: Regression metrics (MAE, RMSE, R2, Inference Time) across MedianBaseline, Ridge, RandomForest, XGBoost, and LightGBM.
- `predictions.csv`: Held-out test set ground truth vs model predictions.
- `optiq_scheduling_trace.csv`: Granular decision-by-decision log recording ready queue, chosen PID, effective priority, and selection reasons.
- `actual_vs_predicted.png`: Scatter plot and residual distribution.
- `ml_model_comparison.png`: MAE, RMSE, and R2 comparison charts.
- `feature_importance.png`: Feature importance breakdown.
- `scheduler_comparison.png`: Comparative metrics bar charts.
- `waiting_time_distribution.png`: Boxplot showing waiting time distribution and dispersion.
- `ablation_comparison.png`: Visual comparison of ablation policies.
- `optiq_gantt_chart.png`: Visual CPU execution timeline slices.
- `experiment_metadata.json`: Audit log of parameters, dataset provenance, and runtime settings.

Model artifacts and scalers are saved under `saved_models/run_YYYYMMDD_HHMMSS/`.

---

## 5. Metric Definitions & Simulation Invariants

- **Turnaround Time**: $\text{Completion Time} - \text{Arrival Time}$
- **Waiting Time**: $\text{Turnaround Time} - \text{Burst Time}$
- **Response Time**: $\text{First CPU Start Time} - \text{Arrival Time}$
- **Makespan**: $\max(\text{Completion Time}) - \min(\text{Arrival Time})$.
  *(Note: For dense single-CPU workloads without intermediate idle gaps, all work-conserving schedulers produce identical makespan $\sum \text{burst}_i$ and throughput).*
- **Throughput**: $\frac{\text{Completed Processes}}{\text{Makespan}}$
- **Context Switches**: Count of process-to-process transitions between different non-idle PIDs (consecutive time slices allocated to the same process do not increment context switches).
- **Starvation Risk Count**: Number of processes with $\text{Waiting Time} \ge \text{starvation\_threshold}$.
- **Jain's Fairness Index**: $\frac{(\sum W_i)^2}{n \sum W_i^2}$, measuring dispersion and fairness across process waiting times.
- **Effective Priority (Aging)**: $\min(\text{max\_priority\_ceiling}, \text{base\_priority} + \lfloor \frac{\text{ready\_queue\_wait}}{\text{aging\_interval}} \rfloor)$, where $\text{ready\_queue\_wait}$ resets to 0 whenever the process receives CPU service.

---

## 6. Target Leakage Prevention

Strict pre-scheduling feature validation is enforced in `optiq/data/loader.py`:
- **Features Used**: `req_cpus`, `req_time`, `req_mem`, `queue_num`, `submit_hour`, `submit_dayofweek`, `inter_arrival_time`.
- **Target Variable**: `run_time` (HPC job runtime proxy for CPU burst).
- **Excluded Fields**: All post-execution metadata (actual runtime, actual wait time, CPU usage, used memory, completion status, user IDs, and group IDs) are excluded from training inputs to prevent data snooping and spurious memorization.
- **Model Selection Protocol**: Best model selection is performed strictly on validation-set MAE. Test-set evaluation occurs only once on the final selected model.
