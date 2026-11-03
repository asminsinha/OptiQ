
# OptiQ
## An Adaptive Machine Learning-Based Hybrid CPU Scheduling Framework

<p align="center">
  <strong>Predictive Scheduling · Dynamic Priority Aging · Starvation Prevention · Explainable Evaluation</strong>
</p>

OptiQ is an adaptive CPU scheduling framework that combines machine learning-based CPU burst prediction with dynamic priority aging and starvation override mechanisms.

Traditional CPU scheduling algorithms generally rely on fixed scheduling rules or require prior knowledge of process burst times. OptiQ explores a hybrid approach in which historical workload characteristics are used to estimate job execution requirements, while scheduling policies dynamically account for waiting time and starvation risk.

The framework implements a complete experimental pipeline encompassing workload ingestion, feature engineering, regression model training, scheduler simulation, comparative evaluation, ablation studies, visualization, and an interactive web dashboard.

Rather than evaluating scheduling performance using a single metric, OptiQ examines the relationship between average waiting time, turnaround time, response time, throughput, context switches, and fairness.

> **Important scope clarification:** OptiQ is an experimental scheduling simulator, not a replacement for the Linux or Windows kernel scheduler. It uses HPC workload execution time as a proxy for CPU service time within a simplified single-logical-CPU simulation environment.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem](#2-the-problem)
3. [Project Objectives](#3-project-objectives)
4. [How OptiQ Works](#4-how-optiq-works)
5. [System Architecture](#5-system-architecture)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Regression Models](#7-regression-models)
8. [Feature Engineering and Target Leakage Prevention](#8-feature-engineering-and-target-leakage-prevention)
9. [CPU Scheduling Algorithms](#9-cpu-scheduling-algorithms)
10. [The OptiQ Hybrid Scheduling Policy](#10-the-optiq-hybrid-scheduling-policy)
11. [Dynamic Priority Aging](#11-dynamic-priority-aging)
12. [Starvation Prevention](#12-starvation-prevention)
13. [Experimental Evaluation](#13-experimental-evaluation)
14. [Performance Metrics](#14-performance-metrics)
15. [Ablation Study](#15-ablation-study)
16. [Interactive Web Dashboard](#16-interactive-web-dashboard)
17. [Generated Outputs](#17-generated-outputs)
18. [Dataset and Workload Generation](#18-dataset-and-workload-generation)
19. [Installation and Environment Setup](#19-installation-and-environment-setup)
20. [Running OptiQ](#20-running-optiq)
21. [Running Tests](#21-running-tests)
22. [Project Directory Structure](#22-project-directory-structure)
23. [Technology Stack](#23-technology-stack)
24. [Design Decisions and Engineering Considerations](#24-design-decisions-and-engineering-considerations)
25. [Limitations and Future Improvements](#25-limitations-and-future-improvements)
26. [Author](#26-author)

---

# 1. Project Overview

CPU scheduling is one of the fundamental responsibilities of an operating system. Whenever multiple processes compete for CPU resources, the scheduler determines which process should execute, when it should execute, and how CPU time should be distributed.

Different scheduling policies produce different outcomes depending on the characteristics of the workload.

For example:

- Short jobs may benefit from Shortest Job First scheduling.
- Interactive workloads require low response times.
- Time-sharing environments benefit from Round Robin scheduling.
- Systems with processes of different importance may use priority-based scheduling.
- Long-running processes require protection against indefinite postponement.

No single traditional scheduling algorithm performs optimally across every workload characteristic.

OptiQ investigates whether machine learning-based workload prediction, combined with dynamic scheduling policies, can improve scheduling decisions while maintaining reasonable fairness.

### Core capabilities

| Component | Function |
|---|---|
| Workload Loader | Loads real HPC traces or generates synthetic workloads |
| Feature Engineering | Extracts pre-execution workload characteristics |
| ML Training Pipeline | Trains and evaluates CPU burst prediction models |
| Scheduler Simulator | Simulates multiple CPU scheduling policies |
| Adaptive Scheduling | Combines predictions with dynamic priority aging |
| Starvation Override | Prevents processes from waiting indefinitely |
| Benchmark Engine | Compares scheduling algorithms using common workloads |
| Ablation Engine | Evaluates individual components of the hybrid policy |
| Visualization Engine | Generates performance plots and scheduling timelines |
| FastAPI Backend | Exposes experiment and dashboard functionality |
| Web Dashboard | Presents experiment results through an interactive interface |

---

# 2. The Problem

Traditional CPU scheduling algorithms are designed around different assumptions about process execution requirements.

Some algorithms require the scheduler to know the CPU burst time in advance. Others use fixed priorities, arrival order, or predetermined time slices.

These approaches introduce several trade-offs.

### 2.1 First Come First Served

First Come First Served (FCFS) executes processes in their order of arrival.

Although simple, FCFS can suffer from the convoy effect.

A long-running process at the beginning of the queue can force several short processes to wait, increasing average waiting time.

### 2.2 Shortest Job First

Shortest Job First (SJF) selects the process with the smallest expected CPU burst.

It can minimize average waiting time under suitable assumptions, but it requires knowledge of burst lengths.

In real systems, future CPU burst durations are generally unknown.

### 2.3 Round Robin

Round Robin allocates a fixed time quantum to each ready process.

It is suitable for time-sharing environments, but its performance depends heavily on the chosen quantum.

A very small quantum increases context-switch overhead, while a very large quantum makes the algorithm behave more like FCFS.

### 2.4 Static Priority Scheduling

Priority scheduling executes processes according to assigned priority values.

However, processes with lower priority can experience prolonged waiting when higher-priority processes continuously occupy the CPU.

This creates a starvation problem.

### 2.5 The Motivation Behind OptiQ

OptiQ explores three complementary mechanisms:

1. **Prediction:** Estimate job execution requirements using machine learning.
2. **Adaptation:** Adjust scheduling priority according to how long a process has waited.
3. **Protection:** Override normal scheduling decisions when a process crosses a starvation threshold.

The objective is not simply to minimize waiting time at any cost. It is to investigate a more balanced scheduling policy that considers both efficiency and fairness.

---

# 3. Project Objectives

The main objectives of OptiQ are:

- Develop a machine learning pipeline for estimating workload execution time.
- Use pre-execution workload characteristics rather than post-execution information.
- Compare multiple regression models using validation and test metrics.
- Implement and benchmark traditional CPU scheduling algorithms.
- Investigate prediction-assisted scheduling.
- Introduce dynamic priority aging to account for waiting time.
- Introduce a starvation override mechanism.
- Evaluate individual scheduling components through controlled ablation experiments.
- Measure scheduling performance across multiple metrics.
- Generate reproducible experiment outputs.
- Provide an interactive interface for inspecting benchmark results.

---

# 4. How OptiQ Works

OptiQ follows an end-to-end experimental workflow.

### Overall execution pipeline

```text
             WORKLOAD INPUT
                   |
          +--------+--------+
          |                 |
       Real SWF          Synthetic
       Dataset           Generator
          |                 |
          +--------+--------+
                   |
             DATA LOADING
                   |
          FEATURE ENGINEERING
                   |
          TRAIN / VALIDATION /
              TEST SPLIT
                   |
          +--------+--------+
          |                 |
      ML Training       Workload
      and Evaluation    Preparation
          |                 |
      Best Model             |
      Selection              |
          |                 |
          +--------+--------+
                   |
          BURST PREDICTION
                   |
          SCHEDULER SIMULATION
                   |
          +--------+--------+
          |                 |
      Traditional        OptiQ Hybrid
      Schedulers         Scheduling
          |                 |
          +--------+--------+
                   |
          PERFORMANCE EVALUATION
                   |
          +--------+--------+
          |        |        |
       Metrics   Ablation  Scheduling
       Analysis  Analysis  Trace
          |        |        |
          +--------+--------+
                   |
          CSV / JSON / PLOTS
                   |
          INTERACTIVE DASHBOARD
```

### Execution stages

**Stage 1 — Workload acquisition**

The framework obtains process information from either a real Standard Workload Format trace or the synthetic workload generator.

**Stage 2 — Data preparation**

The workload loader extracts the relevant job characteristics, performs feature preparation, and constructs the data required by the machine learning pipeline.

**Stage 3 — Model training**

Multiple regression algorithms are trained to estimate job execution time. Their performance is evaluated using regression metrics.

**Stage 4 — Model selection**

The best-performing candidate is selected using validation-set Mean Absolute Error (MAE).

**Stage 5 — Scheduling simulation**

The simulator executes the workload under different scheduling algorithms, including OptiQ.

**Stage 6 — Performance evaluation**

Scheduling metrics are calculated and stored for comparative analysis.

**Stage 7 — Experiment analysis**

The framework performs ablation analysis and generates visualizations, scheduling traces, and experiment metadata.

**Stage 8 — Dashboard presentation**

The FastAPI backend exposes experiment information to the interactive web interface.

---

# 5. System Architecture

OptiQ is organized into modular components so that data ingestion, machine learning, scheduling, simulation, evaluation, and visualization remain separated.

### 5.1 Data Layer

Responsible for obtaining and preparing workload information.

Main responsibilities:

- Loading SWF workload traces.
- Generating synthetic workloads.
- Extracting scheduling-related attributes.
- Constructing machine learning features.
- Preparing training, validation, and test data.

Primary module:

`optiq/data/loader.py`

### 5.2 Machine Learning Layer

Responsible for training regression models and estimating job execution requirements.

Main responsibilities:

- Feature preprocessing.
- Regression model training.
- Validation-based model selection.
- Test-set evaluation.
- Prediction generation.
- Model and scaler serialization.
- Feature importance analysis where supported.

Primary module:

`optiq/models/trainer.py`

### 5.3 Scheduling Layer

Contains the scheduling algorithms used during experimentation.

Main responsibilities:

- FCFS scheduling.
- SJF scheduling.
- ML-assisted SJF scheduling.
- SRTF scheduling.
- Round Robin scheduling.
- Static Priority scheduling.
- OptiQ hybrid scheduling.

Primary modules:

- `optiq/schedulers/base.py`
- `optiq/schedulers/classical.py`
- `optiq/schedulers/optiq_scheduler.py`

### 5.4 Simulation Layer

Responsible for representing process execution and calculating scheduling-related measurements.

Main responsibilities:

- Process state management.
- Arrival and completion tracking.
- CPU allocation.
- Time-slice execution.
- Waiting-time calculation.
- Response-time calculation.
- Context-switch tracking.
- Scheduling timeline generation.

Primary modules:

- `optiq/simulator/process.py`
- `optiq/simulator/metrics.py`

### 5.5 Evaluation Layer

Responsible for evaluating the performance of scheduling algorithms and individual OptiQ components.

Main responsibilities:

- Scheduler benchmarking.
- Ablation experiments.
- Metric aggregation.
- Performance comparisons.
- Plot generation.

Primary modules:

- `optiq/evaluation/ablation.py`
- `optiq/evaluation/plots.py`

### 5.6 API and Presentation Layer

The framework includes a FastAPI backend and a web-based interface.

The backend makes experiment information accessible to the dashboard, while the frontend provides an interactive environment for inspecting scheduling results.

Primary module:

`optiq/api/server.py`

Frontend assets:

- `optiq/web/index.html`
- `optiq/web/style.css`
- `optiq/web/app.js`

---

# 6. Machine Learning Pipeline

A major component of OptiQ is the prediction of job execution requirements.

Traditional SJF and SRTF algorithms assume that burst information is available. OptiQ investigates the use of regression models to estimate execution requirements from information available before execution.

### 6.1 Prediction Objective

The prediction target is:

`run_time`

This represents the recorded runtime of an HPC job.

Within OptiQ, runtime is used as a proxy for CPU service time in the scheduling simulator.

The regression problem can be represented as:

\[
\hat{y}=f(X)
\]

Where:

- \(X\) represents the input workload features.
- \(f\) represents the trained regression model.
- \(y\) represents the recorded job runtime.
- \(\hat{y}\) represents the predicted runtime.

The prediction is subsequently used as an estimated execution requirement during scheduling.

### 6.2 Training Workflow

The ML pipeline follows these stages:

1. Load the workload dataset.
2. Extract permitted pre-execution features.
3. Identify the runtime target variable.
4. Prepare training, validation, and test subsets.
5. Fit the candidate regression models.
6. Calculate validation performance.
7. Select the best model using validation MAE.
8. Evaluate the selected model on the held-out test set.
9. Save model artifacts and the fitted scaler.
10. Generate predictions and model comparison outputs.

### 6.3 Why Multiple Models?

Different regression models capture different types of relationships.

A simple linear model can be useful when relationships are approximately linear. Tree-based ensemble models can represent more complex nonlinear relationships and interactions between workload attributes.

Comparing several models makes it possible to determine which approach performs better for the selected workload and feature representation.

---

# 7. Regression Models

OptiQ evaluates five regression approaches.

## 7.1 Median Baseline

The Median Baseline is a simple reference model.

It predicts the median runtime observed in the training data.

Its purpose is to establish a baseline against which more sophisticated models can be evaluated.

A model that cannot outperform a simple baseline provides limited evidence that its additional complexity is useful.

## 7.2 Ridge Regression

Ridge Regression is a linear regression technique with L2 regularization.

Its objective can be represented as:

\[
\min_{\beta}
\left[
\sum_{i=1}^{n}(y_i-X_i\beta)^2
+
\lambda\sum_{j=1}^{p}\beta_j^2
\right]
\]

The regularization term penalizes excessively large coefficients.

**Advantages:**

- Computationally efficient.
- Relatively simple to interpret.
- Helps control coefficient magnitude.
- Provides a useful linear reference model.

**Limitations:**

- Cannot naturally represent complex nonlinear relationships without additional feature transformations.
- May underfit workloads with complicated interactions.

## 7.3 Random Forest Regressor

Random Forest is an ensemble learning method that combines predictions from multiple decision trees.

Each tree is trained using a randomized sample of the training data and feature space.

The final prediction is obtained by aggregating the individual tree predictions.

**Advantages:**

- Captures nonlinear relationships.
- Handles feature interactions.
- Generally requires limited feature scaling.
- Provides feature importance estimates.

**Limitations:**

- Can require more memory and computation than simpler regression models.
- Large forests may increase inference time.
- Predictions are generally less straightforward to interpret than linear regression.

## 7.4 XGBoost Regressor

XGBoost is a gradient-boosted decision-tree implementation.

Instead of training independent trees, boosting builds an ensemble sequentially, with later trees learning to correct errors made by earlier trees.

**Advantages:**

- Captures complex nonlinear relationships.
- Models feature interactions.
- Includes regularization mechanisms.
- Often performs well on structured tabular datasets.

**Limitations:**

- Requires hyperparameter choices.
- Can overfit without suitable regularization.
- May require more computational resources than simpler models.

## 7.5 LightGBM Regressor

LightGBM is a gradient-boosting framework designed for efficient tree-based learning.

It uses histogram-based techniques and leaf-wise tree growth to improve training efficiency.

**Advantages:**

- Efficient on structured datasets.
- Supports nonlinear feature interactions.
- Can provide strong predictive performance.
- Designed to reduce training overhead on suitable workloads.

**Limitations:**

- Performance depends on hyperparameter configuration.
- Leaf-wise growth can overfit smaller datasets if not controlled.
- Training efficiency does not automatically imply lower inference latency.

### 7.6 Model Comparison

| Model | Learning Approach | Main Strength | Main Trade-off |
|---|---|---|---|
| Median Baseline | Statistical baseline | Simple reference | No feature-dependent learning |
| Ridge Regression | Regularized linear regression | Fast and interpretable | Limited nonlinear representation |
| Random Forest | Bagged decision trees | Nonlinear relationships | Model size and inference cost |
| XGBoost | Gradient-boosted trees | Complex predictive patterns | Tuning and computational cost |
| LightGBM | Histogram-based gradient boosting | Efficient tree boosting | Sensitivity to configuration |

### 7.7 How Model Efficiency Is Evaluated

OptiQ records the following regression metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (\(R^2\))
- Inference Time

These metrics answer different questions.

| Metric | What It Measures | Preferred Direction |
|---|---|---|
| MAE | Average absolute prediction error | Lower |
| RMSE | Prediction error with greater penalty for large mistakes | Lower |
| \(R^2\) | Variance explained by the model | Generally higher |
| Inference Time | Time required to produce predictions | Lower |

A model with the lowest MAE is not necessarily the fastest model.

Likewise, a model with a high \(R^2\) may still produce substantial absolute prediction errors, particularly when the target runtime values have a wide range.

Therefore, model selection and efficiency analysis should consider predictive accuracy and computational cost separately.

**Important:** The actual measured values are generated in `model_metrics.csv`. The README does not assume that XGBoost, LightGBM, or any other model always wins. The winning model depends on the dataset, feature representation, and experimental configuration.

---

# 8. Feature Engineering and Target Leakage Prevention

Feature selection is particularly important in predictive scheduling.

A scheduler must make decisions using information available when a job is waiting to execute. Using information that becomes available only after execution would create an unrealistic prediction environment.

OptiQ therefore uses a restricted set of workload features.

### 8.1 Input Features

| Feature | Description |
|---|---|
| `req_cpus` | Number of CPUs requested by the job |
| `req_time` | Requested execution time |
| `req_mem` | Requested memory |
| `queue_num` | Workload queue identifier |
| `submit_hour` | Hour of job submission |
| `submit_dayofweek` | Day of the week on which the job was submitted |
| `inter_arrival_time` | Time difference between consecutive job arrivals |

These features describe workload characteristics that can be available before job execution.

### 8.2 Target Variable

`run_time`

The actual recorded job runtime is used as the regression target.

In the simulation, this runtime serves as the service-time proxy.

### 8.3 Excluded Information

The following categories of information are excluded from the model's input features:

- Actual runtime.
- Actual waiting time.
- CPU usage measured after execution.
- Used memory.
- Completion status.
- User identifiers.
- Group identifiers.
- Other post-execution metadata.

### 8.4 Why Target Leakage Matters

Target leakage occurs when a model receives information that would not legitimately be available at prediction time.

For example, using actual runtime as an input feature while attempting to predict runtime would make the model appear artificially accurate.

Similarly, using actual waiting time would expose the model to information produced by the scheduling process itself.

Such a model might achieve impressive test metrics while being unsuitable for genuine predictive scheduling.

OptiQ addresses this concern through pre-scheduling feature validation in:

`optiq/data/loader.py`

### 8.5 Model Selection Protocol

The model selection process follows a validation-based approach:

1. Candidate models are trained using the training data.
2. Their validation MAE values are compared.
3. The best candidate is selected using validation MAE.
4. The selected model is evaluated on the held-out test data.

The test set is not used to choose the winning model.

This separation helps reduce optimistic performance estimates caused by repeatedly tuning a model against the test data.

---

# 9. CPU Scheduling Algorithms

OptiQ compares its hybrid scheduling policy against six other scheduling approaches.

The algorithms provide different scheduling strategies and reference points for evaluating the proposed approach.

## 9.1 First Come First Served (FCFS)

FCFS executes processes in arrival order.

**Selection rule:** Choose the earliest-arriving ready process.

**Characteristics:**

- Non-preemptive scheduling.
- Simple implementation.
- No burst prediction required.
- Can suffer from the convoy effect.

FCFS provides a useful baseline for evaluating whether more adaptive policies improve waiting-time distribution.

## 9.2 Shortest Job First (SJF) — Oracle

SJF selects the process with the shortest known CPU burst.

**Selection rule:** Choose the ready process with the smallest burst requirement.

**Characteristics:**

- Non-preemptive scheduling.
- Can minimize average waiting time under standard assumptions.
- Requires knowledge of burst lengths.

In OptiQ, SJF is treated as an oracle benchmark because it has access to actual burst information.

This is important when interpreting results: it is not an equivalent information setting to a scheduler that must predict runtime.

## 9.3 Machine Learning Shortest Job First (ML-SJF)

ML-SJF uses estimated execution requirements instead of actual burst lengths.

**Selection rule:** Choose the ready process with the smallest predicted burst.

**Characteristics:**

- Prediction-assisted scheduling.
- Uses the selected regression model.
- Does not require actual future burst information for its selection rule.
- Performance depends on prediction quality.

ML-SJF helps isolate the contribution of machine learning-based burst prediction.

It also demonstrates that prediction accuracy alone does not guarantee fairness.

## 9.4 Shortest Remaining Time First (SRTF) — Oracle

SRTF is the preemptive counterpart of SJF.

It selects the process with the smallest remaining execution time.

If a newly available process has a smaller remaining requirement than the currently executing process, the scheduler may preempt the current process.

**Characteristics:**

- Preemptive scheduling.
- Uses actual remaining burst information in the oracle benchmark.
- Can produce low average waiting times.
- Can cause long processes to experience prolonged waiting.

SRTF serves as an idealized reference rather than a directly equivalent deployable prediction-based policy.

## 9.5 Round Robin (RR)

Round Robin distributes CPU time using a fixed time quantum.

Each ready process receives up to the configured quantum before the scheduler moves to another process.

**Characteristics:**

- Preemptive scheduling.
- Suitable as a time-sharing reference.
- Provides repeated opportunities for processes to execute.
- Performance depends on the selected time quantum.

A very small quantum can increase context-switch frequency. A very large quantum reduces the algorithm's responsiveness.

## 9.6 Static Priority Scheduling

Static Priority scheduling selects processes according to their assigned priority.

**Characteristics:**

- Priority-driven scheduling.
- Priority values do not dynamically reflect waiting time.
- Lower-priority processes may be postponed by higher-priority processes.

This algorithm provides a useful comparison for understanding the purpose of dynamic priority aging.

## 9.7 OptiQ Hybrid Scheduling

OptiQ combines prediction-assisted scheduling with dynamic priority aging and a starvation override policy.

Its purpose is to balance the preference for shorter jobs against the need to prevent indefinite waiting.

Rather than relying exclusively on predicted burst length or an unchanging priority, OptiQ incorporates additional scheduling considerations.

The complete implementation is located in:

`optiq/schedulers/optiq_scheduler.py`

---

# 10. The OptiQ Hybrid Scheduling Policy

The central idea behind OptiQ is that scheduling decisions should account for more than estimated execution duration.

A scheduler that always prioritizes the shortest predicted job may repeatedly postpone longer jobs.

A scheduler that only follows fixed priorities can create a similar problem for processes assigned lower priorities.

OptiQ introduces additional mechanisms to address these trade-offs.

### 10.1 Prediction-Assisted Decisions

The trained regression model estimates the runtime of each job using its available workload features.

This prediction supplies an estimated service requirement for scheduling decisions.

Unlike oracle SJF and SRTF, prediction-assisted scheduling does not require the actual burst length to be known in advance.

### 10.2 Dynamic Priority Adjustment

The framework incorporates priority aging.

As a process waits in the ready queue, its effective priority is adjusted according to its accumulated ready-queue waiting time.

This gives waiting processes an opportunity to become more competitive during subsequent scheduling decisions.

### 10.3 Starvation Override

A separate starvation threshold provides protection for processes that have waited excessively long.

Once the configured threshold is reached, the normal scheduling preference can be overridden according to the implemented starvation policy.

This prevents the scheduler from treating predicted execution duration as the only consideration.

### 10.4 Why Combine These Mechanisms?

Consider a workload containing:

| Process | Predicted Burst | Waiting Situation |
|---|---:|---|
| P1 | Short | Newly arrived |
| P2 | Medium | Waiting for some time |
| P3 | Long | Waiting for a long time |

A purely prediction-based shortest-job policy may repeatedly prefer newly arriving short processes.

OptiQ introduces waiting-time-based priority adjustment and starvation protection to reduce the risk of repeatedly postponing P2 and P3.

The intention is to retain the benefits of prediction while accounting for the consequences of repeatedly selecting shorter jobs.

---

# 11. Dynamic Priority Aging

Priority aging is a technique used to reduce starvation in priority-based scheduling.

The effective priority in OptiQ is represented by:

\[
P_{\text{effective}} =
\min\left(
P_{\text{ceiling}},
P_{\text{base}}+
\left\lfloor
\frac{W_{\text{ready}}}{T_{\text{aging}}}
\right\rfloor
\right)
\]

Where:

- \(P_{\text{effective}}\): Adjusted priority.
- \(P_{\text{base}}\): Base priority.
- \(P_{\text{ceiling}}\): Maximum permitted priority.
- \(W_{\text{ready}}\): Accumulated ready-queue waiting time.
- \(T_{\text{aging}}\): Configured aging interval.

The implementation resets ready-queue waiting time whenever the process receives CPU service.

### Example

Assume:

- Base priority = 2.
- Aging interval = 20 time units.
- Maximum priority ceiling = 10.

The effective priority changes as follows:

| Ready-Queue Waiting Time | Effective Priority |
|---:|---:|
| 0 | 2 |
| 20 | 3 |
| 40 | 4 |
| 60 | 5 |
| 80 | 6 |
| 100 | 7 |

This illustrates how priority increases as waiting time accumulates, subject to the configured ceiling.

The purpose is to prevent a process from remaining permanently disadvantaged because of its initial priority.

---

# 12. Starvation Prevention

Starvation occurs when a process remains ready to execute but repeatedly fails to receive CPU time.

A process can experience starvation even when the CPU is continuously busy and other processes are completing successfully.

OptiQ introduces a configurable starvation threshold.

The threshold is used to identify processes whose waiting time has become excessive.

The starvation risk count is calculated as:

\[
N_{\text{starved}} =
\sum_{i=1}^{n}
\mathbf{1}
\left(W_i \geq T_{\text{starvation}}\right)
\]

Where:

- \(W_i\): Waiting time of process \(i\).
- \(T_{\text{starvation}}\): Configured starvation threshold.
- \(\mathbf{1}\): Indicator function.

A lower starvation risk count indicates fewer processes exceeded the configured waiting threshold.

### Aging versus Starvation Override

These mechanisms address related but different problems.

| Mechanism | Purpose |
|---|---|
| Priority Aging | Gradually improves the scheduling position of waiting processes |
| Starvation Override | Provides an explicit response when a waiting process crosses a threshold |

Aging is gradual. Starvation override is threshold-based.

Together, they provide additional protection beyond simply selecting the shortest predicted job.

---

# 13. Experimental Evaluation

OptiQ evaluates scheduling policies using a common simulation environment.

The objective is to compare how different algorithms behave under the same workload rather than comparing unrelated workloads.

### 13.1 Scheduler Benchmark

The framework compares:

1. FCFS
2. SJF (Oracle)
3. ML-SJF
4. SRTF (Oracle)
5. Round Robin
6. Static Priority
7. OptiQ

Each algorithm is evaluated using scheduling metrics generated by the simulator.

### 13.2 Why Use Multiple Algorithms?

Each algorithm provides a different reference point.

| Algorithm | What It Helps Evaluate |
|---|---|
| FCFS | Basic arrival-order scheduling |
| SJF | Idealized short-job scheduling |
| ML-SJF | Prediction-assisted short-job scheduling |
| SRTF | Idealized preemptive short-job scheduling |
| Round Robin | Time-sharing behavior |
| Static Priority | Fixed-priority scheduling |
| OptiQ | Prediction combined with adaptive fairness mechanisms |

### 13.3 Interpreting Oracle Comparisons

SJF and SRTF use actual burst information in the oracle benchmarks.

ML-SJF and OptiQ rely on predicted workload requirements.

This creates an important distinction.

Oracle algorithms have information that a practical predictive scheduler would not ordinarily possess.

Therefore, oracle results should be interpreted as idealized reference points, not as evidence that a prediction-based algorithm has access to the same information.

### 13.4 Makespan and Throughput

For a fixed workload on a single CPU, work-conserving schedulers may produce identical total completion horizons when all processes complete and the CPU has the same idle intervals.

In such cases, differences in scheduling policy are more likely to appear in:

- Waiting time.
- Response time.
- Turnaround-time distribution.
- Context switches.
- Starvation risk.
- Fairness.

Identical throughput across scheduling policies is therefore not necessarily a defect.

It can be an expected consequence of processing the same workload under the same single-CPU execution assumptions.

---

# 14. Performance Metrics

OptiQ uses multiple metrics because scheduling performance cannot be described accurately using average waiting time alone.

## 14.1 Turnaround Time

The total time between a process arriving and completing execution.

\[
TAT_i = C_i-A_i
\]

Where:

- \(C_i\): Completion time.
- \(A_i\): Arrival time.

Lower turnaround time generally indicates that processes finish sooner relative to their arrival.

## 14.2 Waiting Time

The total time a process spends waiting in the ready queue.

For the non-I/O workload model:

\[
WT_i = TAT_i-BT_i
\]

Where \(BT_i\) is the process burst time.

Lower average waiting time indicates less time spent waiting for CPU service.

## 14.3 Response Time

The time between process arrival and its first CPU allocation.

\[
RT_i = S_i-A_i
\]

Where \(S_i\) is the first CPU start time.

Response time is particularly relevant to interactive workloads.

## 14.4 Makespan

The total interval between the earliest process arrival and the final process completion.

\[
Makespan =
\max(C_i)-\min(A_i)
\]

Lower makespan means the workload completed within a shorter overall time interval.

## 14.5 Throughput

Throughput measures the number of completed processes per unit of simulated time.

\[
Throughput =
\frac{N_{\text{completed}}}{Makespan}
\]

Higher throughput indicates more completed processes per unit of time.

## 14.6 Context Switches

A context switch occurs when CPU execution moves from one process to another.

OptiQ counts transitions between different non-idle process identifiers.

Consecutive time slices assigned to the same process do not count as process-to-process context switches.

A high number of context switches may indicate additional scheduling overhead in a real system, although the simulator's context-switch metric does not itself measure actual operating-system switching time.

## 14.7 Starvation Risk Count

The number of processes whose waiting time reaches or exceeds the configured starvation threshold.

\[
StarvationRisk =
\sum_{i=1}^{n}
\mathbf{1}(WT_i \geq T_{\text{starvation}})
\]

Lower values indicate fewer processes exceeded the selected threshold.

## 14.8 Jain's Fairness Index

Jain's Fairness Index measures how evenly a set of values is distributed.

The implementation uses process waiting times:

\[
J(W)=
\frac{\left(\sum_{i=1}^{n}W_i\right)^2}
{n\sum_{i=1}^{n}W_i^2}
\]

Where \(W_i\) represents the waiting time of process \(i\).

The index ranges from 0 to 1 when the denominator is nonzero.

A value closer to 1 indicates more equal waiting-time values.

**Interpretation caution:** Equal waiting times do not necessarily mean low waiting times. A scheduler can give every process an equally long wait and still have poor average waiting time. Fairness should therefore be interpreted alongside average waiting time and starvation risk.

If every waiting time is zero, the denominator is zero and the index requires special handling.

## 14.9 Effective Priority

Effective priority is the dynamically adjusted priority after applying the aging mechanism.

\[
P_{\text{effective}} =
\min\left(
P_{\text{ceiling}},
P_{\text{base}}+
\left\lfloor
\frac{W_{\text{ready}}}{T_{\text{aging}}}
\right\rfloor
\right)
\]

It provides a measure of how waiting time influences scheduling priority.

---

# 15. Ablation Study

An ablation study examines the contribution of individual components of a system.

Instead of evaluating only the final hybrid approach, OptiQ compares multiple configurations.

The current experiment output identifies four variants:

| Variant | Purpose |
|---|---|
| OptiQ-Baseline | Baseline scheduling configuration |
| OptiQ-PredictionOnly | Evaluates prediction without the full combination of adaptive mechanisms |
| OptiQ-AgingOnly | Evaluates the contribution of priority aging |
| OptiQ-Hybrid (Final) | Evaluates the combined hybrid configuration |

The ablation study helps investigate whether combining predictive scheduling and fairness mechanisms produces different results from using individual components.

### Why Ablation Matters

Suppose the final hybrid policy has a lower average waiting time than a baseline.

That alone does not reveal which component contributed to the improvement.

Ablation experiments provide a way to investigate the individual contributions of prediction and aging.

The study should be interpreted using multiple measurements:

- Average waiting time.
- Average turnaround time.
- Average response time.
- Context switches.
- Starvation risk count.
- Jain's fairness index.
- Throughput.

### Reproducibility

For a meaningful ablation comparison, variants should use the same workload and compatible simulation parameters.

This reduces the possibility that apparent differences are caused by different input workloads rather than the scheduling policies themselves.

Results are written to:

`ablation_study.csv`

Visual comparisons are written to:

`ablation_comparison.png`

---

# 16. Interactive Web Dashboard

OptiQ includes a FastAPI-powered web dashboard for inspecting experiment results.

The dashboard is organized into six functional pages.

### 16.1 Overview

Provides a high-level view of the experiment.

It serves as the main entry point for understanding the current run and its overall scheduling results.

### 16.2 Benchmark

Presents the comparative evaluation of scheduling algorithms.

It allows the performance of traditional schedulers and OptiQ to be examined across the available metrics.

### 16.3 Simulation

Provides access to the scheduling simulation view and execution-related information.

The generated scheduling trace and Gantt chart provide additional detail about CPU allocation decisions.

### 16.4 ML Diagnostics

Focuses on the machine learning component.

Relevant outputs include:

- Model evaluation metrics.
- Actual-versus-predicted comparisons.
- Prediction error analysis.
- Feature importance visualization.

### 16.5 Ablation and Fairness

Provides access to the ablation variants and fairness-related evaluation.

It helps examine the effects of individual scheduling mechanisms.

### 16.6 Experiment Registry

Provides an interface for inspecting recorded experiment information.

Experiment metadata is stored in JSON format, allowing run configurations and dataset information to be retained alongside generated outputs.

### Starting the Dashboard

From the project root:

```powershell
python -m uvicorn optiq.api.server:app --host 127.0.0.1 --port 8000
```

Open:

http://127.0.0.1:8000

The dashboard runs locally and uses the FastAPI backend.

---

# 17. Generated Outputs

Every experiment creates a timestamped output directory.

Example:

```text
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── algorithm_comparison.csv
    ├── ablation_study.csv
    ├── model_metrics.csv
    ├── predictions.csv
    ├── optiq_scheduling_trace.csv
    ├── actual_vs_predicted.png
    ├── ml_model_comparison.png
    ├── feature_importance.png
    ├── scheduler_comparison.png
    ├── waiting_time_distribution.png
    ├── ablation_comparison.png
    ├── optiq_gantt_chart.png
    └── experiment_metadata.json
```

The exact files generated depend on the execution path and available experiment components.

### 17.1 CSV Files

| File | Contents |
|---|---|
| `algorithm_comparison.csv` | Comparative scheduling metrics |
| `ablation_study.csv` | Results from the OptiQ ablation variants |
| `model_metrics.csv` | Regression model performance |
| `predictions.csv` | Ground-truth and predicted runtime values |
| `optiq_scheduling_trace.csv` | Detailed scheduling decisions and trace information |

### 17.2 Visualization Files

| File | Purpose |
|---|---|
| `actual_vs_predicted.png` | Compares predicted and actual runtime values |
| `ml_model_comparison.png` | Compares regression model performance |
| `feature_importance.png` | Displays feature importance information |
| `scheduler_comparison.png` | Visualizes scheduler benchmark results |
| `waiting_time_distribution.png` | Compares waiting-time distributions |
| `ablation_comparison.png` | Visualizes ablation experiment results |
| `optiq_gantt_chart.png` | Displays CPU execution intervals and process allocation |

### 17.3 Experiment Metadata

`experiment_metadata.json` stores information about the experiment configuration and execution.

It supports reproducibility by retaining information such as parameters, dataset provenance, and runtime settings.

### 17.4 Model Artifacts

Trained models and preprocessing artifacts are saved separately under:

```text
saved_models/
└── run_YYYYMMDD_HHMMSS/
```

The model artifacts are serialized using Joblib.

The saved scaler is important because inference must use the same fitted preprocessing transformation as training.

---

# 18. Dataset and Workload Generation

OptiQ supports real and synthetic workload inputs.

## 18.1 Standard Workload Format

The real workload used by the project is the LANL Thinking Machines CM-5 cleaned trace.

Dataset:

`LANL-CM5-1994-4.1-cln.swf.gz`

Source:

Parallel Workloads Archive

The Standard Workload Format (SWF) is a commonly used format for recording parallel workload job information.

The trace provides historical workload characteristics that can be used for experimentation and runtime prediction.

### Dataset Placement

The expected location is:

```text
data/
└── raw/
    └── LANL-CM5-1994-4.1-cln.swf.gz
```

Download the trace from:

http://www.cs.huji.ac.il/labs/parallel/workload/l_lanl_cm5/LANL-CM5-1994-4.1-cln.swf.gz

Create the directory if necessary:

```powershell
New-Item -ItemType Directory -Force -Path "data/raw"
```

Place the downloaded file inside `data/raw`.

## 18.2 Synthetic Workloads

OptiQ also includes a seeded synthetic workload generator.

Synthetic workloads are useful for controlled experiments because workload characteristics can be varied without relying exclusively on a historical trace.

The supported profiles are:

| Profile | Intended Workload Characteristic |
|---|---|
| `mixed` | Mixed workload characteristics |
| `short_heavy` | Greater representation of shorter jobs |
| `long_heavy` | Greater representation of longer jobs |

Synthetic generation also supports repeatable experiments through a random seed.

## 18.3 Real versus Synthetic Workloads

| Aspect | Real SWF Trace | Synthetic Workload |
|---|---|---|
| Source | Historical HPC workload | Generated workload |
| Characteristics | Recorded workload behavior | Configurable workload profiles |
| Main Use | Evaluation using real workload records | Controlled experimentation |
| Repeatability | Depends on fixed dataset and configuration | Can use a fixed random seed |
| Limitations | Reflects a particular historical workload | May not represent real-world workload distributions |

### Simulation Scope

The workload originates from an HPC environment.

However, OptiQ simplifies the scheduling environment to a single logical CPU.

Consequently, the recorded job runtime is used as a service-time proxy rather than treated as a direct measurement of operating-system CPU burst behavior.

This distinction is important when interpreting the results.

---

# 19. Installation and Environment Setup

## 19.1 Requirements

- Python 3.11 or newer.
- pip.
- Git (for cloning the repository).
- A modern web browser (for the dashboard).

## 19.2 Clone the Repository

```bash
git clone https://github.com/asminsinha/OptiQ.git
cd OptiQ
```

## 19.3 Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 19.4 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Core Dependencies

| Package | Purpose |
|---|---|
| pandas | Workload data processing |
| NumPy | Numerical operations |
| scikit-learn | Regression models and preprocessing |
| Matplotlib | Experiment visualizations |
| Joblib | Model artifact serialization |
| pytest | Automated testing |
| XGBoost | Gradient-boosted regression |
| LightGBM | Efficient gradient-boosted regression |
| FastAPI | Backend API |
| Uvicorn | ASGI application server |

The full dependency list is maintained in `requirements.txt`.

---

# 20. Running OptiQ

All commands should be executed from the project root.

## 20.1 Interactive Wizard

Launch the interactive terminal interface:

```bash
python -m optiq.cli.main --interactive
```

This provides an interactive entry point for configuring and launching experiments.

## 20.2 Full Experiment Using the Real Dataset

```powershell
python -m optiq.cli.main --source real --jobs 300 --quantum 30.0 --aging 60.0 --starvation 300.0
```

This runs the end-to-end experiment using the real workload trace.

The parameters control the workload size and scheduling configuration.

## 20.3 Full Experiment Using Synthetic Workloads

```powershell
python -m optiq.cli.main --source synthetic --jobs 250 --quantum 15.0 --aging 45.0 --starvation 100.0
```

This executes the experiment using generated workload data.

## 20.4 Running the Web Dashboard

Start the backend:

```powershell
python -m uvicorn optiq.api.server:app --host 127.0.0.1 --port 8000
```

Navigate to:

http://127.0.0.1:8000

To stop the server, press `Ctrl + C` in the terminal.

## 20.5 Running Automated Tests

```powershell
python -m pytest tests/test_optiq.py -v
```

The tests cover the project components defined in the test suite, helping verify expected functionality and reduce regressions.

---

# 21. Running Tests

OptiQ includes an automated test suite.

Tests are executed using pytest.

```bash
python -m pytest tests/test_optiq.py -v
```

The test suite is intended to validate the implemented project behavior, including the components covered by the individual test cases.

A successful test run indicates that the tested functionality passed under the tested conditions. It does not, by itself, establish performance superiority or production readiness.

---

# 22. Project Directory Structure

```text
OptiQ/
│
├── data/
│   └── raw/
│       └── LANL-CM5-1994-4.1-cln.swf.gz
│
├── optiq/
│   │
│   ├── api/
│   │   └── server.py
│   │
│   ├── cli/
│   │   └── main.py
│   │
│   ├── data/
│   │   └── loader.py
│   │
│   ├── models/
│   │   └── trainer.py
│   │
│   ├── schedulers/
│   │   ├── base.py
│   │   ├── classical.py
│   │   └── optiq_scheduler.py
│   │
│   ├── simulator/
│   │   ├── metrics.py
│   │   └── process.py
│   │
│   ├── evaluation/
│   │   ├── ablation.py
│   │   └── plots.py
│   │
│   └── web/
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── tests/
│   └── test_optiq.py
│
├── outputs/
│   └── run_YYYYMMDD_HHMMSS/
│
├── saved_models/
│   └── run_YYYYMMDD_HHMMSS/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Generated output and model directories are created during experiment execution.

---

# 23. Technology Stack

| Technology | Role |
|---|---|
| Python | Core implementation |
| pandas | Dataset processing |
| NumPy | Numerical computation |
| scikit-learn | Machine learning and preprocessing |
| Random Forest | Ensemble regression |
| Ridge Regression | Regularized linear regression |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| Matplotlib | Visualization |
| Joblib | Model serialization |
| FastAPI | Backend API |
| Uvicorn | Backend server |
| HTML | Dashboard structure |
| CSS | Dashboard styling |
| JavaScript | Dashboard interaction |
| pytest | Automated testing |
| SWF | Historical workload representation |

---

# 24. Design Decisions and Engineering Considerations

## 24.1 Prediction versus Oracle Knowledge

A practical predictive scheduler cannot normally know the exact future execution duration of a job.

OptiQ therefore distinguishes prediction-based scheduling from oracle algorithms.

SJF and SRTF provide idealized reference points, while ML-SJF and OptiQ represent prediction-assisted approaches.

## 24.2 Prediction Accuracy versus Scheduling Quality

A regression model can achieve low prediction error without necessarily producing the best scheduling outcome.

Scheduling quality depends on how prediction errors influence job ordering and resource allocation.

For example, a prediction error affecting a short job may have a different scheduling consequence from an error affecting a long job.

Therefore, OptiQ evaluates both:

- Regression model performance.
- End-to-end scheduling performance.

## 24.3 Fairness versus Efficiency

Minimizing average waiting time can favor shorter processes.

However, repeatedly favoring shorter processes can disadvantage longer jobs.

OptiQ explores priority aging and starvation override as mechanisms for balancing these competing considerations.

## 24.4 Reproducibility

Experiments use configurable parameters and seeded synthetic workloads.

Timestamped output directories help separate experiment runs and retain their generated artifacts.

## 24.5 Model Persistence

Model artifacts and preprocessing scalers are stored using Joblib.

This preserves the fitted model state and associated preprocessing information for subsequent use.

## 24.6 Simulation versus Real Operating Systems

The simulator abstracts away many details of real operating systems.

It does not reproduce the complete behavior of a production kernel, hardware-level context switching, multicore execution, I/O blocking, or real CPU scheduling overhead.

The results should therefore be interpreted within the simulator's defined scope.

---

# 25. Limitations and Future Improvements

OptiQ provides an experimental environment for investigating predictive and adaptive scheduling. Several extensions could make the framework more representative of real-world scheduling systems.

### 25.1 Multicore Scheduling

The current simulator uses a simplified single-logical-CPU execution model.

A future extension could simulate multiple CPU cores and investigate load balancing, core affinity, and parallel job allocation.

### 25.2 More Realistic Process Behavior

Real operating-system processes alternate between CPU-bound and I/O-bound execution.

Introducing CPU bursts, I/O waiting periods, and repeated ready/block transitions would make the simulator more representative of general-purpose operating systems.

### 25.3 Scheduling Overhead

The current context-switch metric counts process-to-process transitions but does not model actual context-switch duration.

A future version could assign configurable switching costs and include them in total simulated execution time.

### 25.4 Improved Prediction

Additional approaches could be investigated, including:

- Hyperparameter optimization.
- Additional ensemble methods.
- Workload-specific model selection.
- Online or incremental learning.
- Prediction uncertainty estimation.

### 25.5 Dynamic Workload Adaptation

A future scheduling policy could adjust its behavior according to workload characteristics observed during execution.

For example, it could adapt to changes in the proportion of short and long jobs.

### 25.6 More Extensive Fairness Analysis

Further experiments could investigate alternative fairness metrics, including measures based on CPU service received rather than waiting-time equality alone.

### 25.7 Larger-Scale Evaluation

Additional workload traces and larger experimental configurations could help establish whether observed results generalize across different workload distributions.

---

# 26. Author

**Developed by Asmin Sinha**

B.Tech — Computer Science and Engineering (Artificial Intelligence and Machine Learning)  
Vellore Institute of Technology, Vellore

GitHub: [@asminsinha](https://github.com/asminsinha)

---

<p align="center">
  <strong>OptiQ — Predictive Scheduling with Adaptive Fairness.</strong>
</p>
