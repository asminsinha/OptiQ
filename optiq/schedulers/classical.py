"""Standard Classical CPU Schedulers:
1. FCFS (First-Come, First-Served)
2. Non-preemptive SJF (Oracle: uses ground-truth burst)
3. Preemptive SRTF (Oracle: Shortest Remaining Time First)
4. Round Robin (RR with configurable time quantum)
5. Priority Scheduling (Non-preemptive, higher integer = higher priority)
6. ML-SJF (Non-preemptive SJF using predicted burst without aging/preemption)
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from optiq.schedulers.base import BaseScheduler
from optiq.simulator.process import ExecutionSlice, Process, ScheduleDecisionLog
from optiq.simulator.metrics import SchedulerBenchmarkSummary, compute_metrics


class FCFSScheduler(BaseScheduler):
    """First-Come, First-Served CPU Scheduler."""

    def __init__(self):
        super().__init__(name="FCFS", is_oracle=False, notes="Non-preemptive arrival order.")

    def run(
        self,
        processes: List[Process],
        starvation_threshold: float = 100.0,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        procs = self._clone_processes(processes)
        procs.sort(key=lambda p: (p.arrival_time, p.pid))
        self.timeline.clear()
        self.decision_logs.clear()

        current_time = 0.0

        for p in procs:
            if current_time < p.arrival_time:
                current_time = p.arrival_time

            start = current_time
            end = start + p.burst_time
            p.start_time = start
            p.completion_time = end
            p.remaining_time = 0.0
            p.time_executed = p.burst_time
            current_time = end

            self.timeline.append(
                ExecutionSlice(
                    pid=p.pid,
                    start_time=start,
                    end_time=end,
                    duration=p.burst_time,
                    reason="FCFS queue head",
                    effective_priority=p.effective_priority,
                    predicted_remaining=0.0,
                )
            )

        summary, df = compute_metrics(
            procs,
            self.timeline,
            self.name,
            starvation_threshold=starvation_threshold,
            is_oracle=self.is_oracle,
            notes=self.notes,
        )
        return summary, df


class SJFScheduler(BaseScheduler):
    """Non-Preemptive Shortest Job First (Oracle Baseline)."""

    def __init__(self):
        super().__init__(
            name="SJF (Oracle)",
            is_oracle=True,
            notes="Idealized reference knowing actual ground truth burst time.",
        )

    def run(
        self,
        processes: List[Process],
        starvation_threshold: float = 100.0,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        procs = self._clone_processes(processes)
        self.timeline.clear()
        self.decision_logs.clear()

        uncompleted = list(procs)
        completed: List[Process] = []
        current_time = 0.0

        while uncompleted:
            ready = [p for p in uncompleted if p.arrival_time <= current_time]
            if not ready:
                # Fast forward to next arrival
                current_time = min(p.arrival_time for p in uncompleted)
                continue

            # Pick process with smallest actual burst time (tie-break by arrival, then pid)
            ready.sort(key=lambda p: (p.burst_time, p.arrival_time, p.pid))
            chosen = ready[0]

            start = current_time
            end = start + chosen.burst_time
            chosen.start_time = start
            chosen.completion_time = end
            chosen.remaining_time = 0.0
            chosen.time_executed = chosen.burst_time
            current_time = end

            self.timeline.append(
                ExecutionSlice(
                    pid=chosen.pid,
                    start_time=start,
                    end_time=end,
                    duration=chosen.burst_time,
                    reason="SJF minimum actual burst",
                    effective_priority=chosen.effective_priority,
                    predicted_remaining=0.0,
                )
            )
            completed.append(chosen)
            uncompleted.remove(chosen)

        summary, df = compute_metrics(
            completed,
            self.timeline,
            self.name,
            starvation_threshold=starvation_threshold,
            is_oracle=self.is_oracle,
            notes=self.notes,
        )
        return summary, df


class MLSJFScheduler(BaseScheduler):
    """Non-Preemptive SJF using ML Predicted Burst (Ablation Baseline)."""

    def __init__(self):
        super().__init__(
            name="ML-SJF",
            is_oracle=False,
            notes="Ablation baseline: selects shortest predicted burst without aging or quantum.",
        )

    def run(
        self,
        processes: List[Process],
        starvation_threshold: float = 100.0,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        procs = self._clone_processes(processes)
        self.timeline.clear()
        self.decision_logs.clear()

        uncompleted = list(procs)
        completed: List[Process] = []
        current_time = 0.0

        while uncompleted:
            ready = [p for p in uncompleted if p.arrival_time <= current_time]
            if not ready:
                current_time = min(p.arrival_time for p in uncompleted)
                continue

            # Pick process with smallest predicted burst (tie-break by arrival, then pid)
            ready.sort(key=lambda p: (p.predicted_burst, p.arrival_time, p.pid))
            chosen = ready[0]

            start = current_time
            end = start + chosen.burst_time
            chosen.start_time = start
            chosen.completion_time = end
            chosen.remaining_time = 0.0
            chosen.time_executed = chosen.burst_time
            current_time = end

            self.timeline.append(
                ExecutionSlice(
                    pid=chosen.pid,
                    start_time=start,
                    end_time=end,
                    duration=chosen.burst_time,
                    reason="ML-SJF minimum predicted burst",
                    effective_priority=chosen.effective_priority,
                    predicted_remaining=0.0,
                )
            )
            completed.append(chosen)
            uncompleted.remove(chosen)

        summary, df = compute_metrics(
            completed,
            self.timeline,
            self.name,
            starvation_threshold=starvation_threshold,
            is_oracle=self.is_oracle,
            notes=self.notes,
        )
        return summary, df


class SRTFScheduler(BaseScheduler):
    """Preemptive Shortest Remaining Time First (Oracle Baseline)."""

    def __init__(self):
        super().__init__(
            name="SRTF (Oracle)",
            is_oracle=True,
            notes="Idealized preemptive reference knowing actual ground truth remaining time.",
        )

    def run(
        self,
        processes: List[Process],
        starvation_threshold: float = 100.0,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        procs = self._clone_processes(processes)
        self.timeline.clear()
        self.decision_logs.clear()

        uncompleted = list(procs)
        completed: List[Process] = []
        current_time = 0.0

        while uncompleted:
            ready = [p for p in uncompleted if p.arrival_time <= current_time]
            if not ready:
                current_time = min(p.arrival_time for p in uncompleted)
                continue

            # Pick shortest actual remaining time
            ready.sort(key=lambda p: (p.remaining_time, p.arrival_time, p.pid))
            chosen = ready[0]

            if chosen.start_time is None:
                chosen.start_time = current_time

            # Find when the next event could occur (either chosen completes or new arrival arrives)
            future_arrivals = [p.arrival_time for p in uncompleted if p.arrival_time > current_time]
            next_arrival = min(future_arrivals) if future_arrivals else float("inf")

            time_to_complete = chosen.remaining_time
            time_to_arrival = next_arrival - current_time

            duration = min(time_to_complete, time_to_arrival)
            duration = max(0.001, duration)

            start = current_time
            end = round(start + duration, 4)
            current_time = end

            chosen.remaining_time = round(chosen.remaining_time - duration, 4)
            chosen.time_executed = round(chosen.time_executed + duration, 4)

            # Record slice
            if self.timeline and self.timeline[-1].pid == chosen.pid:
                self.timeline[-1].end_time = end
                self.timeline[-1].duration = round(end - self.timeline[-1].start_time, 4)
            else:
                self.timeline.append(
                    ExecutionSlice(
                        pid=chosen.pid,
                        start_time=start,
                        end_time=end,
                        duration=round(duration, 4),
                        reason="SRTF shortest actual remaining",
                        effective_priority=chosen.effective_priority,
                        predicted_remaining=0.0,
                    )
                )

            if chosen.remaining_time <= 1e-6:
                chosen.remaining_time = 0.0
                chosen.completion_time = end
                completed.append(chosen)
                uncompleted.remove(chosen)

        summary, df = compute_metrics(
            completed,
            self.timeline,
            self.name,
            starvation_threshold=starvation_threshold,
            is_oracle=self.is_oracle,
            notes=self.notes,
        )
        return summary, df


class RoundRobinScheduler(BaseScheduler):
    """Round Robin CPU Scheduler with configurable time quantum."""

    def __init__(self, quantum: float = 10.0):
        super().__init__(
            name=f"RoundRobin (Q={quantum})",
            is_oracle=False,
            notes=f"Fair time slicing with quantum = {quantum}s.",
        )
        self.quantum = quantum

    def run(
        self,
        processes: List[Process],
        starvation_threshold: float = 100.0,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        procs = self._clone_processes(processes)
        self.timeline.clear()
        self.decision_logs.clear()

        # Arrival pool
        arrival_pool = sorted(procs, key=lambda p: (p.arrival_time, p.pid))
        ready_queue: List[Process] = []
        completed: List[Process] = []
        current_time = 0.0

        def check_arrivals(curr_t: float):
            nonlocal arrival_pool, ready_queue
            arrived = [p for p in arrival_pool if p.arrival_time <= curr_t]
            for p in arrived:
                ready_queue.append(p)
                arrival_pool.remove(p)

        while arrival_pool or ready_queue:
            check_arrivals(current_time)

            if not ready_queue:
                current_time = arrival_pool[0].arrival_time
                check_arrivals(current_time)

            chosen = ready_queue.pop(0)

            if chosen.start_time is None:
                chosen.start_time = current_time

            duration = min(self.quantum, chosen.remaining_time)
            duration = max(0.001, round(duration, 4))
            start = current_time
            end = round(start + duration, 4)

            chosen.remaining_time = round(chosen.remaining_time - duration, 4)
            chosen.time_executed = round(chosen.time_executed + duration, 4)
            current_time = end

            # Log execution slice (consolidate consecutive executions if single process in queue)
            if self.timeline and self.timeline[-1].pid == chosen.pid:
                self.timeline[-1].end_time = end
                self.timeline[-1].duration = round(end - self.timeline[-1].start_time, 4)
            else:
                self.timeline.append(
                    ExecutionSlice(
                        pid=chosen.pid,
                        start_time=start,
                        end_time=end,
                        duration=duration,
                        reason=f"RR quantum slice ({duration}s)",
                        effective_priority=chosen.effective_priority,
                        predicted_remaining=0.0,
                    )
                )

            # Check new arrivals before re-queueing chosen process
            check_arrivals(current_time)

            if chosen.remaining_time <= 1e-6:
                chosen.remaining_time = 0.0
                chosen.completion_time = end
                completed.append(chosen)
            else:
                ready_queue.append(chosen)

        summary, df = compute_metrics(
            completed,
            self.timeline,
            self.name,
            starvation_threshold=starvation_threshold,
            is_oracle=self.is_oracle,
            notes=self.notes,
        )
        return summary, df


class PriorityScheduler(BaseScheduler):
    """Non-Preemptive Static Priority Scheduler (higher integer = higher priority)."""

    def __init__(self):
        super().__init__(
            name="Priority (Static)",
            is_oracle=False,
            notes="Non-preemptive highest static priority first.",
        )

    def run(
        self,
        processes: List[Process],
        starvation_threshold: float = 100.0,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        procs = self._clone_processes(processes)
        self.timeline.clear()
        self.decision_logs.clear()

        uncompleted = list(procs)
        completed: List[Process] = []
        current_time = 0.0

        while uncompleted:
            ready = [p for p in uncompleted if p.arrival_time <= current_time]
            if not ready:
                current_time = min(p.arrival_time for p in uncompleted)
                continue

            # Higher base_priority first, tie-break by earlier arrival, then pid
            ready.sort(key=lambda p: (-p.base_priority, p.arrival_time, p.pid))
            chosen = ready[0]

            start = current_time
            end = start + chosen.burst_time
            chosen.start_time = start
            chosen.completion_time = end
            chosen.remaining_time = 0.0
            chosen.time_executed = chosen.burst_time
            current_time = end

            self.timeline.append(
                ExecutionSlice(
                    pid=chosen.pid,
                    start_time=start,
                    end_time=end,
                    duration=chosen.burst_time,
                    reason=f"Priority highest ({chosen.base_priority})",
                    effective_priority=chosen.effective_priority,
                    predicted_remaining=0.0,
                )
            )
            completed.append(chosen)
            uncompleted.remove(chosen)

        summary, df = compute_metrics(
            completed,
            self.timeline,
            self.name,
            starvation_threshold=starvation_threshold,
            is_oracle=self.is_oracle,
            notes=self.notes,
        )
        return summary, df
