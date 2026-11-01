"""OptiQ: Adaptive Machine Learning-Based Hybrid CPU Scheduler.

Mathematical Formulation & Decision Precedence:
1. State at current simulation time t:
   - Ready queue: R(t) = { p | p.arrival_time <= t and p.remaining_time > 0 }
   - Ready queue wait: W_p(t) = t - p.arrival_time - p.time_executed
   - Effective priority with bounded aging:
     P_eff(p) = min(P_max, p.base_priority + floor(W_p(t) / A))
     where A = aging_interval, P_max = 25 (bounded ceiling).
   - Predicted remaining service time:
     R_hat(p) = max(0.1, p.predicted_burst - p.time_executed)
     Strict rule: Uses ONLY pre-scheduling ML burst prediction and simulated service already received.

2. Decision Precedence at Scheduling Event:
   - Starvation Override:
     Let S(t) = { p in R(t) | W_p(t) >= StarvationThreshold }
     If S(t) is non-empty:
       Choose p* = argmin_{p in S(t)} ( R_hat(p), -P_eff(p), p.arrival_time, p.pid )
       Reason: "Starvation override"
     Else:
       Choose p* = argmin_{p in R(t)} ( R_hat(p), -P_eff(p), p.arrival_time, p.pid )
       Reason: "Shortest predicted remaining"

3. Execution Interval & Context Switch Minimization:
   - Slice duration:
     d = min(Quantum, p*.remaining_time)
   - Preemption on significantly shorter arrival:
     If a new process arrives at t_arr in (t, t + d) with predicted_burst < 0.5 * p*.predicted_remaining_time:
       duration is reduced to t_arr - t to allow preemption.
   - Consecutive execution slices by the same PID are consolidated into a continuous slice.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

from optiq.schedulers.base import BaseScheduler
from optiq.simulator.process import ExecutionSlice, Process, ScheduleDecisionLog
from optiq.simulator.metrics import SchedulerBenchmarkSummary, compute_metrics


class OptiQScheduler(BaseScheduler):
    """OptiQ Adaptive Hybrid CPU Scheduler with Burst Prediction and Priority Aging."""

    def __init__(
        self,
        quantum: float = 30.0,
        aging_interval: float = 60.0,
        starvation_threshold: float = 1800.0,
        max_priority_ceiling: int = 25,
        preemption_threshold_ratio: float = 0.5,
    ):
        super().__init__(
            name="OptiQ (Hybrid ML+Aging)",
            is_oracle=False,
            notes=(
                f"Adaptive scheduler with ML predicted burst, quantum={quantum}s, "
                f"aging_interval={aging_interval}s, starvation_threshold={starvation_threshold}s."
            ),
        )
        self.quantum = quantum
        self.aging_interval = aging_interval
        self.starvation_threshold = starvation_threshold
        self.max_priority_ceiling = max_priority_ceiling
        self.preemption_threshold_ratio = preemption_threshold_ratio

    def run(
        self,
        processes: List[Process],
        starvation_threshold: Optional[float] = None,
    ) -> Tuple[SchedulerBenchmarkSummary, pd.DataFrame]:
        thresh = starvation_threshold if starvation_threshold is not None else self.starvation_threshold
        procs = self._clone_processes(processes)
        self.timeline.clear()
        self.decision_logs.clear()

        arrival_pool = sorted(procs, key=lambda p: (p.arrival_time, p.pid))
        ready_queue: List[Process] = []
        completed: List[Process] = []
        current_time = 0.0

        def update_ready_queue_state(curr_t: float):
            nonlocal arrival_pool, ready_queue
            # 1. Admit newly arrived processes
            newly_arrived = [p for p in arrival_pool if p.arrival_time <= curr_t]
            for p in newly_arrived:
                p.last_queued_time = p.arrival_time
                ready_queue.append(p)
                arrival_pool.remove(p)

            # 2. Update ready queue wait times and aging for waiting processes
            for p in ready_queue:
                p.update_wait(curr_t)
                if self.aging_interval > 0:
                    aging_boost = int(p.ready_queue_wait // self.aging_interval)
                    p.effective_priority = min(self.max_priority_ceiling, p.base_priority + aging_boost)
                else:
                    p.effective_priority = p.base_priority

                # Predicted remaining time computed strictly from predicted burst and simulated service received
                p.predicted_remaining_time = max(0.1, round(p.predicted_burst - p.time_executed, 4))
          
        while arrival_pool or ready_queue:
            update_ready_queue_state(current_time)

            if not ready_queue:
                current_time = arrival_pool[0].arrival_time
                update_ready_queue_state(current_time)

            # Snapshot for logging
            p_priorities = {p.pid: p.effective_priority for p in ready_queue}
            p_pred_rem = {p.pid: p.predicted_remaining_time for p in ready_queue}
            p_act_rem = {p.pid: p.remaining_time for p in ready_queue}

            # SELECTION LOGIC
            # Starvation override should only be considered an override if there is contention (len(ready_queue) > 1)
            starved_candidates = [p for p in ready_queue if p.ready_queue_wait >= thresh]

            if len(ready_queue) > 1 and starved_candidates:
                # Starvation override: among starved processes, pick highest effective priority, then shortest predicted remaining
                starved_candidates.sort(
                    key=lambda p: (-p.effective_priority, p.predicted_remaining_time, p.arrival_time, p.pid)
                )
                chosen = starved_candidates[0]
                reason = f"Starvation override (wait={chosen.ready_queue_wait:.1f}s >= {thresh}s, eff_prio={chosen.effective_priority})"
            else:
                # Shortest predicted remaining: shortest predicted remaining, highest priority, arrival, pid
                ready_queue.sort(
                    key=lambda p: (p.predicted_remaining_time, -p.effective_priority, p.arrival_time, p.pid)
                )
                chosen = ready_queue[0]
                if len(ready_queue) == 1:
                    reason = f"Lone ready process (rem={chosen.predicted_remaining_time:.1f}s, wait={chosen.ready_queue_wait:.1f}s)"
                else:
                    reason = f"Shortest predicted remaining ({chosen.predicted_remaining_time:.1f}s, eff_prio={chosen.effective_priority})"

            ready_queue.remove(chosen)
            # Reset queue wait upon receiving CPU service
            chosen.on_service(current_time)

            # Log decision
            self.decision_logs.append(
                ScheduleDecisionLog(
                    timestamp=current_time,
                    scheduler_name=self.name,
                    ready_queue_pids=[p.pid for p in ready_queue] + [chosen.pid],
                    chosen_pid=chosen.pid,
                    reason=reason,
                    effective_priorities=p_priorities,
                    predicted_remaining_times=p_pred_rem,
                    actual_remaining_times=p_act_rem,
                )
            )

            if chosen.start_time is None:
                chosen.start_time = current_time

            duration = min(self.quantum, chosen.remaining_time)

            # Preemption on significantly shorter arrival
            if self.preemption_threshold_ratio > 0 and arrival_pool:
                next_arr_time = arrival_pool[0].arrival_time
                time_to_next = next_arr_time - current_time
                if 0.0 < time_to_next < duration:
                    if arrival_pool[0].predicted_burst < (self.preemption_threshold_ratio * chosen.predicted_remaining_time):
                        duration = time_to_next

            duration = max(0.001, round(duration, 4))
            start_t = current_time
            end_t = round(start_t + duration, 4)

            chosen.remaining_time = round(chosen.remaining_time - duration, 4)
            chosen.time_executed = round(chosen.time_executed + duration, 4)
            current_time = end_t

            # Consolidate consecutive slices by same process into a continuous slice
            if self.timeline and self.timeline[-1].pid == chosen.pid:
                self.timeline[-1].end_time = end_t
                self.timeline[-1].duration = round(end_t - self.timeline[-1].start_time, 4)
            else:
                self.timeline.append(
                    ExecutionSlice(
                        pid=chosen.pid,
                        start_time=start_t,
                        end_time=end_t,
                        duration=duration,
                        reason=reason,
                        effective_priority=chosen.effective_priority,
                        predicted_remaining=chosen.predicted_remaining_time,
                    )
                )

            update_ready_queue_state(current_time)

            if chosen.remaining_time <= 1e-6:
                chosen.remaining_time = 0.0
                chosen.completion_time = end_t
                assert chosen.completion_time >= chosen.arrival_time, f"Invariant violated: Process {chosen.pid} completed before arrival!"
                assert chosen.time_executed <= round(chosen.burst_time + 1e-4, 4), f"Invariant violated: Process {chosen.pid} executed more than burst time!"
                completed.append(chosen)
            else:
                chosen.last_queued_time = current_time
                ready_queue.append(chosen)

        # Invariant checks
        assert len(completed) == len(processes), f"Invariant violated: Expected {len(processes)} completed jobs, got {len(completed)}."
        for sl_idx in range(len(self.timeline) - 1):
            assert self.timeline[sl_idx].end_time <= self.timeline[sl_idx + 1].start_time + 1e-4, (
                f"Invariant violated: Execution slices overlap at {self.timeline[sl_idx].end_time} and {self.timeline[sl_idx+1].start_time}."
            )

        summary, df = compute_metrics(
            completed,
            self.timeline,
            self.name,
            starvation_threshold=thresh,
            is_oracle=self.is_oracle,
            notes=self.notes,
        )
        return summary, df
