// OptiQ Interactive Dashboard JavaScript
let currentRunId = null;
let runsList = [];

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSimulationForm();
  checkBackendHealth();
  loadRunsList();

  document.getElementById('btnRefreshRun').addEventListener('click', () => {
    if (currentRunId) loadRunData(currentRunId);
  });

  document.getElementById('globalRunSelect').addEventListener('change', (e) => {
    currentRunId = e.target.value;
    loadRunData(currentRunId);
  });
});

function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const panes = document.querySelectorAll('.tab-pane');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetTab = item.getAttribute('data-tab');
      navItems.forEach(n => n.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      item.classList.add('active');
      const targetPane = document.getElementById(`tab-${targetTab}`);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

async function checkBackendHealth() {
  const statusEl = document.getElementById('connectionStatus');
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      statusEl.querySelector('.status-indicator').className = 'status-indicator live';
      statusEl.querySelector('.status-text').textContent = 'Backend Connected';
    } else {
      throw new Error();
    }
  } catch (e) {
    statusEl.querySelector('.status-indicator').className = 'status-indicator';
    statusEl.querySelector('.status-text').textContent = 'Backend Offline';
  }
}

async function loadRunsList() {
  try {
    const res = await fetch('/api/runs');
    if (!res.ok) throw new Error('Failed to load runs');
    runsList = await res.json();

    const select = document.getElementById('globalRunSelect');
    select.innerHTML = '';

    runsList.forEach((r, idx) => {
      const opt = document.createElement('option');
      opt.value = r.run_id;
      const tag = r.is_historical ? ' [Historical]' : '';
      opt.textContent = `${r.run_id}${tag}`;
      select.appendChild(opt);
    });

    if (runsList.length > 0) {
      currentRunId = runsList[0].run_id;
      select.value = currentRunId;
      loadRunData(currentRunId);
      renderHistoryTable();
    }
  } catch (err) {
    console.error(err);
  }
}

async function loadRunData(runId) {
  document.getElementById('activeRunBadge').textContent = runId;
  document.getElementById('btnDownloadComparison').href = `/api/runs/${runId}/file/algorithm_comparison.csv`;
  document.getElementById('btnDownloadTrace').href = `/api/runs/${runId}/file/optiq_scheduling_trace.csv`;

  // Update chart images
  const timestamp = new Date().getTime();
  document.getElementById('imgSchedComp').src = `/api/runs/${runId}/file/scheduler_comparison.png?t=${timestamp}`;
  document.getElementById('imgWaitDist').src = `/api/runs/${runId}/file/waiting_time_distribution.png?t=${timestamp}`;
  document.getElementById('imgGantt').src = `/api/runs/${runId}/file/optiq_gantt_chart.png?t=${timestamp}`;
  document.getElementById('imgActualVsPred').src = `/api/runs/${runId}/file/actual_vs_predicted.png?t=${timestamp}`;
  document.getElementById('imgMlComp').src = `/api/runs/${runId}/file/ml_model_comparison.png?t=${timestamp}`;
  document.getElementById('imgFeatImp').src = `/api/runs/${runId}/file/feature_importance.png?t=${timestamp}`;
  document.getElementById('imgAblationComp').src = `/api/runs/${runId}/file/ablation_comparison.png?t=${timestamp}`;

  try {
    const res = await fetch(`/api/runs/${runId}/summary`);
    if (!res.ok) throw new Error('Failed to fetch summary');
    const data = await res.json();

    renderOverview(data);
    renderComparisonTable(data.comparison);
    renderMlTable(data.ml_metrics);
    renderAblationTable(data.ablation);
    loadTrace(runId);
  } catch (err) {
    console.error(err);
  }
}

function renderOverview(data) {
  const comp = data.comparison || [];
  const meta = data.metadata || {};
  const params = meta.parameters || {};

  const optiqRow = comp.find(c => c.Scheduler && c.Scheduler.includes('OptiQ')) || {};
  const fcfsRow = comp.find(c => c.Scheduler === 'FCFS') || {};

  const optiqWait = optiqRow['Avg Waiting Time (s)'];
  const fcfsWait = fcfsRow['Avg Waiting Time (s)'];

  if (optiqWait !== undefined) {
    document.getElementById('kpiAvgWait').textContent = `${Number(optiqWait).toFixed(1)}s`;
    if (fcfsWait) {
      const imp = ((fcfsWait - optiqWait) / fcfsWait * 100).toFixed(1);
      document.getElementById('kpiAvgWaitSub').textContent = `${imp > 0 ? '−' + imp + '%' : '+' + Math.abs(imp) + '%'} vs FCFS (${Number(fcfsWait).toFixed(1)}s)`;
    }
  }

  if (optiqRow['Avg Turnaround Time (s)'] !== undefined) {
    document.getElementById('kpiAvgTat').textContent = `${Number(optiqRow['Avg Turnaround Time (s)']).toFixed(1)}s`;
    document.getElementById('kpiAvgTatSub').textContent = `Median: ${Number(optiqRow['Median Turnaround Time (s)']).toFixed(1)}s`;
  }

  if (optiqRow['Fairness Index'] !== undefined) {
    document.getElementById('kpiFairness').textContent = Number(optiqRow['Fairness Index']).toFixed(4);
  }

  document.getElementById('kpiBestModel').textContent = params.best_ml_model || 'RandomForest';

  // 1. Dynamic Run Highlights (up to 3 insights)
  const highlightsEl = document.getElementById('overviewHighlights');
  if (highlightsEl) {
    const highlights = [];

    // Insight 1: OptiQ vs FCFS wait difference
    if (optiqWait !== undefined && fcfsWait !== undefined) {
      const waitDiff = Number(fcfsWait) - Number(optiqWait);
      const pct = Math.abs((waitDiff / Number(fcfsWait)) * 100).toFixed(1);
      if (waitDiff > 0) {
        highlights.push(
          `<strong>Average Wait Time:</strong> OptiQ reduces average queue wait by <strong>${waitDiff.toFixed(1)}s (${pct}%)</strong> compared to baseline FCFS (${Number(fcfsWait).toFixed(1)}s vs ${Number(optiqWait).toFixed(1)}s).`
        );
      } else {
        highlights.push(
          `<strong>Average Wait Time:</strong> OptiQ averages <strong>${Number(optiqWait).toFixed(1)}s</strong> compared to FCFS at <strong>${Number(fcfsWait).toFixed(1)}s</strong> on this workload.`
        );
      }
    }

    // Insight 2: Fairness / Scheduling observation
    if (optiqRow['Fairness Index'] !== undefined) {
      const fairness = Number(optiqRow['Fairness Index']).toFixed(4);
      const rrRow = comp.find(c => c.Scheduler && c.Scheduler.startsWith('RoundRobin'));
      const switches = optiqRow['Context Switches'];
      if (rrRow && rrRow['Context Switches'] !== undefined) {
        const switchDiff = Number(rrRow['Context Switches']) - Number(switches);
        highlights.push(
          `<strong>Fairness & Preemption:</strong> Achieved a fairness index of <strong>${fairness}</strong> with <strong>${switches} context switches</strong> (${switchDiff > 0 ? switchDiff + ' fewer than Round Robin' : 'balanced against time slicing'}).`
        );
      } else {
        highlights.push(
          `<strong>Fairness & Preemption:</strong> Achieved a fairness index of <strong>${fairness}</strong> across completed jobs with <strong>${switches || 0} total context switches</strong>.`
        );
      }
    }

    // Insight 3: Model Selection Rationale
    const bestModel = params.best_ml_model || 'RandomForest';
    const mlMetrics = data.ml_metrics || [];
    const bestMl = mlMetrics.find(m => m.Model === bestModel);
    if (bestMl && bestMl['Val MAE (s)'] !== undefined) {
      highlights.push(
        `<strong>Predictor Selection:</strong> <strong>${bestModel}</strong> was selected as the top model with the lowest validation error (Validation MAE: <strong>${Number(bestMl['Val MAE (s)']).toFixed(1)}s</strong>).`
      );
    } else {
      highlights.push(
        `<strong>Predictor Selection:</strong> <strong>${bestModel}</strong> was selected automatically based on lowest validation-split error across all candidate regressors.`
      );
    }

    highlightsEl.innerHTML = highlights.map(h => `<li>${h}</li>`).join('');
  }

  // Overview details grid
  const ds = meta.dataset || {};
  const evalJobsCount = data.metadata && data.metadata.parameters ? Math.round((data.metadata.parameters.n_jobs || 500) * 0.15) : 75;
  const detailsHtml = `
    <div class="form-row">
      <div><strong>Data Source:</strong> ${ds.source_type || 'N/A'} (${ds.filepath_or_name || 'N/A'})</div>
      <div><strong>Total Workload Jobs:</strong> ${params.n_jobs || 'N/A'} (Train: 70%, Val: 15%, Test: 15%)</div>
    </div>
    <div class="form-row mt-2">
      <div><strong>Held-Out Test Set (Evaluated):</strong> N=${evalJobsCount} jobs</div>
      <div><strong>Random Seed:</strong> ${params.seed !== undefined ? params.seed : 'N/A'}</div>
    </div>
    <div class="form-row mt-2">
      <div><strong>Quantum (Q):</strong> ${params.quantum || 'N/A'}s</div>
      <div><strong>Aging Interval:</strong> ${params.aging_interval || 'N/A'}s</div>
    </div>
    <div class="form-row mt-2">
      <div><strong>Starvation Threshold:</strong> ${params.starvation_threshold || 'N/A'}s (wait time violation threshold)</div>
      <div><strong>Fairness Metric:</strong> Jain's Index on Normalized Slowdown</div>
    </div>
  `;
  document.getElementById('overviewDetails').innerHTML = detailsHtml;
}

function renderComparisonTable(records) {
  const tbody = document.querySelector('#comparisonTable tbody');
  tbody.innerHTML = '';
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="11" class="text-muted">No comparison data available.</td></tr>';
    return;
  }

  records.forEach(row => {
    const tr = document.createElement('tr');
    const isOracle = row.Type && row.Type.includes('Oracle');
    const badgeClass = isOracle ? 'badge-oracle' : 'badge-deployable';

    tr.innerHTML = `
      <td><strong>${row.Scheduler || ''}</strong></td>
      <td><span class="badge ${badgeClass}">${row.Type || 'Standard'}</span></td>
      <td>${Number(row['Avg Waiting Time (s)']).toFixed(2)}</td>
      <td>${Number(row['Median Waiting Time (s)']).toFixed(2)}</td>
      <td>${Number(row['Avg Turnaround Time (s)']).toFixed(2)}</td>
      <td>${Number(row['Median Turnaround Time (s)']).toFixed(2)}</td>
      <td>${Number(row['Avg Response Time (s)']).toFixed(2)}</td>
      <td>${row['Context Switches']}</td>
      <td>${row['Starvation Risk Count']}</td>
      <td>${Number(row['Fairness Index']).toFixed(4)}</td>
      <td>${Number(row['Throughput (jobs/s)']).toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderMlTable(records) {
  const tbody = document.querySelector('#mlMetricsTable tbody');
  tbody.innerHTML = '';
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-muted">No ML metrics available.</td></tr>';
    return;
  }

  records.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${row.Model || ''}</strong></td>
      <td>${Number(row['Val MAE (s)']).toFixed(2)}</td>
      <td>${Number(row['Test MAE (s)']).toFixed(2)}</td>
      <td>${Number(row['Test RMSE (s)']).toFixed(2)}</td>
      <td>${Number(row['Test R2']).toFixed(4)}</td>
      <td>${Number(row['Inference (ms)']).toFixed(2)}</td>
      <td>${row['Train Samples']}</td>
      <td>${row['Test Samples']}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAblationTable(records) {
  const tbody = document.querySelector('#ablationTable tbody');
  const insightsEl = document.getElementById('ablationInsightsList');
  tbody.innerHTML = '';
  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" class="text-muted">No ablation data available.</td></tr>';
    if (insightsEl) {
      insightsEl.innerHTML = '<li>No ablation data available for this run.</li>';
    }
    return;
  }

  records.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${row.Policy || ''}</strong></td>
      <td>${Number(row['Avg Waiting Time (s)']).toFixed(2)}</td>
      <td>${Number(row['Median Waiting Time (s)']).toFixed(2)}</td>
      <td>${Number(row['Avg Turnaround Time (s)']).toFixed(2)}</td>
      <td>${Number(row['Median Turnaround Time (s)']).toFixed(2)}</td>
      <td>${row['Context Switches']}</td>
      <td>${row['Starvation Risk Count']}</td>
      <td>${Number(row['Fairness Index']).toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });

  // Populate dynamic Ablation Insights
  if (insightsEl) {
    const insights = [];
    const baseP = records.find(r => r.Policy && r.Policy.includes('Baseline'));
    const predOnly = records.find(r => r.Policy && r.Policy.includes('PredictionOnly'));
    const agingOnly = records.find(r => r.Policy && r.Policy.includes('AgingOnly'));
    const hybrid = records.find(r => r.Policy && r.Policy.includes('Hybrid'));

    if (hybrid) {
      // 1. Compare with PredictionOnly
      if (predOnly) {
        const waitDiff = Number(hybrid['Avg Waiting Time (s)']) - Number(predOnly['Avg Waiting Time (s)']);
        const fairnessDiff = Number(hybrid['Fairness Index']) - Number(predOnly['Fairness Index']);
        if (Math.abs(waitDiff) < 1.0 && Math.abs(fairnessDiff) < 0.005) {
          insights.push(
            `<strong>Prediction Influence:</strong> <strong>Hybrid</strong> closely tracks <strong>PredictionOnly</strong> on this workload (Avg Wait: <strong>${Number(hybrid['Avg Waiting Time (s)']).toFixed(1)}s</strong> vs <strong>${Number(predOnly['Avg Waiting Time (s)']).toFixed(1)}s</strong>), indicating queue wait times rarely exceeded the starvation threshold under this workload profile.`
          );
        } else if (waitDiff > 0) {
          insights.push(
            `<strong>Trade-Off vs Pure Prediction:</strong> <strong>PredictionOnly</strong> achieves lower average wait (<strong>${Number(predOnly['Avg Waiting Time (s)']).toFixed(1)}s</strong> vs <strong>${Number(hybrid['Avg Waiting Time (s)']).toFixed(1)}s</strong>), while <strong>Hybrid</strong> trades marginal mean delay to enforce starvation overrides and guarantee progress for delayed jobs.`
          );
        } else {
          insights.push(
            `<strong>Prediction & Aging Synergy:</strong> <strong>Hybrid</strong> achieves lower average wait than <strong>PredictionOnly</strong> (<strong>${Number(hybrid['Avg Waiting Time (s)']).toFixed(1)}s</strong> vs <strong>${Number(predOnly['Avg Waiting Time (s)']).toFixed(1)}s</strong>), showing preemption and aging prevented priority inversion.`
          );
        }
      }

      // 2. Compare with AgingOnly
      if (agingOnly) {
        const waitDiff = Number(agingOnly['Avg Waiting Time (s)']) - Number(hybrid['Avg Waiting Time (s)']);
        if (waitDiff > 0) {
          insights.push(
            `<strong>Value of Burst Prediction:</strong> Incorporating predicted burst time reduces average waiting time by <strong>${waitDiff.toFixed(1)}s</strong> compared to <strong>AgingOnly</strong> (${Number(hybrid['Avg Waiting Time (s)']).toFixed(1)}s vs ${Number(agingOnly['Avg Waiting Time (s)']).toFixed(1)}s).`
          );
        } else {
          insights.push(
            `<strong>Aging vs Hybrid:</strong> <strong>AgingOnly</strong> yields <strong>${Number(agingOnly['Avg Waiting Time (s)']).toFixed(1)}s</strong> average wait compared to <strong>${Number(hybrid['Avg Waiting Time (s)']).toFixed(1)}s</strong> for <strong>Hybrid</strong> on this workload.`
          );
        }
      }

      // 3. Compare with Baseline
      if (baseP) {
        const switchDiff = Number(baseP['Context Switches']) - Number(hybrid['Context Switches']);
        const waitDiff = Number(baseP['Avg Waiting Time (s)']) - Number(hybrid['Avg Waiting Time (s)']);
        if (switchDiff > 0) {
          insights.push(
            `<strong>Context Switch Overhead:</strong> The refined policy eliminates aggressive preemption chatter, reducing context switches by <strong>${switchDiff}</strong> compared to <strong>Baseline</strong> (${hybrid['Context Switches']} vs ${baseP['Context Switches']}).`
          );
        } else {
          insights.push(
            `<strong>Baseline Comparison:</strong> <strong>Hybrid</strong> produces <strong>${hybrid['Context Switches']}</strong> context switches compared to <strong>${baseP['Context Switches']}</strong> in the uncalibrated Baseline policy.`
          );
        }
      }
    }

    if (insights.length === 0) {
      insights.push(`<strong>Policy Summary:</strong> Evaluated ${records.length} policy configurations under identical arrivals and burst distributions.`);
    }

    insightsEl.innerHTML = insights.map(i => `<li>${i}</li>`).join('');
  }
}

async function loadTrace(runId) {
  const tbody = document.querySelector('#traceTable tbody');
  tbody.innerHTML = '<tr><td colspan="4" class="text-muted">Loading trace...</td></tr>';
  try {
    const res = await fetch(`/api/runs/${runId}/trace?limit=50`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    tbody.innerHTML = '';
    (data.trace || []).forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${Number(row.timestamp).toFixed(1)}</td>
        <td><code>PID ${row.chosen_pid}</code></td>
        <td>${row.reason || ''}</td>
        <td>${row.ready_count}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-muted">Trace not available.</td></tr>';
  }
}

function renderHistoryTable() {
  const tbody = document.querySelector('#historyTable tbody');
  tbody.innerHTML = '';

  runsList.forEach(r => {
    const tr = document.createElement('tr');
    const p = r.parameters || {};
    const d = r.dataset || {};
    const badge = r.is_historical ? '<span class="badge badge-historical">Historical</span>' : '<span class="badge badge-deployable">Live</span>';

    tr.innerHTML = `
      <td><code>${r.run_id}</code></td>
      <td>${badge}</td>
      <td>${r.timestamp ? r.timestamp.replace('T', ' ').substring(0, 19) : '—'}</td>
      <td>${d.source_type || '—'}</td>
      <td>${p.n_jobs || '—'}</td>
      <td>${p.quantum ? p.quantum + 's' : '—'}</td>
      <td>${p.aging_interval ? p.aging_interval + 's' : '—'}</td>
      <td>${p.starvation_threshold ? p.starvation_threshold + 's' : '—'}</td>
      <td><button class="btn btn-secondary btn-sm" onclick="selectRun('${r.run_id}')">Inspect</button></td>
    `;
    tbody.appendChild(tr);
  });
}

window.selectRun = function(runId) {
  currentRunId = runId;
  document.getElementById('globalRunSelect').value = runId;
  loadRunData(runId);
  // Switch to overview tab
  document.querySelector('.nav-item[data-tab="overview"]').click();
};

function initSimulationForm() {
  const form = document.getElementById('simForm');
  const statusBox = document.getElementById('simStatusBox');
  const btn = document.getElementById('btnRunSim');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    btn.disabled = true;
    btn.textContent = 'Executing Simulation...';
    statusBox.className = 'status-box info';
    statusBox.classList.remove('d-none');
    statusBox.textContent = 'Queuing and executing backend CPU simulation...';

    const payload = {
      source: document.getElementById('simSource').value,
      profile: document.getElementById('simProfile').value,
      jobs: parseInt(document.getElementById('simJobs').value),
      seed: parseInt(document.getElementById('simSeed').value),
      quantum: parseFloat(document.getElementById('simQuantum').value),
      aging: parseFloat(document.getElementById('simAging').value),
      starvation: parseFloat(document.getElementById('simStarvation').value),
    };

    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error('Simulation launch failed');
      const data = await res.json();
      const jobId = data.job_id;

      // Poll job status
      const pollInterval = setInterval(async () => {
        try {
          const pollRes = await fetch(`/api/jobs/${jobId}`);
          if (!pollRes.ok) return;
          const pollData = await pollRes.json();
          if (pollData.status === 'completed') {
            clearInterval(pollInterval);
            btn.disabled = false;
            btn.textContent = 'Launch Simulation Run';
            statusBox.className = 'status-box success';
            statusBox.textContent = `Simulation Completed! Created run: ${pollData.run_id}`;
            await loadRunsList();
            window.selectRun(pollData.run_id);
          } else if (pollData.status === 'failed') {
            clearInterval(pollInterval);
            btn.disabled = false;
            btn.textContent = 'Launch Simulation Run';
            statusBox.className = 'status-box error';
            statusBox.textContent = `Simulation Failed: ${pollData.error}`;
          }
        } catch (e) {
          // Keep polling
        }
      }, 1500);

    } catch (err) {
      btn.disabled = false;
      btn.textContent = 'Launch Simulation Run';
      statusBox.className = 'status-box error';
      statusBox.textContent = `Error: ${err.message}`;
    }
  });
}
