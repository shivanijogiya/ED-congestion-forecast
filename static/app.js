/* ── ED Congestion Dashboard ─────────────────────────────────────────────── */

const API = '';   // same origin
let forecastChart = null;
let historyChart  = null;
let activeHospital = null;
let autoRefreshTimer = null;

const COLORS = {
  green: '#3fb950',
  amber: '#d29922',
  red:   '#f85149',
  blue:  '#58a6ff',
  purple:'#bc8cff',
  muted: '#8b949e',
};

/* ── Clock ───────────────────────────────────────────────────────────────── */
function tickClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toLocaleTimeString('en-GB', { hour12: false });
}
setInterval(tickClock, 1000);
tickClock();

/* ── Health bar ──────────────────────────────────────────────────────────── */
async function loadHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const d = await r.json();
    const mae = d.components?.model?.mae_rolling;
    if (mae !== undefined) document.getElementById('maeVal').textContent = mae.toFixed(3);
  } catch (_) {}
}

/* ── Hospital cards ──────────────────────────────────────────────────────── */
async function loadAllHospitals() {
  const grid = document.getElementById('hospitalGrid');
  grid.innerHTML = '<div class="spinner"></div>';

  try {
    const [topoRes, ...forecastRes] = await Promise.all([
      fetch(`${API}/hospitals`),
      ...['H1','H2','H3','H4','H5','H6'].map(id => fetch(`${API}/forecast/${id}`))
    ]);

    const topo      = await topoRes.json();
    const forecasts = await Promise.all(forecastRes.map(r => r.json()));
    const fMap      = Object.fromEntries(forecasts.map(f => [f.hospital_id, f]));

    grid.innerHTML = '';
    topo.hospitals.forEach(h => {
      const f   = fMap[h.hospital_id] || {};
      const sev = f.hospital_severity || 'green';
      const pct = Math.round((f.hospital_max_congestion || 0) * 100);

      const card = document.createElement('div');
      card.className = `hospital-card ${sev}`;
      card.dataset.id = h.hospital_id;
      card.innerHTML = `
        <div class="card-top">
          <div>
            <div class="card-name">${h.hospital_name}</div>
            <div class="card-id">${h.hospital_id} · ${h.departments.length} departments</div>
          </div>
          <div class="severity-pill ${sev}">${sev}</div>
        </div>
        <div class="congestion-row">
          <div class="congestion-bar-wrap">
            <div class="congestion-bar ${sev}" style="width:${pct}%"></div>
          </div>
          <div class="congestion-pct" style="color:var(--${sev})">${pct}%</div>
        </div>
        <div class="card-depts">Peak: ${peakDept(f)} · Next 4h forecast</div>
      `;
      card.addEventListener('click', () => selectHospital(h.hospital_id, h.hospital_name));
      grid.appendChild(card);
    });

    // Re-select active hospital if any
    if (activeHospital) {
      document.querySelector(`[data-id="${activeHospital}"]`)?.classList.add('active');
    }
  } catch (e) {
    grid.innerHTML = `<p style="color:var(--red);font-size:13px;padding:10px">Failed to load hospitals.<br/>${e.message}</p>`;
  }
}

function peakDept(forecast) {
  if (!forecast.departments?.length) return '—';
  const peak = forecast.departments.reduce((a, b) =>
    a.max_congestion > b.max_congestion ? a : b);
  return `${peak.dept_name} (${Math.round(peak.max_congestion * 100)}%)`;
}

/* ── Select hospital ─────────────────────────────────────────────────────── */
async function selectHospital(id, name) {
  activeHospital = id;
  document.querySelectorAll('.hospital-card').forEach(c => c.classList.remove('active'));
  document.querySelector(`[data-id="${id}"]`)?.classList.add('active');

  document.getElementById('placeholder').classList.add('hidden');
  document.getElementById('detailContent').classList.remove('hidden');
  document.getElementById('detailHospitalName').textContent = name;

  clearTimeout(autoRefreshTimer);
  await Promise.all([loadForecast(id), loadHistory(id)]);
  autoRefreshTimer = setTimeout(() => {
    if (activeHospital === id) selectHospital(id, name);
  }, 30000);  // auto-refresh every 30s
}

/* ── Forecast ────────────────────────────────────────────────────────────── */
async function loadForecast(hospitalId) {
  try {
    const r = await fetch(`${API}/forecast/${hospitalId}`);
    const d = await r.json();

    // Severity badge
    const badge = document.getElementById('detailBadge');
    badge.textContent = d.hospital_severity?.toUpperCase();
    badge.className = `severity-badge ${d.hospital_severity}`;

    // Context cards
    const first = d.departments?.[0]?.contributing_factors || {};
    setContextCard('ctxWeather', first.weather_score,  '%', v => Math.round(v*100));
    setContextCard('ctxFlu',     first.flu_index,       '',  v => v.toFixed(1));
    setContextCard('ctxTraffic', first.traffic_score,  '%', v => Math.round(v*100));
    document.getElementById('ctxPeak').textContent =
      Math.round((d.hospital_max_congestion || 0) * 100) + '%';
    colorContextCard('ctx-max', d.hospital_severity);

    // Department grid
    renderDeptGrid(d.departments || []);

    // Forecast chart
    renderForecastChart(d.departments || []);
  } catch (e) {
    console.error('Forecast error:', e);
  }
}

function setContextCard(elId, val, suffix, fmt) {
  document.getElementById(elId).textContent =
    val != null ? fmt(val) + suffix : '--';
}
function colorContextCard(elId, sev) {
  const el = document.getElementById(elId);
  el.style.borderColor = sev === 'red' ? 'rgba(248,81,73,0.4)'
    : sev === 'amber' ? 'rgba(210,153,34,0.4)' : 'transparent';
}

/* ── Department grid ─────────────────────────────────────────────────────── */
function renderDeptGrid(departments) {
  const grid = document.getElementById('deptGrid');
  grid.innerHTML = '';
  departments.forEach(dept => {
    const sev = dept.severity_label;
    const pct = Math.round((dept.max_congestion || 0) * 100);
    const card = document.createElement('div');
    card.className = `dept-card ${sev}`;
    card.innerHTML = `
      <div class="dept-name">${dept.dept_name}</div>
      <div class="dept-type">${dept.dept_type}</div>
      <div class="horizon-grid">
        ${['1h','2h','4h','8h'].map(h => {
          const v = dept.forecasts?.[h] ?? 0;
          const s = severity(v);
          return `<div class="horizon-item">
            <div class="horizon-label">+${h}</div>
            <div class="horizon-val ${s}">${Math.round(v*100)}%</div>
          </div>`;
        }).join('')}
      </div>
      <div class="dept-bar-wrap">
        <div class="dept-bar ${sev}" style="width:${pct}%"></div>
      </div>
    `;
    grid.appendChild(card);
  });
}

function severity(v) {
  return v >= 0.8 ? 'red' : v >= 0.6 ? 'amber' : 'green';
}

/* ── Forecast Chart ──────────────────────────────────────────────────────── */
function renderForecastChart(departments) {
  const labels   = departments.map(d => d.dept_name.replace(' ', '\n'));
  const horizons = ['1h', '2h', '4h', '8h'];
  const palette  = ['#58a6ff', '#bc8cff', '#d29922', '#f85149'];

  const datasets = horizons.map((h, i) => ({
    label: `+${h}`,
    data: departments.map(d => Math.round((d.forecasts?.[h] ?? 0) * 100)),
    backgroundColor: palette[i] + '33',
    borderColor: palette[i],
    borderWidth: 2,
    borderRadius: 4,
  }));

  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(document.getElementById('forecastChart'), {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8b949e', font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw}%` } },
      },
      scales: {
        x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
        y: {
          ticks: { color: '#8b949e', callback: v => v + '%' },
          grid: { color: '#21262d' },
          min: 0, max: 100,
        },
      },
      animation: { duration: 600 },
    }
  });

  // Draw threshold line at 80%
  const thresholdPlugin = {
    id: 'threshold',
    afterDraw(chart) {
      const { ctx, chartArea, scales } = chart;
      const y = scales.y.getPixelForValue(80);
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = 'rgba(248,81,73,0.5)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(chartArea.left, y);
      ctx.lineTo(chartArea.right, y);
      ctx.stroke();
      ctx.fillStyle = 'rgba(248,81,73,0.7)';
      ctx.font = '10px sans-serif';
      ctx.fillText('80% threshold', chartArea.right - 90, y - 4);
      ctx.restore();
    }
  };
  forecastChart.config.plugins = [thresholdPlugin];
  forecastChart.update();
}

/* ── History Chart ───────────────────────────────────────────────────────── */
async function loadHistory(hospitalId) {
  try {
    const r = await fetch(`${API}/history/${hospitalId}?hours=24`);
    const d = await r.json();
    renderHistoryChart(d.timeline || []);
  } catch (e) {
    console.error('History error:', e);
  }
}

function renderHistoryChart(timeline) {
  if (!timeline.length) return;
  const labels = timeline.map(e => {
    const t = new Date(e.timestamp);
    return t.getHours().toString().padStart(2, '0') + ':00';
  });

  // Get first few department IDs
  const deptIds = Object.keys(timeline[0]?.departments || {}).slice(0, 4);
  const palette = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff'];

  const deptNames = {
    '_TRIAGE': 'Triage', '_RESUS': 'Resus', '_FASTTRACK': 'Fast Track',
    '_OBS': 'Obs', '_RADIOLOGY': 'Radiology', '_BOARDING': 'Boarding', '_DISCHARGE': 'Discharge'
  };

  const datasets = deptIds.map((id, i) => {
    const suffix = Object.keys(deptNames).find(k => id.endsWith(k)) || id;
    const label  = deptNames[suffix] || id.split('_').pop();
    return {
      label,
      data: timeline.map(e => Math.round((e.departments?.[id] ?? 0) * 100)),
      borderColor: palette[i],
      backgroundColor: palette[i] + '15',
      fill: true,
      tension: 0.4,
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
    };
  });

  if (historyChart) historyChart.destroy();
  historyChart = new Chart(document.getElementById('historyChart'), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#8b949e', font: { size: 11 }, boxWidth: 12 } },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw}%` } },
      },
      scales: {
        x: {
          ticks: { color: '#8b949e', font: { size: 10 }, maxTicksLimit: 12 },
          grid: { color: '#21262d' },
        },
        y: {
          ticks: { color: '#8b949e', callback: v => v + '%' },
          grid: { color: '#21262d' },
          min: 0, max: 100,
        },
      },
      animation: { duration: 600 },
    }
  });
}

/* ── Init ─────────────────────────────────────────────────────────────────── */
loadAllHospitals();
loadHealth();
setInterval(loadHealth, 15000);
setInterval(() => {
  if (!activeHospital) loadAllHospitals();
}, 30000);
