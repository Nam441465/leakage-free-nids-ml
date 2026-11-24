let OVERVIEW = null;
let ovChart = null;
let noiseChart = null;
let lofoChart = null;
let streamTimer = null;
let streamStats = { total: 0, atk: 0, baseOk: 0, p4Ok: 0 };

const $ = (id) => document.getElementById(id);

const CLASS_LISTS = {
  cicids2017: ["ALL", "Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Web", "Bot", "Rare"],
  nf_unsw: ["ALL", "Benign", "Exploits", "Fuzzers", "DoS", "Generic", "Shellcode", "Reconnaissance", "Analysis", "Backdoor", "Rare"],
};

function fmtNum(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(d);
}

// Theme toggle
$("themeToggle").addEventListener("click", () => {
  const html = document.documentElement;
  const isDark = html.getAttribute("data-theme") === "dark";
  html.setAttribute("data-theme", isDark ? "light" : "dark");
  $("themeToggle").textContent = isDark ? "🌙 Chế độ Tối" : "☀ Chế độ Sáng";
});

// Tabs navigation
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "noise" && !noiseChart) renderNoiseFullChart();
    if (btn.dataset.tab === "lofo" && !lofoChart) renderLofoFullChart();
  });
});

// ==================== TAB 1: OVERVIEW ====================
async function initOverview() {
  const res = await fetch("/api/overview");
  OVERVIEW = await res.json();
  const k = OVERVIEW.kpis;
  $("kpi-cic-rows").textContent = k.cicids2017_rows.toLocaleString("en-US");
  $("kpi-nf-rows").textContent = k.nf_unsw_rows.toLocaleString("en-US");
  $("kpi-bundles").textContent = `${k.total_saved_files} files (${k.saved_baseline_files} Baseline + ${k.saved_phase4_files} Phase 4) • ${k.pytest_passed}/69 pytest`;

  renderOverviewTable();
  renderFigures("phase4");
  populateLiveClasses();
  // Auto-run 1 initial demo in Noise Lab & LOFO Lab on demand when user visits or right away
  runNoiseLab();
  runLofoLab();
}

function renderOverviewTable() {
  if (!OVERVIEW) return;
  const ds = $("ov-dataset").value;
  const task = $("ov-task").value;
  const paper = OVERVIEW.paper_reference[ds][task];
  const baseTable = OVERVIEW.baseline_tables[`${ds}_${task}`];
  const p4Task = OVERVIEW.phase4_tables.main.datasets[ds][task];

  const thead = $("ov-comp-table").querySelector("thead");
  const tbody = $("ov-comp-table").querySelector("tbody");
  tbody.innerHTML = "";

  if (task === "binary") {
    thead.innerHTML = `<tr>
      <th>Mô hình</th>
      <th>Bài báo Gốc — F1 (%)</th>
      <th>Tái tạo Baseline — F1 / FPR</th>
      <th>Phase 4 Cải tiến — F1 / FPR</th>
      <th>Mức tăng Δ F1 (Paired)</th>
      <th>Holm p</th>
    </tr>`;

    ["xgboost", "lightgbm", "random_forest"].forEach((m) => {
      const bm = baseTable.models[m];
      const p4m = p4Task.variants[`phase4_best_${m}`];
      const pairedList = p4Task.paired_vs_baseline?.f1 || [];
      const paired = pairedList.find((item) => item.model_a === `phase4_best_${m}`);
      const diff = paired ? paired.mean_diff : (p4m.f1.mean - bm.f1.mean);
      const pholm = paired ? paired.p_adjusted : null;
      const diffCls = diff >= 0 ? "badge-ok" : "badge-warn";
      tbody.innerHTML += `<tr>
        <td><strong>${m.toUpperCase()}</strong></td>
        <td>${paper[m].f1} <span class="muted">(FPR ${paper[m]["fpr_at_0.5"]}%)</span></td>
        <td>${bm.f1.formatted}% <span class="muted">/ FPR ${bm["fpr_at_0.5"].formatted}%</span></td>
        <td><strong>${p4m.f1.formatted}%</strong> <span class="badge badge-info">FPR ${p4m["fpr_at_0.5"].formatted}%</span></td>
        <td><span class="badge ${diffCls}">${diff >= 0 ? "+" : ""}${fmtNum(diff, 2)} pp</span></td>
        <td>${pholm !== null ? fmtNum(pholm, 4) : "—"}</td>
      </tr>`;
    });
    tbody.innerHTML += `<tr style="opacity:0.8">
      <td><em>Tham chiếu: R-GCN (Bài báo)</em></td>
      <td>${paper.r_gcn_ref.f1} <span class="muted">(FPR ${paper.r_gcn_ref["fpr_at_0.5"]}%)</span></td>
      <td colspan="4" class="muted">Mô hình đồ thị trong bài báo gốc — chỉ dùng để đối chiếu, không triển khai GNN theo quy tắc dự án</td>
    </tr>`;
  } else {
    thead.innerHTML = `<tr>
      <th>Mô hình</th>
      <th>Bài báo Gốc — Macro-F1 (%)</th>
      <th>Tái tạo Baseline — Macro-F1 / W-F1</th>
      <th>Phase 4 Cải tiến — Macro-F1 / W-F1</th>
      <th>Mức tăng Δ Macro-F1</th>
      <th>Điểm nhấn lớp thiểu số</th>
    </tr>`;

    ["xgboost", "lightgbm", "random_forest"].forEach((m) => {
      const bm = baseTable.models[m];
      const p4m = p4Task.variants[`phase4_best_${m}`];
      const diff = p4m.macro_f1.mean - bm.macro_f1.mean;
      let note = "Cải thiện đều các lớp hiếm";
      if (ds === "cicids2017" && bm.per_class_f1?.Bot && p4m.per_class_f1?.Bot) {
        note = `Bot F1: ${fmtNum(bm.per_class_f1.Bot.mean, 1)}% → <strong>${fmtNum(p4m.per_class_f1.Bot.mean, 1)}%</strong>`;
      } else if (ds === "nf_unsw" && m === "xgboost") {
        const ratioM = p4Task.variants.xgboost_ratio_calibrated;
        note = `Bản Ratio đạt <strong>${ratioM.macro_f1.formatted}%</strong>`;
      }
      tbody.innerHTML += `<tr>
        <td><strong>${m.toUpperCase()}</strong></td>
        <td>${paper[m].macro_f1}</td>
        <td>${bm.macro_f1.formatted}% <span class="muted">/ ${bm.weighted_f1.formatted}%</span></td>
        <td><strong>${p4m.macro_f1.formatted}%</strong> <span class="muted">/ ${p4m.weighted_f1.formatted}%</span></td>
        <td><span class="badge badge-ok">+${fmtNum(diff, 2)} pp</span></td>
        <td>${note}</td>
      </tr>`;
    });
    tbody.innerHTML += `<tr style="opacity:0.8">
      <td><em>Tham chiếu: R-GCN (Bài báo)</em></td>
      <td>${paper.r_gcn_ref.macro_f1}</td>
      <td colspan="4" class="muted">XGBoost Phase 4 vượt R-GCN bài báo +${fmtNum(p4Task.variants.phase4_best_xgboost.macro_f1.mean - parseFloat(paper.r_gcn_ref.macro_f1), 2)} pp</td>
    </tr>`;
  }

  renderOverviewChart(ds, task);
}

function renderOverviewChart(ds, task) {
  const ctx = $("ovPerClassChart").getContext("2d");
  if (ovChart) ovChart.destroy();

  if (task === "multiclass") {
    $("ov-chart-title").textContent = `So sánh F1 từng lớp (${ds.toUpperCase()}) — Baseline vs. Phase 4 (XGBoost)`;
    const basePc = OVERVIEW.baseline_tables[`${ds}_multiclass`].models.xgboost.per_class_f1;
    const p4Pc = OVERVIEW.phase4_tables.main.datasets[ds].multiclass.variants.phase4_best_xgboost.per_class_f1;
    const labels = CLASS_LISTS[ds].slice(1).filter((c) => basePc[c]);
    ovChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Baseline Bài báo (XGBoost)", data: labels.map((c) => basePc[c].mean), backgroundColor: "rgba(148, 163, 184, 0.65)", borderRadius: 5 },
          { label: "Phase 4 Cải tiến (XGBoost)", data: labels.map((c) => p4Pc[c].mean), backgroundColor: "rgba(16, 185, 129, 0.85)", borderRadius: 5 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { y: { min: 0, max: 105, title: { display: true, text: "F1 (%)" } } },
      },
    });
  } else {
    $("ov-chart-title").textContent = `So sánh Binary F1 (${ds.toUpperCase()}) — Baseline vs. Phase 4`;
    const baseM = OVERVIEW.baseline_tables[`${ds}_binary`].models;
    const p4M = OVERVIEW.phase4_tables.main.datasets[ds].binary.variants;
    const labels = ["XGBoost", "LightGBM", "Random Forest"];
    const keys = ["xgboost", "lightgbm", "random_forest"];
    ovChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Baseline F1 (%)", data: keys.map((k) => baseM[k].f1.mean), backgroundColor: "rgba(148, 163, 184, 0.65)", borderRadius: 5 },
          { label: "Phase 4 F1 (%)", data: keys.map((k) => p4M[`phase4_best_${k}`].f1.mean), backgroundColor: "rgba(59, 130, 246, 0.85)", borderRadius: 5 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { y: { min: ds === "nf_unsw" ? 88 : 99, max: 100.2, title: { display: true, text: "Binary F1 (%)" } } },
      },
    });
  }
}

function renderFigures(group) {
  const gal = $("fig-gallery");
  gal.innerHTML = "";
  $("btn-fig-p4").classList.toggle("active-pill", group === "phase4");
  $("btn-fig-base").classList.toggle("active-pill", group === "baseline");
  OVERVIEW.figures[group].forEach((f) => {
    const el = document.createElement("div");
    el.className = "fig-thumb";
    el.innerHTML = `<img src="${f.url}" alt="${f.title}" loading="lazy" /><div>${f.title}</div>`;
    el.addEventListener("click", () => {
      $("lightbox-title").textContent = f.title;
      $("lightbox-img").src = f.url;
      $("lightbox").classList.remove("hidden");
    });
    gal.appendChild(el);
  });
}

$("ov-dataset").addEventListener("change", renderOverviewTable);
$("ov-task").addEventListener("change", renderOverviewTable);
$("btn-fig-p4").addEventListener("click", () => renderFigures("phase4"));
$("btn-fig-base").addEventListener("click", () => renderFigures("baseline"));
$("lightbox-close").addEventListener("click", () => $("lightbox").classList.add("hidden"));
$("lightbox").addEventListener("click", (e) => { if (e.target.id === "lightbox") $("lightbox").classList.add("hidden"); });

// ==================== TAB 2: LIVE DETECTION ====================
function populateLiveClasses() {
  const ds = $("live-dataset").value;
  const sel = $("live-class");
  sel.innerHTML = CLASS_LISTS[ds]
    .map((c) => `<option value="${c}">${c === "ALL" ? "Tất cả các lớp" : `Lớp: ${c}`}</option>`)
    .join("");
  $("btn-dl-csv").href = `/api/sample-csv/${ds}`;
}
$("live-dataset").addEventListener("change", populateLiveClasses);

async function runSinglePredict(onlyFixed = false) {
  const ds = $("live-dataset").value;
  const task = $("live-task").value;
  const model = $("live-model").value;
  const cls = $("live-class").value;
  const btn = onlyFixed ? $("btn-predict-fixed") : $("btn-predict-one");
  btn.disabled = true;
  try {
    const q = new URLSearchParams({ dataset: ds, task, model, class_filter: cls, only_phase4_fixed: onlyFixed });
    const res = await fetch(`/api/predict/sample?${q}`);
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "Inference error");

    $("single-result").classList.remove("hidden");
    $("sr-gt").textContent = d.ground_truth;
    $("sr-gt").className = `gt-badge ${d.ground_truth_is_attack ? "text-danger" : "text-success"}`;
    $("sr-rowid").textContent = `(Test Sample Row #${d.sample_row_id})`;

    const v = $("sr-verdict");
    if (!d.baseline.correct && d.phase4.correct) {
      v.textContent = "🎯 Phase 4 ĐÃ SỬA ĐÚNG lỗi của Baseline!";
      v.className = "verdict-badge badge-ok";
    } else if (d.baseline.correct && d.phase4.correct) {
      v.textContent = "✔ Cả hai mô hình đều dự đoán chính xác";
      v.className = "verdict-badge badge-info";
    } else {
      v.textContent = "⚠ Ca biên khó — đối chiếu xác suất bên dưới";
      v.className = "verdict-badge badge-warn";
    }

    $("sr-base-ms").textContent = `${d.baseline.latency_ms} ms`;
    $("sr-base-pred").innerHTML = `${d.baseline.prediction} <span class="badge ${d.baseline.correct ? "badge-ok" : "badge-bad"}">${d.baseline.correct ? "ĐÚNG" : "SAI"}</span>`;
    $("sr-base-probs").innerHTML = d.baseline.probabilities.slice(0, 5).map((p, idx) =>
      `<div class="prob-row"><span>${p.class}</span><div class="prob-track"><div class="prob-fill ${idx === 0 ? "top" : ""}" style="width:${p.prob}%"></div></div><span>${p.prob}%</span></div>`
    ).join("");

    $("sr-p4-variant").textContent = d.phase4.variant;
    $("sr-p4-ms").textContent = `${d.phase4.latency_ms} ms`;
    $("sr-p4-pred").innerHTML = `${d.phase4.prediction} <span class="badge ${d.phase4.correct ? "badge-ok" : "badge-bad"}">${d.phase4.correct ? "ĐÚNG" : "SAI"}</span>`;
    $("sr-p4-probs").innerHTML = d.phase4.probabilities.slice(0, 5).map((p, idx) =>
      `<div class="prob-row"><span>${p.class}</span><div class="prob-track"><div class="prob-fill ${idx === 0 ? "top" : ""}" style="width:${p.prob}%"></div></div><span>${p.prob}%</span></div>`
    ).join("");

    $("sr-features").innerHTML = Object.entries(d.key_features)
      .map(([k, val]) => `<div class="fchip"><span>${k}</span><strong>${val}</strong></div>`)
      .join("");
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
  }
}

$("btn-predict-one").addEventListener("click", () => runSinglePredict(false));
$("btn-predict-fixed").addEventListener("click", () => runSinglePredict(true));

// SOC Traffic Stream
async function fetchStreamTick() {
  const ds = $("live-dataset").value;
  const task = $("live-task").value;
  const model = $("live-model").value;
  const res = await fetch(`/api/predict/stream?dataset=${ds}&task=${task}&model=${model}&batch_size=6`);
  const d = await res.json();
  const tbody = $("stream-tbody");
  d.events.forEach((ev) => {
    streamStats.total++;
    if (ev.is_attack) streamStats.atk++;
    if (ev.baseline_correct) streamStats.baseOk++;
    if (ev.phase4_correct) streamStats.p4Ok++;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>#${ev.flow_id}</td>
      <td><span class="badge ${ev.is_attack ? "badge-bad" : "badge-ok"}">${ev.ground_truth}</span></td>
      <td>${ev.baseline_pred} ${ev.baseline_correct ? "✔" : "❌"}</td>
      <td><strong>${ev.phase4_pred}</strong> ${ev.phase4_correct ? "✔" : "❌"}</td>
      <td>${ev.phase4_conf}%</td>`;
    tbody.prepend(tr);
  });
  while (tbody.children.length > 60) tbody.removeChild(tbody.lastChild);

  $("st-total").textContent = streamStats.total;
  $("st-atk").textContent = streamStats.atk;
  $("st-base-acc").textContent = `${((streamStats.baseOk / streamStats.total) * 100).toFixed(1)}%`;
  $("st-p4-acc").textContent = `${((streamStats.p4Ok / streamStats.total) * 100).toFixed(1)}%`;
}

$("btn-stream-toggle").addEventListener("click", () => {
  if (streamTimer) {
    clearInterval(streamTimer);
    streamTimer = null;
    $("btn-stream-toggle").textContent = "▶ Bắt đầu Stream";
  } else {
    fetchStreamTick();
    streamTimer = setInterval(fetchStreamTick, 1100);
    $("btn-stream-toggle").textContent = "⏸ Tạm dừng Stream";
  }
});
$("btn-stream-reset").addEventListener("click", () => {
  streamStats = { total: 0, atk: 0, baseOk: 0, p4Ok: 0 };
  $("stream-tbody").innerHTML = "";
  $("st-total").textContent = "0";
  $("st-atk").textContent = "0";
  $("st-base-acc").textContent = "—";
  $("st-p4-acc").textContent = "—";
});

// CSV Upload
$("btn-upload-csv").addEventListener("click", async () => {
  const fileInput = $("csv-file-input");
  if (!fileInput.files.length) return alert("Hãy chọn file CSV (hoặc bấm nút 'Tải CSV mẫu' ngay phía trên)!");
  const form = new FormData();
  form.append("dataset", $("live-dataset").value);
  form.append("task", $("live-task").value);
  form.append("model", $("live-model").value);
  form.append("file", fileInput.files[0]);

  $("btn-upload-csv").disabled = true;
  try {
    const res = await fetch("/api/predict/upload-csv", { method: "POST", body: form });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "Upload failed");

    $("csv-summary").classList.remove("hidden");
    $("csv-kpis").innerHTML = `
      <div><span>Tổng dòng</span><strong>${d.total_rows}</strong></div>
      <div><span>Khớp cột</span><strong>${d.matched_features}/${d.expected_features}</strong></div>
      <div><span>Baseline Đúng</span><strong>${d.baseline_accuracy !== null ? d.baseline_accuracy + "%" : d.baseline_attacks_flagged + " atk"}</strong></div>
      <div><span>Phase 4 Đúng</span><strong class="text-success">${d.phase4_accuracy !== null ? d.phase4_accuracy + "%" : d.phase4_attacks_flagged + " atk"}</strong></div>`;
    $("csv-tbody").innerHTML = d.preview_rows.map((r) => `<tr>
      <td>${r.row}</td>
      <td>${r.ground_truth ?? "—"}</td>
      <td>${r.baseline_pred} ${r.baseline_correct === null ? "" : (r.baseline_correct ? "✔" : "❌")}</td>
      <td><strong>${r.phase4_pred}</strong> ${r.phase4_correct === null ? "" : (r.phase4_correct ? "✔" : "❌")}</td>
    </tr>`).join("");
  } catch (err) {
    alert(err.message);
  } finally {
    $("btn-upload-csv").disabled = false;
  }
});

// ==================== TAB 3: GAUSSIAN NOISE LAB ====================
$("noise-sigma").addEventListener("input", () => {
  $("noise-sigma-val").textContent = Number($("noise-sigma").value).toFixed(2);
});
document.querySelectorAll(".noise-preset").forEach((b) => {
  b.addEventListener("click", () => {
    $("noise-sigma").value = b.dataset.sigma;
    $("noise-sigma-val").textContent = Number(b.dataset.sigma).toFixed(2);
    runNoiseLab();
  });
});

async function runNoiseLab() {
  const model = $("noise-model").value;
  const sigma = $("noise-sigma").value;
  const n_samples = $("noise-samples").value;
  $("btn-run-noise").disabled = true;
  try {
    const res = await fetch(`/api/lab/noise?model=${model}&sigma=${sigma}&n_samples=${n_samples}`);
    const d = await res.json();
    $("ns-base-ms").textContent = `${d.baseline.latency_ms} ms`;
    $("ns-base-f1").textContent = `${d.baseline.f1}%`;
    $("ns-base-details").innerHTML = `
      <div><span>Recall</span><strong>${d.baseline.recall}%</strong></div>
      <div><span>Precision</span><strong>${d.baseline.precision}%</strong></div>
      <div><span>PR-AUC</span><strong>${d.baseline.pr_auc}%</strong></div>
      <div><span>FPR</span><strong>${d.baseline.fpr}%</strong></div>`;

    $("ns-p4-title").textContent = `Phase 4: ${d.p4_model} (τ* = ${d.phase4.threshold})`;
    $("ns-p4-ms").textContent = `${d.phase4.latency_ms} ms`;
    $("ns-p4-f1").textContent = `${d.phase4.f1}%`;
    $("ns-p4-details").innerHTML = `
      <div><span>Recall</span><strong class="text-success">${d.phase4.recall}%</strong></div>
      <div><span>Precision</span><strong>${d.phase4.precision}%</strong></div>
      <div><span>PR-AUC</span><strong>${d.phase4.pr_auc}%</strong></div>
      <div><span>FPR</span><strong>${d.phase4.fpr}%</strong></div>`;
  } finally {
    $("btn-run-noise").disabled = false;
  }
}
$("btn-run-noise").addEventListener("click", runNoiseLab);

function renderNoiseFullChart() {
  if (!OVERVIEW) return;
  const ctx = $("noiseFullChart").getContext("2d");
  const sigmas = ["sigma_0.0", "sigma_0.1", "sigma_0.25", "sigma_0.5"];
  const labels = ["σ = 0.0 (Sạch)", "σ = 0.1", "σ = 0.25", "σ = 0.5"];
  const b = OVERVIEW.baseline_tables.noise.models;
  const p = OVERVIEW.phase4_tables.noise.models;

  noiseChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "XGBoost — Phase 4 Noise-Robust", data: sigmas.map((s) => p.xgboost_noise_robust[s].f1.mean), borderColor: "#10b981", backgroundColor: "#10b981", borderWidth: 3, tension: 0.15 },
        { label: "LightGBM — Phase 4 Noise-Robust", data: sigmas.map((s) => p.lightgbm_noise_robust[s].f1.mean), borderColor: "#3b82f6", backgroundColor: "#3b82f6", borderWidth: 3, tension: 0.15 },
        { label: "Random Forest — Phase 4 Calibrated", data: sigmas.map((s) => p.random_forest_noise_calibrated[s].f1.mean), borderColor: "#f59e0b", backgroundColor: "#f59e0b", borderWidth: 2.5, tension: 0.15 },
        { label: "XGBoost — Baseline Bài báo (Bảng 12)", data: sigmas.map((s) => b.xgboost[s].f1.mean), borderColor: "#ef4444", borderDash: [6, 4], borderWidth: 2.5, tension: 0.15 },
        { label: "LightGBM — Baseline", data: sigmas.map((s) => b.lightgbm[s].f1.mean), borderColor: "#94a3b8", borderDash: [6, 4], borderWidth: 2, tension: 0.15 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { min: 0, max: 105, title: { display: true, text: "Binary F1 (%)" } } },
    },
  });
}

// ==================== TAB 4: ZERO-DAY LOFO LAB ====================
async function runLofoLab() {
  const family = $("lofo-family").value;
  const seed = $("lofo-seed").value;
  $("btn-run-lofo").disabled = true;
  try {
    const res = await fetch(`/api/lab/lofo?family=${family}&seed=${seed}`);
    const d = await res.json();
    const full = d.full_test_504k_metrics;
    const live = d.live_sample_metrics;

    $("lf-base-ms").textContent = `Live: ${live.baseline_ms} ms`;
    $("lf-base-rec").textContent = `${full.baseline_unseen_recall}%`;
    $("lf-base-details").innerHTML = `
      <div><span>Live Sample Recall</span><strong>${live.baseline_unseen_recall}%</strong></div>
      <div><span>Seen Recall (504k)</span><strong>${full.baseline_seen_recall}%</strong></div>
      <div><span>Benign FPR (504k)</span><strong>${full.baseline_benign_fpr}%</strong></div>
      <div><span>Số luồng test</span><strong>${full.held_out_flows.toLocaleString("en-US")}</strong></div>`;

    $("lf-hyb-ms").textContent = `Live: ${live.hybrid_ms} ms`;
    $("lf-hyb-rec").textContent = `${full.hybrid_unseen_recall}%`;
    $("lf-hyb-details").innerHTML = `
      <div><span>Live Sample Recall</span><strong class="text-success">${live.hybrid_unseen_recall}%</strong></div>
      <div><span>Seen Recall (504k)</span><strong>${full.hybrid_seen_recall}%</strong></div>
      <div><span>Benign FPR (504k)</span><strong>${full.hybrid_benign_fpr}%</strong></div>
      <div><span>Ngưỡng Val τ_xgb</span><strong>${live.xgb_thresh}</strong></div>`;
  } finally {
    $("btn-run-lofo").disabled = false;
  }
}
$("btn-run-lofo").addEventListener("click", runLofoLab);

function renderLofoFullChart() {
  if (!OVERVIEW) return;
  const ctx = $("lofoFullChart").getContext("2d");
  const fams = ["DoS", "DDoS", "PortScan", "BruteForce", "Web", "Bot"];
  const baseF = OVERVIEW.phase4_tables.lofo.baseline.xgboost;
  const p4F = OVERVIEW.phase4_tables.lofo.models.xgboost_hybrid_lofo;

  lofoChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: fams,
      datasets: [
        { label: "Baseline XGBoost (Bảng 11)", data: fams.map((f) => baseF[f].held_out_recall.mean), backgroundColor: "rgba(148, 163, 184, 0.7)", borderRadius: 5 },
        { label: "Phase 4 Hybrid (XGBoost + Benign Anomaly)", data: fams.map((f) => p4F[f].held_out_recall.mean), backgroundColor: "rgba(16, 185, 129, 0.85)", borderRadius: 5 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { min: 0, max: 100, title: { display: true, text: "Unseen Family Recall (%)" } } },
    },
  });
}

// ==================== TAB 5: FULL TEST AUDITOR ====================
$("btn-run-audit").addEventListener("click", async () => {
  const ds = $("aud-dataset").value;
  const task = $("aud-task").value;
  const model = $("aud-model").value;
  const seed = $("aud-seed").value;
  const btn = $("btn-run-audit");
  btn.disabled = true;
  btn.textContent = "⏳ Đang nạp .joblib & chấm toàn bộ tập Test thật...";
  try {
    const res = await fetch(`/api/audit/full?dataset=${ds}&task=${task}&model=${model}&seed=${seed}`);
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "Audit error");

    $("audit-result").classList.remove("hidden");
    const statusText = d.all_within_tolerance
      ? `✔ KHỚP TUYỆT ĐỐI VỚI CSV BÁO CÁO (Max |Δ| = ${d.max_abs_diff.toExponential(2)})`
      : `⚠ Sai lệch max |Δ| = ${d.max_abs_diff}`;
    $("audit-banner").innerHTML = `
      <div>
        <strong>${ds.toUpperCase()} / ${task} / ${model} / seed_${seed}</strong>
        <span class="muted"> — ${d.n_test_samples.toLocaleString("en-US")} luồng Test thật • Thời gian suy luận: <strong>${d.inference_time_s.toFixed(2)}s</strong></span>
      </div>
      <span class="verdict-badge ${d.all_within_tolerance ? "badge-ok" : "badge-warn"}">${statusText}</span>`;

    const tbody = $("audit-table").querySelector("tbody");
    tbody.innerHTML = Object.entries(d.comparison || {}).map(([metric, c]) => `<tr>
      <td><strong>${metric}</strong></td>
      <td>${fmtNum(c.live, 4)}</td>
      <td>${fmtNum(c.reported, 4)}</td>
      <td><code>${Number(c.abs_diff).toExponential(3)}</code></td>
      <td><span class="badge ${c.abs_diff < 1e-6 ? "badge-ok" : "badge-warn"}">${c.abs_diff < 1e-6 ? "KHỚP 100%" : "Kiểm tra"}</span></td>
    </tr>`).join("");
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 Chấm Toàn bộ Tập Test Thật & Đối chiếu CSV";
  }
});

initOverview();
