/* AIoT HW2 — WHO Life Expectancy multiple linear regression
 * Problem 1 / 2：讀取 notebook 匯出的 model/problem*.json 繪圖與互動預測
 * 自訂資料集：在瀏覽器內完成 CSV 解析、one-hot、OLS、95% CI / PI
 */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
// 奶茶／大地色系（與 style.css 的 :root 對應）
const PAL = {
  caramel: "#8c6240", espresso: "#4a3426", latte: "#dcc5a8", sage: "#7d8f5a",
  terracotta: "#a8442f", honey: "#c9923f", taupe: "#a39383", sand: "#e3d3bf",
  piBand: "rgba(214,188,155,0.5)", ciBand: "rgba(140,98,64,0.65)",
  dotA: "rgba(140,98,64,0.55)", dotB: "rgba(125,143,90,0.7)",
};
const COLORS = [PAL.caramel, PAL.honey, PAL.sage, PAL.terracotta, "#5f6f73"];
if (window.Chart) {
  Chart.defaults.color = "#7a6656";
  Chart.defaults.borderColor = "rgba(120,90,60,0.12)";
}
const state = { charts: {}, models: {}, custom: null, customFit: null };

const fmt = (v, d = 3) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const fmtP = (p) => (p == null || !isFinite(p) ? "–" : p < 0.001 ? "<0.001" : p.toFixed(3));
const fmtRaw = (v) => {
  if (!isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString("en-US");
  if (a >= 100) return v.toFixed(1);
  return v.toFixed(2);
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* ---------------------------------------------------------------- statistics */

function lgamma(x) {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function betacf(a, b, x) {
  const FPMIN = 1e-300;
  const qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 3e-14) break;
  }
  return h;
}

// regularized incomplete beta I_x(a, b)
function ibeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

const tTwoSidedP = (t, df) => ibeta(df / (df + t * t), df / 2, 0.5);

function tCrit(df, alpha = 0.05) {
  let lo = 0, hi = 1000;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (tTwoSidedP(mid, df) > alpha) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

function invert(A) {
  const n = A.length;
  const M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  const scale = Math.max(...A.map((r, i) => Math.abs(r[i])));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-10 * scale) throw new Error("設計矩陣奇異：特徵之間可能完全共線（例如同一資訊重複出現）。");
    [M[c], M[piv]] = [M[piv], M[c]];
    const p = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= p;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((r) => r.slice(n));
}

const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const quad = (x, S) => x.reduce((s, xi, i) => s + xi * dot(S[i], x), 0);

/** OLS on standardized columns (numerically stable), coefficients reported in original units. */
function fitOLS(X, y, names) {
  const n = X.length, p = X[0].length;
  if (n - p - 1 <= 0) throw new Error(`樣本數 (${n}) 不足以估計 ${p} 個係數，請減少特徵。`);
  const mean = [], sd = [];
  for (let j = 0; j < p; j++) {
    const col = X.map((r) => r[j]);
    const m = col.reduce((a, b) => a + b, 0) / n;
    const s = Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / n);
    if (s === 0) throw new Error(`特徵「${names[j]}」在訓練集中為常數，請取消勾選。`);
    mean.push(m);
    sd.push(s);
  }
  const toZ = (r) => [1, ...r.map((v, j) => (v - mean[j]) / sd[j])];
  const Z = X.map(toZ);
  const q = p + 1;
  const XtX = Array.from({ length: q }, () => new Array(q).fill(0));
  const Xty = new Array(q).fill(0);
  Z.forEach((z, i) => {
    for (let a = 0; a < q; a++) {
      Xty[a] += z[a] * y[i];
      for (let b = a; b < q; b++) XtX[a][b] += z[a] * z[b];
    }
  });
  for (let a = 0; a < q; a++) for (let b = 0; b < a; b++) XtX[a][b] = XtX[b][a];
  const inv = invert(XtX);
  const bz = inv.map((r) => dot(r, Xty));
  const fitted = Z.map((z) => dot(z, bz));
  const resid = y.map((v, i) => v - fitted[i]);
  const sse = resid.reduce((a, r) => a + r * r, 0);
  const df = n - q;
  const s2 = sse / df;
  const covZ = inv.map((r) => r.map((v) => v * s2));

  // back-transform: beta = T·bz, cov = T·covZ·Tᵀ
  const T = Array.from({ length: q }, () => new Array(q).fill(0));
  T[0][0] = 1;
  for (let j = 1; j < q; j++) {
    T[0][j] = -mean[j - 1] / sd[j - 1];
    T[j][j] = 1 / sd[j - 1];
  }
  const beta = T.map((r) => dot(r, bz));
  const TC = T.map((r) => covZ[0].map((_, j) => r.reduce((s, v, k) => s + v * covZ[k][j], 0)));
  const cov = TC.map((r) => T.map((t) => dot(r, t)));
  const t = tCrit(df);
  const coefs = beta.map((b, j) => {
    const se = Math.sqrt(cov[j][j]);
    const tv = b / se;
    return { name: j === 0 ? "const" : names[j - 1], coef: b, se, t: tv, p: tTwoSidedP(Math.abs(tv), df) };
  });
  const predict = (row) => {
    const z = toZ(row);
    const yhat = dot(z, bz);
    const seMean = Math.sqrt(Math.max(quad(z, covZ), 0));
    const sePred = Math.sqrt(s2 + seMean * seMean);
    return { yhat, ci: [yhat - t * seMean, yhat + t * seMean], pi: [yhat - t * sePred, yhat + t * sePred] };
  };
  return { n, p, df, s2, t, coefs, predict, fitted, resid };
}

function metrics(y, yhat, k) {
  const n = y.length;
  const mean = y.reduce((a, b) => a + b, 0) / n;
  let sse = 0, sae = 0, sst = 0;
  y.forEach((v, i) => {
    sse += (v - yhat[i]) ** 2;
    sae += Math.abs(v - yhat[i]);
    sst += (v - mean) ** 2;
  });
  const r2 = 1 - sse / sst;
  return { MAE: sae / n, MSE: sse / n, RMSE: Math.sqrt(sse / n), R2: r2, AdjR2: n - k - 1 > 0 ? 1 - ((1 - r2) * (n - 1)) / (n - k - 1) : NaN };
}

/* ---------------------------------------------------------------- charts */

function makeChart(id, config) {
  if (state.charts[id]) state.charts[id].destroy();
  const el = document.getElementById(id);
  if (!el) return null;
  config.options = Object.assign({ responsive: true, maintainAspectRatio: false, animation: false }, config.options || {});
  state.charts[id] = new Chart(el, config);
  return state.charts[id];
}

function intervalChart(id, points, yLabel) {
  const s = [...points].sort((a, b) => a.pred - b.pred);
  const labels = s.map((_, i) => i + 1);
  const outside = s.map((p) => p.actual < p.pi[0] || p.actual > p.pi[1]);
  const band = (data, fill, bg) => ({ data, fill, backgroundColor: bg, borderWidth: 0, pointRadius: 0, tension: 0 });
  makeChart(id, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "_pi_lo", ...band(s.map((p) => p.pi[0]), false, "transparent") },
        { label: "95% 預測區間 (PI)", ...band(s.map((p) => p.pi[1]), "-1", PAL.piBand) },
        { label: "_ci_lo", ...band(s.map((p) => p.ci[0]), false, "transparent") },
        { label: "95% 信賴區間 (CI)", ...band(s.map((p) => p.ci[1]), "-1", PAL.ciBand) },
        { label: "預測值", data: s.map((p) => p.pred), borderColor: PAL.espresso, backgroundColor: PAL.espresso, borderWidth: 1.5, pointRadius: 0, fill: false },
        {
          label: "實際值（測試集）", data: s.map((p) => p.actual), showLine: false, pointRadius: 2.5,
          pointBackgroundColor: outside.map((o) => (o ? PAL.terracotta : PAL.sage)), pointBorderWidth: 0,
          borderColor: PAL.sage, backgroundColor: PAL.sage,
        },
      ],
    },
    options: {
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { filter: (item) => !item.text.startsWith("_") } },
        tooltip: {
          filter: (item) => item.datasetIndex >= 4,
          callbacks: {
            title: (items) => {
              const p = s[items[0].dataIndex];
              return p.country ? `${p.country}${p.year ? " " + p.year : ""}` : `樣本 #${items[0].dataIndex + 1}`;
            },
            afterBody: (items) => {
              const p = s[items[0].dataIndex];
              return [`CI [${fmt(p.ci[0], 2)}, ${fmt(p.ci[1], 2)}]`, `PI [${fmt(p.pi[0], 2)}, ${fmt(p.pi[1], 2)}]`];
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "測試樣本（依預測值排序）" }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: yLabel } },
      },
    },
  });
}

function avpChart(id, groups, label) {
  const all = groups.flatMap((g) => g.points);
  const lo = Math.min(...all.map((p) => Math.min(p.actual, p.pred)));
  const hi = Math.max(...all.map((p) => Math.max(p.actual, p.pred)));
  makeChart(id, {
    type: "scatter",
    data: {
      datasets: [
        ...groups.map((g, i) => ({
          label: g.label, data: g.points.map((p) => ({ x: p.actual, y: p.pred })),
          pointRadius: 2.5, backgroundColor: i === 0 ? PAL.dotA : PAL.dotB,
        })),
        { label: "y = x", type: "line", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: PAL.terracotta, borderDash: [5, 4], pointRadius: 0, borderWidth: 1 },
      ],
    },
    options: {
      scales: {
        x: { title: { display: true, text: `實際 ${label}` } },
        y: { title: { display: true, text: `預測 ${label}` } },
      },
    },
  });
}

function residualChart(id, fitted, resid) {
  makeChart(id, {
    type: "scatter",
    data: {
      datasets: [
        { label: "殘差", data: fitted.map((f, i) => ({ x: f, y: resid[i] })), pointRadius: 2.5, backgroundColor: PAL.dotA },
        { label: "0", type: "line", data: [{ x: Math.min(...fitted), y: 0 }, { x: Math.max(...fitted), y: 0 }], borderColor: PAL.terracotta, pointRadius: 0, borderWidth: 1 },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "預測值 (fitted)" } }, y: { title: { display: true, text: "殘差" } } },
    },
  });
}

function hbarChart(id, labels, values, xLabel, colorFn) {
  makeChart(id, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: values.map(colorFn || ((v) => (v >= 0 ? PAL.sage : PAL.terracotta))) }] },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: xLabel } }, y: { ticks: { autoSkip: false, font: { size: 11 } } } },
    },
  });
}

function selectionChart(id, sel) {
  const methods = Object.keys(sel.curves);
  const p = sel.curves[methods[0]].length;
  const chosen = sel.curves[sel.method][sel.k - 1];
  makeChart(id, {
    type: "line",
    data: {
      datasets: [
        ...methods.map((m, i) => ({
          label: m, data: sel.curves[m].map((c) => ({ x: c.k, y: c.rmse })),
          borderColor: COLORS[i % COLORS.length], backgroundColor: COLORS[i % COLORS.length], pointRadius: 2, borderWidth: 1.6,
        })),
        { label: "最佳值 × 1.01", data: [{ x: 1, y: sel.threshold }, { x: p, y: sel.threshold }], borderColor: PAL.taupe, borderDash: [5, 4], pointRadius: 0, borderWidth: 1 },
        { label: `選擇：${sel.method}, k=${sel.k}`, data: [{ x: sel.k, y: chosen.rmse }], showLine: false, pointRadius: 7, pointStyle: "star", borderColor: PAL.terracotta, backgroundColor: PAL.terracotta, borderWidth: 2 },
      ],
    },
    options: {
      scales: {
        x: { type: "linear", min: 1, max: p, ticks: { stepSize: 1 }, title: { display: true, text: "特徵數 k" } },
        y: { title: { display: true, text: "GroupKFold CV RMSE" } },
      },
    },
  });
}

/* ---------------------------------------------------------------- Problem 1 / 2 */

const kpi = (label, value, note = "") =>
  `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div><div class="note">${note}</div></div>`;

const ASSUMPTION_TEXT = [
  "殘差 vs 預測值應無系統性曲線；Rainbow 檢定 p > 0.05 表示線性關係成立。",
  "Durbin-Watson ≈ 2 表示誤差無自相關。panel data（同國家逐年）天生會違反 → 改用以國家為群集的穩健標準誤。",
  "Breusch-Pagan p > 0.05 表示殘差變異固定；違反時係數仍不偏，但標準誤需用穩健估計。",
  "Jarque-Bera p > 0.05 表示殘差近似常態；大樣本下輕微偏離影響有限（中央極限定理）。",
  "VIF < 10 表示沒有嚴重共線性；特徵選擇後已迭代剔除 VIF > 10 的特徵。",
];

async function loadProblem(key) {
  const view = $("#problem-view");
  if (!state.models[key]) {
    view.innerHTML = `<div class="loading">載入模型中…</div>`;
    try {
      const res = await fetch(`model/${key}.json`);
      if (!res.ok) throw new Error(res.status);
      state.models[key] = await res.json();
    } catch (e) {
      view.innerHTML = `<div class="card"><p class="status err">無法讀取 model/${key}.json（${esc(e.message)}）。若是直接用檔案開啟 index.html，請改用本機伺服器，例如 <code>python -m http.server</code>。</p></div>`;
      return;
    }
  }
  renderProblem(state.models[key]);
}

function renderProblem(m) {
  const sel = m.selection;
  const L = (n) => m.labels[n] || n;
  const rowOf = (prefix) => m.models.find((r) => r.Model.startsWith(prefix));
  const selRow = rowOf("OLS – selected"), allRow = rowOf("OLS – all"), base = rowOf("Baseline");
  const iv = m.intervals;
  const ds = m.dataset;
  const corrSorted = Object.entries(m.corr).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const methods = Object.keys(sel.rankings);
  const asmPass = m.assumptions.filter((a) => a.pass).length;
  const pts = iv.points;
  // 取測試集最新年度、依預測值平均間隔抽 10 筆，涵蓋低到高壽命的國家
  const latestYear = Math.max(...pts.map((p) => p.year));
  const pool = pts.filter((p) => p.year === latestYear).sort((a, b) => a.pred - b.pred);
  const nSample = Math.min(10, pool.length);
  const samples = Array.from({ length: nSample }, (_, i) => pool[Math.round((i * (pool.length - 1)) / Math.max(nSample - 1, 1))]);

  $("#problem-view").innerHTML = `
  <div class="kpis">
    ${kpi("測試集 R²", fmt(selRow.R2, 3), `OLS · ${sel.selected.length} 個特徵（未見過的國家）`)}
    ${kpi("測試集 RMSE", `${fmt(selRow.RMSE, 2)}`, `${m.unit} · MAE ${fmt(selRow.MAE, 2)}`)}
    ${kpi("Adjusted R²", fmt(selRow.AdjR2, 3), `全特徵 OLS：${fmt(allRow.AdjR2, 3)}`)}
    ${kpi("95% PI 覆蓋率", pct(iv.coverage), `平均寬度 ${fmt(iv.mean_pi_width, 1)} ${m.unit}`)}
    ${kpi("特徵選擇", `${sel.selected.length} / ${ds.n_features}`, `${sel.method} + VIF`)}
    ${kpi("Baseline R²", fmt(base.R2, 3), esc(base.Model.replace("Baseline simple LR ", "單一特徵 ")))}
  </div>

  <section class="card">
    <h2>互動預測（Deployment）</h2>
    <p class="muted">以全部資料重新估計的最終 OLS 模型（n = ${m.final.n}，R² = ${fmt(m.final.r2, 3)}）。選擇國家帶入其最新年度數值，或拖曳滑桿進行 what-if 模擬；預測值與 95% 區間即時在瀏覽器中計算：
      CI = ŷ ± t·√(x₀ᵀΣx₀)，PI = ŷ ± t·√(σ̂² + x₀ᵀΣx₀)。</p>
    <div class="predictor">
      <div>
        <label class="field-label" for="preset">帶入國家數值（最新年度）</label>
        <select id="preset"></select>
        <div id="sliders" style="margin-top:14px"></div>
      </div>
      <div class="result-box" id="pred-out"></div>
    </div>
  </section>

  <section class="card">
    <h2>預測圖：95% 信賴區間與預測區間（測試集）</h2>
    <p class="muted">測試集為 ${ds.test_countries} 個<b>完全沒出現在訓練集</b>的國家（${ds.n_test} 筆）。焦糖色帶 = 平均值的信賴區間，奶茶色帶 = 單一觀測的預測區間；綠點 = 實際值，赤陶色點 = 落在 PI 外。
      平均 CI 寬度 ${fmt(iv.mean_ci_width, 2)}、PI 寬度 ${fmt(iv.mean_pi_width, 2)} ${m.unit}。</p>
    <div class="chart-box tall"><canvas id="c-interval"></canvas></div>
    <h3>測試集預測樣本（${latestYear} 年，依預測值平均抽取 ${nSample} 國）</h3>
    <div class="table-scroll"><table>
      <thead><tr><th>國家</th><th>實際</th><th>預測</th><th>誤差</th><th>95% CI</th><th>95% PI</th><th>落在 PI 內</th></tr></thead>
      <tbody>${samples.map((p) => {
        const inside = p.actual >= p.pi[0] && p.actual <= p.pi[1];
        return `<tr><td>${esc(p.country)}</td><td>${fmt(p.actual, 1)}</td><td>${fmt(p.pred, 1)}</td><td>${p.actual - p.pred >= 0 ? "+" : ""}${fmt(p.actual - p.pred, 1)}</td>
          <td>[${fmt(p.ci[0], 1)}, ${fmt(p.ci[1], 1)}]</td><td>[${fmt(p.pi[0], 1)}, ${fmt(p.pi[1], 1)}]</td>
          <td class="${inside ? "pass" : "fail"}">${inside ? "✔" : "✖"}</td></tr>`;
      }).join("")}</tbody>
    </table></div>
  </section>

  <section class="grid-2">
    <div class="card">
      <h2>實際 vs 預測</h2>
      <div class="chart-box"><canvas id="c-avp"></canvas></div>
    </div>
    <div class="card">
      <h2>測試集殘差</h2>
      <div class="chart-box"><canvas id="c-resid"></canvas></div>
    </div>
  </section>

  <section class="card">
    <h2>特徵選擇（Feature Selection）</h2>
    <p class="muted">5 種方法各自產生特徵排序，以<b>訓練集</b> GroupKFold(5) 交叉驗證 RMSE 評估前 k 個特徵。
      最佳為 ${esc(sel.best.method)} k=${sel.best.k}（RMSE ${fmt(sel.best.rmse, 3)}）；在「最佳值 × 1.01」內選最小 k → <b>${esc(sel.method)}，k = ${sel.k}</b>，再以 VIF &gt; 10 剔除共線特徵。</p>
    <div class="chart-box"><canvas id="c-select"></canvas></div>
    <h3>最終特徵</h3>
    <div>${sel.selected.map((f) => `<span class="pill">${esc(L(f))}</span>`).join("")}
      ${sel.vif_dropped.map((d) => `<span class="pill drop" title="VIF = ${fmt(d.vif, 1)}">${esc(L(d.feature))}（VIF ${fmt(d.vif, 0)}）</span>`).join("")}</div>
    <h3>各方法前 10 名排序</h3>
    <div class="table-scroll"><table>
      <thead><tr><th>排名</th>${methods.map((mm) => `<th style="text-align:left">${esc(mm)}</th>`).join("")}</tr></thead>
      <tbody>${Array.from({ length: Math.min(10, sel.rankings[methods[0]].length) }, (_, i) =>
        `<tr><td>${i + 1}</td>${methods.map((mm) => `<td style="text-align:left">${esc(sel.rankings[mm][i])}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>
  </section>

  <section class="grid-2">
    <div class="card">
      <h2>模型比較（測試集）</h2>
      <div class="table-scroll"><table>
        <thead><tr><th>模型</th><th>k</th><th>Train R²</th><th>MAE</th><th>MSE</th><th>RMSE</th><th>R²</th><th>Adj R²</th></tr></thead>
        <tbody>${m.models.map((r) => `<tr class="${r === selRow ? "hl" : ""}"><td>${esc(r.Model)}</td><td>${r.k}</td><td>${fmt(r["Train R2"], 3)}</td>
          <td>${fmt(r.MAE, 2)}</td><td>${fmt(r.MSE, 1)}</td><td>${fmt(r.RMSE, 2)}</td><td>${fmt(r.R2, 3)}</td><td>${fmt(r.AdjR2, 3)}</td></tr>`).join("")}</tbody>
      </table></div>
      <div class="chart-box short" style="margin-top:12px"><canvas id="c-models"></canvas></div>
    </div>
    <div class="card">
      <h2>係數（訓練集 OLS）</h2>
      <p class="muted">panel data 誤差不獨立，因此同時列出以國家為群集的穩健標準誤 p 值（p_cluster）。</p>
      <div class="table-scroll"><table>
        <thead><tr><th>特徵</th><th>係數</th><th>SE</th><th>p</th><th>p_cluster</th></tr></thead>
        <tbody>${m.coef_table_train.map((r) => `<tr><td>${r.name === "const" ? "截距" : esc(L(r.name))}${m.final.features.find((f) => f.name === r.name)?.log ? " <span class='muted'>(log1p)</span>" : ""}</td>
          <td>${fmt(r.coef, 4)}</td><td>${fmt(r.se, 4)}</td><td>${fmtP(r.p)}</td><td>${fmtP(r.p_cluster)}</td></tr>`).join("")}</tbody>
      </table></div>
      <h3>標準化係數（+1 SD 的影響）</h3>
      <div class="chart-box short"><canvas id="c-coef"></canvas></div>
    </div>
  </section>

  <section class="card">
    <h2>五大回歸假設檢驗（${asmPass} / 5 通過）</h2>
    <div class="table-scroll"><table>
      <thead><tr><th>假設</th><th style="text-align:left">檢定</th><th>統計量</th><th>p 值</th><th>結果</th><th style="text-align:left">說明</th></tr></thead>
      <tbody>${m.assumptions.map((a, i) => `<tr><td>${esc(a.assumption)}</td><td style="text-align:left">${esc(a.test)}</td><td>${fmt(a.statistic, 3)}</td>
        <td>${fmtP(a.p_value)}</td><td class="${a.pass ? "pass" : "fail"}">${a.pass ? "✔ 通過" : "✖ 違反"}</td>
        <td style="text-align:left;white-space:normal;min-width:260px">${ASSUMPTION_TEXT[i]}</td></tr>`).join("")}</tbody>
    </table></div>
    <h3>殘差診斷圖（Notebook 輸出）</h3>
    <img class="diag" src="assets/${m.key === "problem1" ? "p1" : "p2"}_residual_diagnostics.png" alt="residual diagnostics" loading="lazy">
  </section>

  <section class="grid-2">
    <div class="card">
      <h2>與目標的相關係數</h2>
      <p class="muted">log 轉換後的 Pearson 相關；赤陶 = 負相關、橄欖綠 = 正相關。</p>
      <div class="chart-box tall"><canvas id="c-corr"></canvas></div>
    </div>
    <div class="card">
      <h2>CRISP-DM 流程</h2>
      <div class="crisp">${crispCards(m, selRow, corrSorted, L)}</div>
    </div>
  </section>`;

  intervalChart("c-interval", pts, m.target_label);
  avpChart("c-avp", [{ label: "測試集", points: pts }], m.target_label);
  residualChart("c-resid", pts.map((p) => p.pred), pts.map((p) => p.actual - p.pred));
  selectionChart("c-select", sel);
  makeChart("c-models", {
    type: "bar",
    data: {
      labels: m.models.map((r) => r.Model.replace(" – ", "\n").replace("Baseline simple LR", "Baseline")),
      datasets: [{ label: "測試集 R²", data: m.models.map((r) => r.R2), backgroundColor: m.models.map((r) => (r === selRow ? PAL.caramel : PAL.sand)) }],
    },
    options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1, title: { display: true, text: "R²" } }, x: { ticks: { font: { size: 10 } } } } },
  });
  const sc = Object.entries(m.std_coef).sort((a, b) => a[1] - b[1]);
  hbarChart("c-coef", sc.map(([k]) => L(k)), sc.map(([, v]) => v), `對${m.target_label}的影響`);
  const cr = Object.entries(m.corr).sort((a, b) => a[1] - b[1]);
  hbarChart("c-corr", cr.map(([k]) => L(k)), cr.map(([, v]) => v), "相關係數 r");

  buildPredictor(m);
}

function crispCards(m, selRow, corrSorted, L) {
  const ds = m.dataset, sel = m.selection;
  const top3 = corrSorted.slice(0, 3).map(([k, v]) => `${L(k)} (${v >= 0 ? "+" : ""}${v.toFixed(2)})`).join("、");
  const dropped = sel.vif_dropped.length ? `，VIF 剔除 ${sel.vif_dropped.map((d) => L(d.feature)).join("、")}` : "";
  const failed = m.assumptions.filter((a) => !a.pass).map((a) => a.assumption.replace(/^\d\.\s*/, "")).join("、");
  const goal = m.key === "problem1"
    ? "預測一國當年平均壽命，找出影響壽命的關鍵健康與社經因子，並提供政策 what-if 模擬。"
    : "預測一國 15–60 歲成人死亡率，作為對照問題，觀察線性模型在雜訊較大的目標上的表現。";
  return [
    ["1. Business Understanding", goal],
    ["2. Data Understanding", `${ds.rows} 筆、${ds.countries} 國、${ds.n_features} 個特徵。與目標相關最強：${top3}。Population、Hepatitis B、GDP 缺值 15–22%。`],
    ["3. Data Preparation", `Status→0/1；收入組成與教育年數的 0 視為缺值；7 個右偏欄位 log1p；以國家分組切分 train ${ds.n_train} / test ${ds.n_test}；訓練集中位數補值、標準化。`],
    ["4. Modeling", `5 種特徵選擇（SelectKBest、RFE、SFS、Lasso path、Permutation）× k=1…${ds.n_features}；選 ${sel.method} k=${sel.k}${dropped}；比較 OLS、Ridge、Lasso。`],
    ["5. Evaluation", `測試集 R² ${fmt(selRow.R2, 3)}、RMSE ${fmt(selRow.RMSE, 2)} ${m.unit}；95% PI 覆蓋 ${pct(m.intervals.coverage)}。未通過假設：${failed || "無"} → 以 cluster-robust SE 判斷顯著性。`],
    ["6. Deployment", "全資料重估 OLS → 匯出係數與共變異矩陣 JSON → 本網頁（Vercel 靜態部署）即時計算預測值與 CI／PI。"],
  ].map(([h, t]) => `<div><b>${h}</b>${esc(t)}</div>`).join("");
}

function buildPredictor(m) {
  const f = m.final;
  const L = (n) => m.labels[n] || n;
  const presetSel = $("#preset");
  const countries = Object.keys(m.presets).sort();
  presetSel.innerHTML = `<option value="">（各特徵中位數）</option>` +
    countries.map((c) => `<option value="${esc(c)}">${esc(c)}（${m.presets[c].year}）</option>`).join("");
  presetSel.value = m.presets.Japan ? "Japan" : countries[0];

  const metas = f.features.map((ft) => {
    const tf = ft.log ? Math.log1p : (v) => v;
    const inv = ft.log ? Math.expm1 : (v) => v;
    const min = tf(ft.min), max = tf(ft.max);
    const step = ft.integer && !ft.log ? 1 : (max - min) / 400;
    return { ft, tf, inv, min, max, step };
  });

  $("#sliders").innerHTML = metas.map((mt, i) => `
    <div class="slider">
      <div class="top"><label for="s${i}">${esc(L(mt.ft.name))}${mt.ft.log ? " <span class='muted'>(log 尺度)</span>" : ""}</label><b id="v${i}"></b></div>
      <input id="s${i}" type="range" min="${mt.min}" max="${mt.max}" step="${mt.step}">
    </div>`).join("");

  const pts = m.intervals.points;
  const axisLo = Math.floor(Math.min(...pts.map((p) => Math.min(p.actual, p.pi[0]))));
  const axisHi = Math.ceil(Math.max(...pts.map((p) => Math.max(p.actual, p.pi[1]))));
  const pos = (v) => `${Math.min(100, Math.max(0, ((v - axisLo) / (axisHi - axisLo)) * 100))}%`;
  const width = (a, b) => `${Math.max(0, ((Math.min(b, axisHi) - Math.max(a, axisLo)) / (axisHi - axisLo)) * 100)}%`;

  const update = () => {
    const country = presetSel.value;
    const x = [1];
    metas.forEach((mt, i) => {
      const v = Number($(`#s${i}`).value);
      x.push(v);
      $(`#v${i}`).textContent = fmtRaw(mt.inv(v));
    });
    const yhat = dot(x, f.params);
    const seMean = Math.sqrt(Math.max(quad(x, f.cov), 0));
    const sePred = Math.sqrt(f.sigma2 + seMean * seMean);
    const ci = [yhat - f.t * seMean, yhat + f.t * seMean];
    const pi = [yhat - f.t * sePred, yhat + f.t * sePred];
    const actual = country ? m.presets[country].actual : null;
    $("#pred-out").innerHTML = `
      <div class="muted" style="margin:0">預測${esc(m.target_label)}</div>
      <div class="big">${fmt(yhat, 1)} <span style="font-size:18px">${m.unit}</span></div>
      <div class="gauge">
        <div class="pi" style="left:${pos(pi[0])};width:${width(pi[0], pi[1])}"></div>
        <div class="ci" style="left:${pos(ci[0])};width:${width(ci[0], ci[1])}"></div>
        ${actual != null ? `<div class="actual" style="left:${pos(actual)}" title="實際值"></div>` : ""}
      </div>
      <div class="gauge-axis"><span>${axisLo}</span><span>${axisHi}</span></div>
      <div class="legend-inline"><span class="sw-pi"></span>95% PI<span class="sw-ci"></span>95% CI${actual != null ? `<span class="sw-actual"></span>實際值` : ""}</div>
      <div style="margin-top:12px">
        <div class="row-kv"><span>95% 信賴區間 (CI)</span><b>[${fmt(ci[0], 1)}, ${fmt(ci[1], 1)}]</b></div>
        <div class="row-kv"><span>95% 預測區間 (PI)</span><b>[${fmt(pi[0], 1)}, ${fmt(pi[1], 1)}]</b></div>
        ${actual != null ? `<div class="row-kv"><span>${esc(country)} ${m.presets[country].year} 實際值</span><b>${fmt(actual, 1)} ${m.unit}</b></div>` : ""}
      </div>
      <p class="muted" style="margin-top:10px">CI 描述「平均」值的不確定性；PI 描述「單一國家」實際值的可能範圍，所以 PI 一定比 CI 寬。</p>`;
  };

  const applyPreset = () => {
    const country = presetSel.value;
    metas.forEach((mt, i) => {
      const raw = country ? m.presets[country].values[mt.ft.name] : mt.ft.median;
      $(`#s${i}`).value = mt.tf(raw);
    });
    update();
  };
  presetSel.onchange = applyPreset;
  metas.forEach((_, i) => $(`#s${i}`).addEventListener("input", update));
  applyPreset();
}

/* ---------------------------------------------------------------- custom dataset */

const NA = new Set(["", "na", "nan", "null", "none", "?", "-"]);
const isNA = (v) => v == null || NA.has(String(v).trim().toLowerCase());

function parseCSV(text) {
  text = text.replace(/^﻿/, "");
  const rows = [];
  let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; } else q = false;
      } else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); cell = "";
      if (row.some((c) => c.trim() !== "")) rows.push(row);
      row = [];
    } else cell += ch;
  }
  row.push(cell);
  if (row.some((c) => c.trim() !== "")) rows.push(row);
  return rows;
}

function loadCSVText(text, sourceName) {
  const rows = parseCSV(text);
  if (rows.length < 3) throw new Error("CSV 內容太少（至少需要標題列與 2 筆資料）。");
  const header = rows[0].map((h, i) => h.trim() || `col${i + 1}`);
  const data = rows.slice(1).filter((r) => r.length >= header.length - 1);
  const columns = header.map((name, j) => {
    const vals = data.map((r) => r[j]).filter((v) => !isNA(v));
    const nums = vals.filter((v) => isFinite(Number(v)) && String(v).trim() !== "");
    const levels = [...new Set(vals.map((v) => String(v).trim()))].sort();
    let type = "skip";
    if (vals.length && nums.length / vals.length >= 0.95) type = levels.length <= 1 ? "skip" : "num";
    else if (levels.length >= 2 && levels.length <= 30) type = "cat";
    return { name, j, type, levels, missing: data.length - vals.length };
  });
  state.custom = { source: sourceName, header, data, columns };
  renderCustomPreview();
}

function renderCustomPreview() {
  const c = state.custom;
  const typeName = { num: "數值", cat: "類別 (one-hot)", skip: "略過" };
  $("#csv-status").className = "status";
  $("#csv-status").textContent = `已載入：${c.source} — ${c.data.length} 筆 × ${c.header.length} 欄`;
  $("#csv-preview").innerHTML = `<table>
    <thead><tr>${c.columns.map((col) => `<th>${esc(col.name)}<br><span class="muted">${typeName[col.type]}${col.missing ? ` · 缺 ${col.missing}` : ""}</span></th>`).join("")}</tr></thead>
    <tbody>${c.data.slice(0, 5).map((r) => `<tr>${c.columns.map((col) => `<td>${esc(r[col.j] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>
  </table>`;

  const numeric = c.columns.filter((col) => col.type === "num");
  if (!numeric.length) throw new Error("找不到數值欄位可當作目標變數。");
  const guess = numeric[numeric.length - 1];
  $("#target-select").innerHTML = numeric.map((col) => `<option value="${col.j}">${esc(col.name)}</option>`).join("");
  $("#target-select").value = guess.j;
  renderFeatureChecks();
  $("#custom-config").hidden = false;
  $("#custom-results").innerHTML = "";
  $("#backward-log").textContent = "";
  runCustom();
}

function renderFeatureChecks() {
  const c = state.custom;
  const target = Number($("#target-select").value);
  $("#feature-checks").innerHTML = c.columns
    .filter((col) => col.type !== "skip" && col.j !== target)
    .map((col) => `<label><input type="checkbox" value="${col.j}" checked> ${esc(col.name)} <span class="tag">${col.type === "cat" ? `類別·${col.levels.length}` : "數值"}</span></label>`)
    .join("");
}

const selectedFeatures = () => [...document.querySelectorAll("#feature-checks input:checked")].map((el) => Number(el.value));

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Build design matrix rows + term metadata; drop rows missing any used column. */
function buildDesign(targetJ, featureJs) {
  const c = state.custom;
  const cols = featureJs.map((j) => c.columns[j]);
  const terms = [];
  cols.forEach((col) => {
    if (col.type === "num") terms.push({ name: col.name, source: col.j });
    else col.levels.slice(1).forEach((lv) => terms.push({ name: `${col.name}=${lv}`, source: col.j, level: lv }));
  });
  const X = [], y = [], idx = [];
  c.data.forEach((r, i) => {
    if (isNA(r[targetJ]) || cols.some((col) => isNA(r[col.j]))) return;
    X.push(terms.map((t) => (t.level !== undefined ? (String(r[t.source]).trim() === t.level ? 1 : 0) : Number(r[t.source]))));
    y.push(Number(r[targetJ]));
    idx.push(i);
  });
  return { X, y, idx, terms };
}

function splitIdx(n, ratio, seed) {
  const order = Array.from({ length: n }, (_, i) => i);
  const rnd = mulberry32(seed);
  for (let i = n - 1; i > 0; i--) {
    const k = Math.floor(rnd() * (i + 1));
    [order[i], order[k]] = [order[k], order[i]];
  }
  const nTest = Math.max(1, Math.round(n * ratio));
  return { test: order.slice(0, nTest), train: order.slice(nTest) };
}

function fitCustom(featureJs) {
  const targetJ = Number($("#target-select").value);
  if (!featureJs.length) throw new Error("請至少選擇一個特徵。");
  const d = buildDesign(targetJ, featureJs);
  if (d.X.length < 5) throw new Error(`去除缺值後只剩 ${d.X.length} 筆，無法建模。`);
  const ratio = Math.min(0.5, Math.max(0.1, Number($("#test-ratio").value) || 0.2));
  const sp = splitIdx(d.X.length, ratio, Number($("#seed").value) || 42);
  const Xtr = sp.train.map((i) => d.X[i]), ytr = sp.train.map((i) => d.y[i]);
  const Xte = sp.test.map((i) => d.X[i]), yte = sp.test.map((i) => d.y[i]);
  const model = fitOLS(Xtr, ytr, d.terms.map((t) => t.name));
  return { d, sp, Xtr, ytr, Xte, yte, model, targetJ, featureJs };
}

function runCustom() {
  const out = $("#custom-results");
  try {
    const fit = fitCustom(selectedFeatures());
    state.customFit = fit;
    renderCustomResults(fit);
  } catch (e) {
    out.innerHTML = `<div class="card"><p class="status err">${esc(e.message)}</p></div>`;
  }
}

function backwardElimination() {
  let feats = selectedFeatures();
  const log = [];
  try {
    for (;;) {
      const fit = fitCustom(feats);
      const pBySource = {};
      fit.model.coefs.slice(1).forEach((cf, i) => {
        const src = fit.d.terms[i].source;
        pBySource[src] = Math.min(pBySource[src] ?? 1, cf.p);
      });
      const [worst, pWorst] = Object.entries(pBySource).sort((a, b) => b[1] - a[1])[0];
      if (pWorst <= 0.05 || feats.length <= 1) break;
      log.push(`移除 ${state.custom.columns[worst].name}（p = ${fmtP(pWorst)}）`);
      feats = feats.filter((j) => j !== Number(worst));
    }
  } catch (e) {
    $("#backward-log").textContent = e.message;
    return;
  }
  document.querySelectorAll("#feature-checks input").forEach((el) => { el.checked = feats.includes(Number(el.value)); });
  $("#backward-log").textContent = log.length ? `後向消去：${log.join(" → ")}；保留 ${feats.length} 個特徵。` : "所有特徵 p ≤ 0.05，不需移除。";
  runCustom();
}

function renderCustomResults(fit) {
  const { model, Xtr, ytr, Xte, yte, d } = fit;
  const c = state.custom;
  const targetName = c.columns[fit.targetJ].name;
  const k = d.terms.length;
  const trPred = Xtr.map((r) => model.predict(r));
  const tePred = Xte.map((r) => model.predict(r));
  const mTr = metrics(ytr, trPred.map((p) => p.yhat), k);
  const mTe = metrics(yte, tePred.map((p) => p.yhat), k);
  const coverage = yte.filter((v, i) => v >= tePred[i].pi[0] && v <= tePred[i].pi[1]).length / yte.length;
  const usedCols = [...new Set(fit.featureJs)].map((j) => c.columns[j]);

  $("#custom-results").innerHTML = `
    <div class="kpis">
      ${kpi("訓練集 R²", fmt(mTr.R2, 3), `Adj R² ${fmt(mTr.AdjR2, 3)}`)}
      ${kpi("測試集 R²", fmt(mTe.R2, 3), `Adj R² ${fmt(mTe.AdjR2, 3)}`)}
      ${kpi("測試集 RMSE", fmtRaw(mTe.RMSE), `MAE ${fmtRaw(mTe.MAE)}`)}
      ${kpi("測試集 MSE", fmtRaw(mTe.MSE), `n_train ${Xtr.length} / n_test ${Xte.length}`)}
      ${kpi("95% PI 覆蓋率", pct(coverage), `σ̂ = ${fmtRaw(Math.sqrt(model.s2))}`)}
    </div>
    <section class="card">
      <h2>預測圖（測試集）：95% 信賴區間與預測區間</h2>
      <div class="chart-box"><canvas id="cc-interval"></canvas></div>
    </section>
    <section class="grid-2">
      <div class="card"><h2>實際 vs 預測</h2><div class="chart-box"><canvas id="cc-avp"></canvas></div></div>
      <div class="card"><h2>殘差 vs 預測值（訓練集）</h2><div class="chart-box"><canvas id="cc-resid"></canvas></div></div>
    </section>
    <section class="card">
      <h2>OLS 係數（訓練集）</h2>
      <p class="muted">${esc(targetName)} = β₀ + Σ βᵢ·xᵢ；類別特徵以第一個水準為基準。df = ${model.df}，t₀.₉₇₅ = ${fmt(model.t, 3)}。</p>
      <div class="table-scroll"><table>
        <thead><tr><th>項目</th><th>係數</th><th>SE</th><th>t</th><th>p</th><th>95% CI</th></tr></thead>
        <tbody>${model.coefs.map((cf) => `<tr><td>${cf.name === "const" ? "截距" : esc(cf.name)}</td><td>${fmtRaw(cf.coef)}</td><td>${fmtRaw(cf.se)}</td>
          <td>${fmt(cf.t, 2)}</td><td>${fmtP(cf.p)}</td><td>[${fmtRaw(cf.coef - model.t * cf.se)}, ${fmtRaw(cf.coef + model.t * cf.se)}]</td></tr>`).join("")}</tbody>
      </table></div>
    </section>
    <section class="card">
      <h2>新樣本預測</h2>
      <div class="inputs-grid">${usedCols.map((col) => {
        const vals = c.data.map((r) => r[col.j]).filter((v) => !isNA(v));
        if (col.type === "cat") return `<label>${esc(col.name)}<select data-j="${col.j}">${col.levels.map((l) => `<option>${esc(l)}</option>`).join("")}</select></label>`;
        const nums = vals.map(Number).sort((a, b) => a - b);
        const med = nums[Math.floor(nums.length / 2)];
        return `<label>${esc(col.name)}<input type="number" step="any" data-j="${col.j}" value="${med}"></label>`;
      }).join("")}</div>
      <div class="row"><button id="btn-predict" class="btn primary">預測</button><span id="custom-pred" class="status"></span></div>
    </section>`;

  const toPts = (pred, y) => pred.map((p, i) => ({ pred: p.yhat, actual: y[i], ci: p.ci, pi: p.pi }));
  intervalChart("cc-interval", toPts(tePred, yte), targetName);
  avpChart("cc-avp", [{ label: "訓練集", points: toPts(trPred, ytr) }, { label: "測試集", points: toPts(tePred, yte) }], targetName);
  residualChart("cc-resid", model.fitted, model.resid);

  $("#btn-predict").onclick = () => {
    const inputs = {};
    document.querySelectorAll("#custom-results [data-j]").forEach((el) => { inputs[el.dataset.j] = el.value; });
    const row = d.terms.map((t) => (t.level !== undefined ? (String(inputs[t.source]).trim() === t.level ? 1 : 0) : Number(inputs[t.source])));
    if (row.some((v) => !isFinite(v))) {
      $("#custom-pred").textContent = "請輸入有效數值。";
      return;
    }
    const p = model.predict(row);
    $("#custom-pred").innerHTML = `預測 ${esc(targetName)} = <b>${fmtRaw(p.yhat)}</b>　95% CI [${fmtRaw(p.ci[0])}, ${fmtRaw(p.ci[1])}]　95% PI [${fmtRaw(p.pi[0])}, ${fmtRaw(p.pi[1])}]`;
  };
}

async function loadFromURL(url, name) {
  $("#csv-status").className = "status";
  $("#csv-status").textContent = "下載中…";
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadCSVText(await res.text(), name || url);
  } catch (e) {
    $("#csv-status").className = "status err";
    $("#csv-status").textContent = `載入失敗：${e.message}（請確認網址是 CSV 原始檔，且該網站允許跨網域讀取）`;
  }
}

/* ---------------------------------------------------------------- init */

function showView(value) {
  const custom = value === "custom";
  $("#problem-view").hidden = custom;
  $("#custom-view").hidden = !custom;
  if (custom) {
    if (!state.custom) loadFromURL("data/50_Startups.csv", "50_Startups.csv（範例）");
  } else {
    loadProblem(value);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const select = $("#problem-select");
  const fromHash = location.hash.slice(1);
  if ([...select.options].some((o) => o.value === fromHash)) select.value = fromHash;
  select.addEventListener("change", () => {
    history.replaceState(null, "", `#${select.value}`);
    showView(select.value);
  });

  $("#btn-load-url").onclick = () => loadFromURL($("#csv-url").value.trim());
  $("#btn-load-sample").onclick = () => loadFromURL("data/50_Startups.csv", "50_Startups.csv（範例）");
  $("#btn-load-who").onclick = () => loadFromURL("data/Life Expectancy Data.csv", "Life Expectancy Data.csv（WHO）");
  $("#csv-file").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      loadCSVText(await file.text(), file.name);
    } catch (err) {
      $("#csv-status").className = "status err";
      $("#csv-status").textContent = err.message;
    }
  };
  $("#target-select").onchange = () => { renderFeatureChecks(); runCustom(); };
  $("#btn-run").onclick = runCustom;
  $("#btn-backward").onclick = backwardElimination;
  $("#btn-all").onclick = () => document.querySelectorAll("#feature-checks input").forEach((el) => { el.checked = true; });
  $("#btn-none").onclick = () => document.querySelectorAll("#feature-checks input").forEach((el) => { el.checked = false; });

  showView(select.value);
});
