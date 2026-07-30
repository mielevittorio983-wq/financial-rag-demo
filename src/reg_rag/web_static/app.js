const state = { answer: null, documents: null, history: [], presets: [], demoMode: false, publicMode: false };
const views = {
  qa: ["TRUSTED REGULATORY WORKSPACE", "智能问答"],
  evidence: ["EVIDENCE-GROUNDED ANSWER", "证据查看"],
  knowledge: ["REGULATORY KNOWLEDGE OPERATIONS", "知识库管理"],
  audit: ["PARSER TRANSPARENCY", "解析审计"],
  health: ["RUNTIME OBSERVABILITY", "系统运行状态"],
  evaluation: ["COMPETITION READINESS", "评估与演示"],
};

const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
}[char]));

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
$("submitQuestion").addEventListener("click", () => submitQuestion());
$("clearHistory").addEventListener("click", () => { state.history = []; renderHistory(); });
$("refreshDocuments").addEventListener("click", loadDocuments);
$("fileInput").addEventListener("change", (event) => uploadFile(event.target.files[0]));
document.querySelectorAll("[data-capability]").forEach((button) => button.addEventListener("click", () => fillCapability(button.dataset.capability)));

function switchView(name) {
  if (!views[name] || !$(`view-${name}`)) return;
  document.querySelectorAll(".nav-item").forEach((item) => {
    const active = item.dataset.view === name;
    item.classList.toggle("active", active);
    item.setAttribute("aria-current", active ? "page" : "false");
  });
  document.querySelectorAll(".view").forEach((item) => item.classList.remove("active"));
  $(`view-${name}`).classList.add("active");
  $("viewEyebrow").textContent = views[name][0];
  $("viewTitle").textContent = views[name][1];
  document.body.dataset.currentView = name;
  if (name === "knowledge") loadDocuments();
  if (name === "health") loadHealth();
  if (name === "evaluation") loadMetrics();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function submitQuestion(targetView = null) {
  const question = $("question").value.trim();
  if (!question) return toast("请输入问题");
  $("answerState").className = "answer-state loading";
  $("answerState").innerHTML = "<strong>正在检索与核验</strong><span>确定性 Pipeline 正在运行</span>";
  $("answerPanel").classList.add("hidden");
  $("submitQuestion").disabled = true;
  try {
    const result = await api("/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, requested_mode: $("requestedMode").value }),
    });
    state.answer = result;
    state.history.unshift({ question, mode: $("requestedMode").value });
    state.history = state.history.slice(0, 8);
    renderHistory();
    renderAnswer(result);
    renderEvidence(result);
    if (targetView) switchView(targetView);
    return result;
  } catch (error) {
    $("answerState").className = "answer-state";
    $("answerState").innerHTML = `<strong>查询失败</strong><span>${escapeHtml(error.message)}</span>`;
  } finally {
    $("submitQuestion").disabled = false;
  }
}

function renderAnswer(result) {
  $("answerState").classList.add("hidden");
  $("answerPanel").classList.remove("hidden");
  const refused = result.status !== "ANSWERED";
  $("trustBadge").textContent = refused ? "安全拒答" : "可信回答";
  $("trustBadge").className = `pill ${refused ? "warning" : "success"}`;
  $("answerType").textContent = result.answer_type;
  $("answerText").textContent = result.answer || "未生成事实答案";
  $("latency").textContent = `${result.latency_ms.toFixed(1)} ms`;
  $("pipeline").textContent = result.pipeline;
  $("answerStatus").textContent = result.status;
  $("confidence").textContent = `${Math.round(result.confidence * 100)}%`;
  $("evidenceCount").textContent = `${result.evidence.length} 条`;
  $("fallback").textContent = result.fallback ? "是" : "否";
  $("refusalReason").classList.toggle("hidden", !refused);
  if (refused) renderRefusalGuidance(result);
  $("traceStages").innerHTML = result.trace_summary.stages.map((stage) => `
    <div class="trace-stage"><strong>${escapeHtml(stage.name)}</strong><small>${escapeHtml(stage.status)} · ${stage.duration_ms.toFixed(1)} ms</small></div>
  `).join("");
  $("llmMode").textContent = result.llm_mode === "SHADOW" ? "LLM 影子模式" : `LLM ${result.llm_mode}`;
  $("systemMode").textContent = result.llm_available ? "LLM 服务可用" : "确定性引擎运行中";
}

function renderRefusalGuidance(result) {
  const accountLevel = /账户|客户|余额/.test(result.question);
  $("refusalTitle").textContent = "当前证据不足";
  $("refusalMessage").textContent = accountLevel
    ? "拒答原因：资料库未包含经授权、可核验的账户级数据"
    : "拒答原因：当前来源不足以支持唯一事实结论";
  $("refusalMissing").textContent = accountLevel ? "缺少账户级授权数据与可核验期间" : "缺少可核验来源、期间或唯一指标定义";
  $("refusalNext").textContent = accountLevel ? "补充经授权的账户数据、主体与查询期间" : "补充来源文件、统计期间和明确指标口径";
}

function renderHistory() {
  $("historyList").innerHTML = state.history.length ? state.history.map((item, index) => `
    <button class="history-item" data-history="${index}" title="${escapeHtml(item.question)}">${escapeHtml(item.question)}</button>
  `).join("") : '<p class="empty">暂无查询记录</p>';
  document.querySelectorAll("[data-history]").forEach((button) => button.addEventListener("click", () => {
    const item = state.history[Number(button.dataset.history)];
    $("question").value = item.question;
    $("requestedMode").value = item.mode;
  }));
}

function renderEvidence(result) {
  $("evidenceSummary").textContent = `${result.evidence.length} 条证据`;
  $("evidenceAnswerSummary").classList.remove("hidden");
  $("evidenceAnswerSummary").innerHTML = `
    <div class="evidence-final">
      <span class="section-label">FINAL ANSWER</span>
      <strong>${escapeHtml(result.answer || "未生成事实答案")}</strong>
      <span class="pill ${result.status === "ANSWERED" ? "success" : "warning"}">${result.status === "ANSWERED" ? "EVIDENCE GROUNDED" : "SAFE DECISION"}</span>
    </div>
    <dl>
      <div><dt>Pipeline</dt><dd>${escapeHtml(result.pipeline)}</dd></div>
      <div><dt>Evidence</dt><dd>${result.evidence.length} 条</dd></div>
      <div><dt>Confidence</dt><dd>${Math.round(result.confidence * 100)}%</dd></div>
      <div><dt>Fallback</dt><dd>${result.fallback ? "ACTIVE" : "NOT TRIGGERED"}</dd></div>
    </dl>
  `;
  renderCrossFileChain(result);
  $("evidenceList").innerHTML = result.evidence.length ? result.evidence.map((item, index) => `
    <details class="evidence-item" ${index === 0 ? "open" : ""}>
      <summary>
        <div class="evidence-index"><span>${String(index + 1).padStart(2, "0")}</span><div><strong>${escapeHtml(item.source_title || item.doc_id)}</strong><small>${escapeHtml(item.doc_id)}</small></div></div>
        <span class="pill neutral">${escapeHtml(item.coordinate || item.clause || "TEXT EVIDENCE")}</span>
      </summary>
      <div class="evidence-body">
        <div class="inspector-grid">
          <section class="inspector-group">
            <header><span>01</span><div><strong>Data Coordinate</strong><small>数据坐标</small></div></header>
            <dl>
              <div><dt>Workbook / 文件</dt><dd>${escapeHtml(item.source_title || item.doc_id)}</dd></div>
              <div><dt>Sheet</dt><dd>${escapeHtml(item.sheet_name || "—")}</dd></div>
              <div class="coordinate-field"><dt>目标 Cell</dt><dd>${escapeHtml(item.coordinate || item.clause || "—")}</dd></div>
              <div><dt>Table</dt><dd>${escapeHtml(item.table_id || "—")}</dd></div>
            </dl>
          </section>
          <section class="inspector-group">
            <header><span>02</span><div><strong>Semantic Coordinate</strong><small>语义坐标</small></div></header>
            <dl>
              <div><dt>指标 Metric</dt><dd>${escapeHtml((item.column_header_path || []).at(-1) || item.clause || "—")}</dd></div>
              <div><dt>Row Header Path</dt><dd>${escapeHtml((item.row_header_path || []).join(" / ") || "—")}</dd></div>
              <div><dt>Column Header Path</dt><dd>${escapeHtml((item.column_header_path || []).join(" / ") || "—")}</dd></div>
              <div><dt>期间 · 单位</dt><dd>${escapeHtml(item.period || "—")} · ${escapeHtml(item.unit || "—")}</dd></div>
            </dl>
          </section>
          <section class="inspector-group extracted-group">
            <header><span>03</span><div><strong>Extracted Evidence</strong><small>提取证据</small></div></header>
            <dl>
              <div class="value-field"><dt>Value</dt><dd>${escapeHtml(item.display_value ?? item.raw_value ?? "—")}</dd></div>
              <div><dt>Evidence Confidence</dt><dd>${item.evidence_confidence == null ? "—" : `${Math.round(item.evidence_confidence * 100)}%`}</dd></div>
              <div><dt>发布机构</dt><dd>${escapeHtml(item.issuer || "—")}</dd></div>
              <div><dt>发布日期</dt><dd>${escapeHtml(item.publish_date || "—")}</dd></div>
            </dl>
          </section>
        </div>
        <div class="evidence-label">ORIGINAL CELL CONTENT / 原文或原单元格内容</div>
        <div class="evidence-text">${escapeHtml(item.text || item.display_value || item.raw_value || "")}</div>
        <div class="semantic-path"><strong>Semantic Path</strong><span>${escapeHtml((item.semantic_path || item.column_header_path || []).join(" / ") || "—")}</span></div>
      </div>
    </details>
  `).join("") : '<div class="empty-state"><strong>没有可展示证据</strong><span>系统已安全拒答或未定位到来源</span></div>';
}

function renderCrossFileChain(result) {
  const chain = result.trace_summary.evidence_chain;
  const bridges = result.trace_summary.metric_bridges || [];
  if (!chain && !bridges.length) {
    $("crossFileChain").classList.add("hidden");
    return;
  }
  const steps = chain?.steps || [];
  const definitions = steps.filter((item) => item.purpose === "definition_and_metric_bridge");
  const values = steps.filter((item) => item.purpose === "table_value_retrieval");
  const final = steps.find((item) => item.purpose === "absolute_difference");
  const stages = [
    {
      label: "STAGE 01",
      title: "监管定义",
      reason: "确定制度口径",
      items: definitions.map((item) => ({ source: item.source, claim: item.claim, coordinate: item.coordinate, confidence: item.confidence })),
    },
    {
      label: "STAGE 02",
      title: "指标映射 / Metric Bridge",
      reason: "监管指标名称标准化",
      items: bridges.map((item) => ({ source: item.selected?.target_doc, claim: `${item.source_metric} → ${item.selected?.target_metric}`, coordinate: item.selected?.target_coordinate_hint, confidence: item.confidence })),
    },
    {
      label: "STAGE 03",
      title: "报表证据 / 统计附件与表格证据",
      reason: "期间与单位一致",
      items: values.map((item) => ({ source: item.source, claim: item.claim, coordinate: item.coordinate, confidence: item.confidence })),
    },
    {
      label: "STAGE 04",
      title: "确定性计算结果",
      reason: "确定性执行器计算",
      items: [{ source: "deterministic_calculator", claim: `${result.answer} · ${final?.claim || "deterministic operation"}`, coordinate: final?.coordinate || "derived:answer", confidence: final?.confidence ?? result.confidence }],
    },
  ];
  $("crossFileChain").classList.remove("hidden");
  $("crossFileChain").innerHTML = `
    <div class="journey-heading">
      <div><span class="section-label">EVIDENCE JOURNEY</span><h2>跨文件证据旅程</h2></div>
      <span>最终数值由确定性执行器计算，LLM 未参与最终计算</span>
    </div>
    <div class="journey-track">
    ${stages.map((stage) => `
      <article class="chain-step">
      <div class="chain-step-title"><small>${stage.label}</small><strong>${stage.title}</strong><span>${stage.reason}</span></div>
      ${stage.items.map((item) => `
        <div class="chain-evidence">
          <b title="${escapeHtml(item.source || "—")}">${escapeHtml(item.source || "—")}</b>
          <span>${escapeHtml(item.claim || "—")}</span>
          <footer><code>${escapeHtml(item.coordinate || "—")}</code><em>${item.confidence == null ? "—" : `${Math.round(item.confidence * 100)}%`}</em></footer>
        </div>
      `).join("")}
      </article>
    `).join("")}
    </div>`;
}

async function loadDocuments() {
  try {
    state.documents = await api("/documents");
    const summary = state.documents.summary;
    $("documentStats").innerHTML = [
      ["Documents", summary.total, "总文档"],
      ["Official", summary.official, "官方语料"],
      ["User Uploaded", summary.user_uploaded, "用户上传"],
      ["Indexed", summary.indexed, "已索引"],
    ].map(([label, value, note]) => `<div class="stat"><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join("") +
      `<div class="type-distribution"><span>FILE TYPE DISTRIBUTION</span><b>PDF ${summary.pdf}</b><b>DOCX ${summary.docx}</b><b>XLSX ${summary.xlsx}</b><small>最近更新 ${escapeHtml(formatDateTime(summary.recent_updated))}</small></div>`;
    $("documentRows").innerHTML = state.documents.documents.map((item) => `
      <tr><td><span class="origin">${escapeHtml(item.origin)}</span></td><td>${escapeHtml(item.title)}</td><td>${escapeHtml(item.file_type || "—")}</td><td><span class="pill ${item.status === "INDEXED" ? "success" : item.status === "FAILED" ? "danger" : "warning"}">${escapeHtml(item.status)}</span></td>
      <td><div class="row-actions">${item.origin === "USER_UPLOADED" ? `<button data-audit="${item.doc_id}">审计</button><button data-index="${item.doc_id}">重建索引</button><button data-delete="${item.doc_id}">删除</button>` : "只读"}</div></td></tr>
    `).join("");
    document.querySelectorAll("[data-audit]").forEach((button) => button.addEventListener("click", () => loadAudit(button.dataset.audit)));
    document.querySelectorAll("[data-index]").forEach((button) => button.addEventListener("click", () => indexDocument(button.dataset.index)));
    document.querySelectorAll("[data-delete]").forEach((button) => button.addEventListener("click", () => deleteDocument(button.dataset.delete)));
  } catch (error) { toast(error.message); }
}

async function uploadFile(file) {
  if (!file) return;
  if (state.publicMode) return toast("公网演示使用隔离预置文档");
  $("uploadProgress").classList.remove("hidden");
  $("uploadMessage").textContent = "上传、解析并索引中";
  try {
    const result = await api("/documents/upload", { method: "POST", headers: { "X-Filename": encodeURIComponent(file.name), "Content-Type": "application/octet-stream" }, body: file });
    $("uploadMessage").textContent = result.document.status === "INDEXED" ? `完成：${result.document.record_count} 条解析记录` : `失败：${result.document.error}`;
    await loadDocuments();
  } catch (error) { $("uploadMessage").textContent = error.message; }
  finally { $("uploadProgress").classList.add("hidden"); $("fileInput").value = ""; }
}

async function loadAudit(docId) {
  try {
    const audit = await api(`/documents/${encodeURIComponent(docId)}`);
    switchView("audit");
    $("auditDocBadge").textContent = audit.title;
    $("auditSummary").innerHTML = [["记录", audit.record_count], ["文本块", audit.chunk_count], ["表格单元格", audit.table_cell_count], ["状态", audit.status]].map(([label, value]) => `<div class="stat"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
    $("auditContent").innerHTML = audit.samples.map((item) => `<div class="audit-row"><span class="pill neutral">${escapeHtml(item.kind)}</span><code>${escapeHtml(item.coordinate || item.locator)}</code><div>${escapeHtml(item.text || "")}<small>${escapeHtml((item.header_path || []).join(" / "))}</small></div><span>${escapeHtml(item.unit || item.period || "—")}</span></div>`).join("") || '<div class="empty-state"><strong>暂无解析记录</strong></div>';
  } catch (error) { toast(error.message); }
}

async function indexDocument(docId) {
  try { await api(`/documents/${encodeURIComponent(docId)}/index`, { method: "POST" }); toast("索引已重建"); loadDocuments(); } catch (error) { toast(error.message); }
}
async function deleteDocument(docId) {
  if (!confirm("删除该用户上传文档？")) return;
  try { await api(`/documents/${encodeURIComponent(docId)}`, { method: "DELETE" }); toast("文档已删除"); loadDocuments(); } catch (error) { toast(error.message); }
}

async function loadHealth() {
  try {
    const health = await api("/health");
    state.publicMode = health.public_mode === true;
    document.body.classList.toggle("public-mode", state.publicMode);
    if (state.publicMode) {
      $("fileInput").disabled = true;
      $("dropZone").classList.add("public-disabled");
      $("dropZone").querySelector("strong").textContent = "公网演示使用隔离预置文档";
      $("dropZone").querySelector("span:last-child").textContent = "匿名上传已关闭 · 请运行“新文档泛化能力”演示";
      $("uploadMessage").textContent = "官方知识库只读，预置新文档在独立沙箱内解析与问答";
      $("llmMode").textContent = "LLM 已禁用";
      $("llmMode").parentElement.querySelector("small").textContent = "LLM DISABLED";
    }
    $("sidebarHealth").textContent = health.status;
    $("sidebarLlm").textContent = health.ok ? "确定性 RAG 运行中" : "需要检查";
    $("healthDot").classList.toggle("ok", health.ok);
    const llmNotice = /unavailable|disabled/i.test(health.llm_notice) ? "LLM 已禁用，确定性 RAG 运行中" : "LLM 服务可用";
    const publicRelease = health.release || "RC1";
    $("healthSummary").innerHTML = `
      <div class="health-state"><span class="status-dot ok"></span><div><small>SYSTEM HEALTH</small><strong>${escapeHtml(health.status)}</strong><p>${escapeHtml(health.system)}</p></div></div>
      <dl><div><dt>Release</dt><dd>${escapeHtml(publicRelease)}</dd></div><div><dt>Mode</dt><dd>${state.publicMode ? "PUBLIC DEMO" : `v${escapeHtml(health.version)}`}</dd></div><div><dt>Runtime</dt><dd>${state.publicMode ? "READ ONLY" : escapeHtml(health.git_commit)}</dd></div><div><dt>Fallback</dt><dd>DETERMINISTIC ACTIVE</dd></div></dl>
      <span class="health-notice">${escapeHtml(llmNotice)}</span>`;
    $("healthGrid").innerHTML = health.checks.map((item) => {
      const statusClass = ["PASS", "READY_GATED", "READY_SHADOW"].includes(item.status) ? "success" : item.status === "DISABLED" ? "neutral" : "warning";
      const role = /LLM/.test(item.component) ? "Optional Intelligence" : /Index|Manifest|Cache/.test(item.component) ? "Data Foundation" : "Core Runtime";
      return `<article class="health-item"><header><div><small>${escapeHtml(role)}</small><strong>${escapeHtml(item.component)}</strong></div><span class="pill ${statusClass}">${escapeHtml(item.status)}</span></header><p>${escapeHtml(item.message)}</p></article>`;
    }).join("");
  } catch (error) { $("sidebarHealth").textContent = "ERROR"; toast(error.message); }
}

async function loadMetrics() {
  try {
    const [metrics, demos] = await Promise.all([api("/metrics/summary"), api("/demo/presets")]);
    state.presets = demos.presets;
    const primary = [
      ["KNOWN", "99.67%", "299 / 300", "hero"],
      ["Evidence Support", "99.67%", "299 / 300", "hero"],
      ["Critical Field Error", "0.38%", "1 / 262 · LOWER IS BETTER", "risk"],
    ];
    const capability = [
      ["Regulatory Fact", "100%", "O1 · 200 / 200"],
      ["Table Lookup", "99%", "O2 · 99 / 100"],
      ["Insufficient Evidence", "90%", "O5 · 18 / 20"],
      ["Cross-file", `${metrics.cross_file.final_answer}/20`, `Evidence Chain ${metrics.cross_file.evidence_chain}/20`],
    ];
    const internal = [
      ["Internal Stress", "27/30", "高难内部压力集"],
      ["Core Table", `${metrics.core_28.M2}/28`, "高难专项审计集，不代表总体表格准确率"],
      ["Dispatcher", `${metrics.specialist_switch.correct}/${metrics.specialist_switch.total}`, "Specialist Switch"],
    ];
    $("metricGrid").innerHTML = primary.map(([label, value, note, kind]) => `<article class="metric metric-${kind}"><span>${label}</span><strong>${value}</strong><small>${note}</small><i></i></article>`).join("");
    $("capabilityMetrics").innerHTML = `<div class="metric-section-head"><div><span>02</span><strong>能力覆盖</strong></div><small>O1 · O2 · O5 · CROSS-FILE</small></div><div class="capability-metric-grid">${capability.map(([label, value, note]) => `<article class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("")}</div>`;
    $("internalMetrics").innerHTML = `<div class="metric-section-head"><div><span>03</span><strong>内部鲁棒性审计</strong></div><small>INTERNAL ROBUSTNESS · NOT OFFICIAL LEADERBOARD</small></div><div class="internal-metric-grid">${internal.map(([label, value, note]) => `<article class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("")}</div>`;
    $("demoPresets").innerHTML = demos.presets.map((item, index) => `<button class="preset" data-preset="${escapeHtml(item.id)}"><span class="preset-number">0${index + 1}</span><div><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.description)}</small></div><b>→</b></button>`).join("");
    document.querySelectorAll("[data-preset]").forEach((button) => button.addEventListener("click", () => {
      const preset = demos.presets.find((item) => item.id === button.dataset.preset);
      runPreset(preset);
    }));
    return demos.presets;
  } catch (error) { toast(error.message); }
}

async function fillCapability(presetId) {
  if (!state.presets.length) await loadMetrics();
  const preset = state.presets.find((item) => item.id === presetId);
  if (!preset) return;
  $("question").value = preset.question;
  $("requestedMode").value = preset.mode;
  switchView("qa");
  $("question").focus();
}

async function runPreset(preset, targetView = null) {
  if (!preset) return;
  $("question").value = preset.question;
  $("requestedMode").value = preset.mode;
  switchView("qa");
  if (preset.id === "new_document") return runNewDocumentDemo(preset, targetView);
  return submitQuestion(targetView);
}

async function runNewDocumentDemo(preset, targetView = null) {
  $("answerState").className = "answer-state loading";
  $("answerState").innerHTML = "<strong>正在解析隔离预置文档</strong><span>真实解析、索引与证据检索正在运行</span>";
  $("answerPanel").classList.add("hidden");
  $("submitQuestion").disabled = true;
  try {
    const result = await api("/demo/new-document", { method: "POST" });
    state.answer = result;
    state.history.unshift({ question: preset.question, mode: preset.mode });
    state.history = state.history.slice(0, 8);
    renderHistory();
    renderAnswer(result);
    renderEvidence(result);
    if (targetView) switchView(targetView);
    return result;
  } catch (error) {
    $("answerState").className = "answer-state";
    $("answerState").innerHTML = `<strong>演示失败</strong><span>${escapeHtml(error.message)}</span>`;
  } finally {
    $("submitQuestion").disabled = false;
  }
}

function formatDateTime(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? String(value) : parsed.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function toast(message) {
  $("toast").textContent = message;
  $("toast").classList.add("show");
  setTimeout(() => $("toast").classList.remove("show"), 2600);
}

async function initialize() {
  const params = new URLSearchParams(window.location.search);
  state.demoMode = params.get("demo") === "1";
  document.body.classList.toggle("demo-mode", state.demoMode);
  await Promise.all([loadHealth(), loadMetrics()]);
  const targetView = params.get("view");
  const presetId = params.get("preset");
  if (presetId) {
    const preset = state.presets.find((item) => item.id === presetId);
    await runPreset(preset, targetView || "qa");
  } else {
    switchView(targetView || "qa");
  }
  document.body.dataset.captureReady = "true";
}

initialize();
