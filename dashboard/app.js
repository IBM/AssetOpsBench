const state = {
  scenarios: [],
  results: [],
  search: "",
  filters: { category: new Set(), type: new Set(), asset: new Set(), agent: new Set() },
  expanded: new Set(),
  apiAvailable: false
};

const byId = (id) => document.getElementById(id);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[character]);
const numberFormat = new Intl.NumberFormat();

async function api(path, options = {}) {
  const response = await fetch(`${window.ASSETOPSBENCH_API_BASE || ""}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers }
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Dashboard API returned ${response.status}.`);
  return payload;
}

function setBackendStatus(label, ready = false) {
  const status = byId("backend-status");
  status.innerHTML = `<span class="live-dot"></span> ${escapeHTML(label)}`;
  status.classList.toggle("backend-ready", ready);
}

function uniqueValues(field) {
  return [...new Set(state.scenarios.flatMap((scenario) => {
    const value = scenario[field];
    return Array.isArray(value) ? value : value ? [value] : [];
  }))].sort((a, b) => a.localeCompare(b));
}

function renderFilters(id, values, key) {
  const container = byId(id);
  container.innerHTML = values.map((value) => `<button class="filter-chip" type="button" aria-pressed="${state.filters[key].has(value)}" data-filter="${key}" data-value="${escapeHTML(value)}">${escapeHTML(value)}</button>`).join("");
  const empty = byId(`${key}-filter-empty`);
  if (empty) empty.hidden = values.length > 0;
}

function renderScenarioOptions() {
  const options = state.scenarios.map((scenario) => `<option value="${escapeHTML(scenario.id)}">${escapeHTML(scenario.id)} · ${escapeHTML(scenario.title)}</option>`).join("");
  byId("scenario-run-select").innerHTML = options || '<option value="">No local scenario files found</option>';
  const judgeScenarios = state.scenarios.filter((scenario) => scenario.characteristic_form);
  byId("evaluation-scenario").innerHTML = judgeScenarios.length
    ? judgeScenarios.map((scenario) => `<option value="${escapeHTML(scenario.id)}">${escapeHTML(scenario.id)} · ${escapeHTML(scenario.title)}</option>`).join("")
    : '<option value="">No scenarios include an evaluation reference</option>';
  byId("run-scenario").disabled = !state.apiAvailable || !state.scenarios.length;
  byId("repo-scenario-count").textContent = numberFormat.format(state.scenarios.length);
}

function renderScenarios() {
  const query = state.search.trim().toLowerCase();
  const filtered = state.scenarios.filter((scenario) => {
    const searchable = [scenario.title, scenario.text, scenario.category, scenario.type, scenario.source, ...(scenario.assets || []), ...(scenario.agents || [])].join(" ").toLowerCase();
    return searchable.includes(query)
      && (!state.filters.category.size || state.filters.category.has(scenario.category))
      && (!state.filters.type.size || state.filters.type.has(scenario.type))
      && (!state.filters.asset.size || (scenario.assets || []).some((asset) => state.filters.asset.has(asset)))
      && (!state.filters.agent.size || (scenario.agents || []).some((agent) => state.filters.agent.has(agent)));
  });
  if (byId("scenario-sort").value === "title") filtered.sort((a, b) => a.title.localeCompare(b.title));

  byId("scenario-list").innerHTML = filtered.length ? filtered.map((scenario) => {
    const expanded = state.expanded.has(scenario.id);
    const tags = [
      scenario.category && `<span class="scenario-tag">${escapeHTML(scenario.category)}</span>`,
      scenario.type && `<span class="scenario-tag scenario-type">${escapeHTML(scenario.type)}</span>`,
      ...(scenario.agents || []).map((agent) => `<span class="scenario-tag agent-tag">${escapeHTML(agent)}</span>`)
    ].filter(Boolean).join("");
    return `<article class="scenario-card${expanded ? " expanded" : ""}">
      <button class="scenario-toggle" type="button" aria-expanded="${expanded}" aria-controls="scenario-detail-${escapeHTML(scenario.id)}" data-scenario="${escapeHTML(scenario.id)}">
        <span class="scenario-title-area"><strong>${escapeHTML(scenario.title)}</strong><span class="scenario-tags">${tags || '<span class="scenario-tag">Repository scenario</span>'}</span></span>
        <span class="scenario-chevron" aria-hidden="true">⌄</span>
      </button>
      <div class="scenario-details" id="scenario-detail-${escapeHTML(scenario.id)}" ${expanded ? "" : "aria-hidden=\"true\""}><div class="scenario-details-inner"><div class="scenario-detail-content">
        <div class="scenario-detail-grid"><div><span class="detail-label">REPOSITORY TASK</span><p class="detail-text">${escapeHTML(scenario.text)}</p></div><div><span class="detail-label">SOURCE FILE</span><p class="detail-text"><code>${escapeHTML(scenario.source)}</code></p>${scenario.agents?.length ? `<span class="detail-label detail-label-spaced">DECLARED AGENTS</span><div class="agent-pills">${scenario.agents.map((agent) => `<span class="agent-pill">${escapeHTML(agent)}</span>`).join("")}</div>` : ""}</div></div>
        <button class="scenario-explain-button" type="button" data-run-id="${escapeHTML(scenario.id)}">Run this repository scenario ↗</button>
      </div></div></div>
    </article>`;
  }).join("") : `<div class="scenario-empty">${state.scenarios.length ? "No repository scenarios match those filters." : "No scenario definitions were found in src/scenarios or src/couchdb/scenarios_data."}</div>`;
  byId("scenario-count").textContent = `${filtered.length} of ${state.scenarios.length} repository scenarios`;
  byId("scenario-summary").textContent = `Loaded ${state.scenarios.length} definitions directly from repository files · showing ${filtered.length}`;
}

function refreshScenarioView() {
  renderFilters("category-filters", uniqueValues("category"), "category");
  renderFilters("type-filters", uniqueValues("type"), "type");
  renderFilters("asset-filters", uniqueValues("assets"), "asset");
  renderFilters("agent-filters", uniqueValues("agents"), "agent");
  renderScenarios();
}

function renderTelemetryChart(values) {
  const chart = byId("telemetry-chart");
  if (!values?.length) {
    chart.innerHTML = '<path class="chart-stroke" d="M0 50 L480 50"/>';
    chart.setAttribute("aria-label", "No numeric sensor readings were found in the checked-in telemetry files.");
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values.map((value, index) => {
    const x = values.length === 1 ? 240 : index * 480 / (values.length - 1);
    const y = 90 - ((value - min) / range) * 76;
    return `${x.toFixed(1)} ${y.toFixed(1)}`;
  });
  chart.innerHTML = `<path class="chart-stroke" d="M${points.join(" L")}"/>`;
  chart.setAttribute("aria-label", `Sensor series read from repository data, with ${values.length} sampled values ranging from ${min.toPrecision(4)} to ${max.toPrecision(4)}.`);
}

function renderOverview(data) {
  byId("telemetry-count").textContent = numberFormat.format(data.telemetry.observations);
  byId("telemetry-caption").textContent = `numeric sensor observations · ${numberFormat.format(data.telemetry.records)} records`;
  byId("telemetry-source").textContent = data.telemetry.sources.map((path) => path.split("/").pop()).join(", ") || "No source files";
  renderTelemetryChart(data.telemetry.series);

  byId("fmea-count").textContent = numberFormat.format(data.fmea.failure_modes);
  byId("fmea-caption").textContent = `failure modes in ${numberFormat.format(data.fmea.records)} source records`;
  byId("fmea-names").textContent = data.fmea.names.join(" · ") || "No failure modes are present in the referenced local sample.";
  byId("fmea-source").textContent = data.fmea.sources.map((path) => path.split("/").pop()).join(", ") || "No source files";

  byId("workorder-count").textContent = numberFormat.format(data.workorders.records);
  const statusEntries = Object.entries(data.workorders.statuses);
  byId("workorder-statuses").innerHTML = statusEntries.length
    ? statusEntries.map(([status, count], index) => `<div><span class="state-dot ${["state-open", "state-progress", "state-done"][index % 3]}"></span><strong>${numberFormat.format(count)}</strong><small>${escapeHTML(status)}</small></div>`).join("")
    : "<span>No records in the repository work-order sample.</span>";
  byId("workorder-source").textContent = data.workorders.sources.map((path) => path.split("/").pop()).join(", ") || "No source files";

  const rate = data.kpis.completion_rate;
  byId("kpi-count").textContent = rate === null ? "—" : `${rate}%`;
  byId("kpi-caption").textContent = `completed work orders · ${data.kpis.completed} of ${data.kpis.total}`;
  byId("kpi-ring").style.background = rate === null
    ? "conic-gradient(#26332f 0 100%)"
    : `conic-gradient(var(--teal) 0 ${rate}%,#26332f ${rate}% 100%)`;
  byId("kpi-ring-label").innerHTML = rate === null ? "—" : `${rate}<small>%</small>`;
  byId("kpi-breakdown").innerHTML = [
    `<div><span>Repository assets</span><strong>${numberFormat.format(data.assets.records)}</strong></div>`,
    `<div><span>Open / awaiting approval</span><strong>${numberFormat.format(data.kpis.open)}</strong></div>`,
    `<div><span>Separate alerts dataset</span><strong>Not present</strong></div>`
  ].join("");
  byId("data-source-note").textContent = `${data.source}. Values reflect checked-in files only; this local seed does not represent live plant telemetry.${data.alerts.message}`;
}

function jsonText(value) {
  if (value === null || value === undefined) return "No output recorded.";
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function renderAgentTrajectory(result) {
  const plan = result.plan?.steps?.length
    ? `<details class="trajectory-group"><summary>Generated execution plan · ${result.plan.steps.length} steps</summary><pre>${escapeHTML(jsonText(result.plan))}</pre></details>`
    : "";
  const trajectory = result.trajectory;
  let body = "";
  if (Array.isArray(trajectory)) {
    body = trajectory.map((step) => `<details class="trajectory-step" open><summary><span>${escapeHTML(step.server || "Agent")} · ${escapeHTML(step.tool || "step")}</span><span>${step.error ? "FAILED" : "RETURNED"}</span></summary>
      <div class="trajectory-content"><p>${escapeHTML(step.task || "")}</p>${step.tool_args && Object.keys(step.tool_args).length ? `<span class="detail-label">TOOL ARGUMENTS</span><pre>${escapeHTML(jsonText(step.tool_args))}</pre>` : ""}<span class="detail-label">RETRIEVED DATA / TOOL OUTPUT</span><pre>${escapeHTML(step.error || step.response || "No output recorded.")}</pre></div></details>`).join("");
  } else if (trajectory?.turns) {
    body = trajectory.turns.map((turn) => `<details class="trajectory-step" open><summary><span>Agent turn ${Number(turn.index) + 1}</span><span>${(turn.tool_calls || []).length} tool calls</span></summary><div class="trajectory-content">
      ${turn.text ? `<p>${escapeHTML(turn.text)}</p>` : ""}
      ${(turn.tool_calls || []).map((call) => `<div class="tool-call"><span class="detail-label">TOOL · ${escapeHTML(call.name)}</span><pre>${escapeHTML(jsonText(call.input))}</pre><span class="detail-label">TOOL OUTPUT</span><pre>${escapeHTML(jsonText(call.output))}</pre></div>`).join("")}
    </div></details>`).join("");
  } else {
    body = `<details class="trajectory-group"><summary>Raw runner trajectory</summary><pre>${escapeHTML(jsonText(trajectory))}</pre></details>`;
  }
  return `${plan}${body || '<p class="execution-empty">The runner returned no structured tool trajectory.</p>'}`;
}

function renderRunResult(result) {
  const evaluation = result.evaluation;
  const evaluationMarkup = evaluation?.error
    ? `<div class="judge-result judge-error"><strong>LLM-As-Judge could not score this run</strong><p>${escapeHTML(evaluation.error)}</p></div>`
    : `<div class="judge-result"><strong>Repository LLM-As-Judge · ${evaluation?.score === undefined ? "no score" : `${Math.round(evaluation.score * 100)}%`}</strong><p>${escapeHTML(evaluation?.rationale || "See the rubric details below.")}</p><details><summary>Rubric result</summary><pre>${escapeHTML(jsonText(evaluation?.details || evaluation))}</pre></details></div>`;
  byId("execution-output").innerHTML = `<div class="execution-result">
    <div class="answer-block"><span class="detail-label">REAL RUN · ${escapeHTML(result.runner)} · ${escapeHTML(result.model)}</span><h3>${escapeHTML(result.scenario_title)}</h3><p>${escapeHTML(result.answer || "The runner returned no final answer.")}</p></div>
    ${evaluationMarkup}
    ${window.AssetOpsReplay.render(result)}
    <details class="trajectory-group"><summary>Agent trajectory and tool calls</summary>${renderAgentTrajectory(result)}</details>
  </div>`;
}

function renderLeaderboard(results) {
  byId("leaderboard-empty").hidden = results.length > 0;
  byId("leaderboard-body").innerHTML = results.map((result, index) => {
    const evaluation = result.evaluation;
    const score = evaluation?.score === undefined ? "—" : `${Math.round(evaluation.score * 100)}%`;
    return `<tr style="animation-delay:${index * 45}ms"><td><div class="model-cell"><span class="rank-num">${String(index + 1).padStart(2, "0")}</span><span class="model-icon">${escapeHTML(result.scenario_id)}</span><span><strong>${escapeHTML(result.scenario_title)}</strong><small>${escapeHTML(result.question)}</small></span></div></td><td>${escapeHTML(result.runner)}</td><td>${escapeHTML(result.model)}</td><td class="score-cell">${score}</td><td>${evaluation?.passed ? "Passed" : evaluation?.error ? "Scoring error" : "Review needed"}</td></tr>`;
  }).join("");
}

async function loadResults() {
  const response = await api("/api/results");
  state.results = response.results;
  renderLeaderboard(state.results);
}

async function refreshRepository() {
  const button = byId("refresh-repo");
  button.disabled = true;
  const original = button.textContent;
  button.textContent = "Refreshing…";
  try {
    const [scenarioResponse, overview] = await Promise.all([api("/api/scenarios"), api("/api/overview")]);
    state.scenarios = scenarioResponse.scenarios;
    renderOverview(overview);
    renderScenarioOptions();
    refreshScenarioView();
    byId("run-message").textContent = "Repository scenarios and data reloaded from disk.";
  } catch (error) {
    byId("run-message").textContent = `Repository refresh failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function runScenario(scenarioId) {
  if (!scenarioId) return;
  const button = byId("run-scenario");
  const runMessage = byId("run-message");
  button.disabled = true;
  button.innerHTML = '<span class="play-icon">◌</span> Running repository agent…';
  byId("console-state").classList.add("running");
  byId("console-state").innerHTML = "<i></i> RUNNING";
  byId("console-message").textContent = "Calling the selected repository agent runner…";
  byId("execution-output").innerHTML = '<p class="execution-empty">Agent execution is in progress. This calls configured model and MCP services and may take a few minutes.</p>';
  runMessage.textContent = "Running the real AssetOpsBench pipeline…";
  try {
    const result = await api("/api/run", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId, runner: byId("runner-select").value })
    });
    renderRunResult(result);
    await loadResults();
    runMessage.textContent = "Repository pipeline completed.";
    byId("console-message").textContent = "Real agent trajectory and evaluation returned.";
    byId("console-state").innerHTML = "<i></i> COMPLETE";
  } catch (error) {
    byId("execution-output").innerHTML = `<div class="scenario-empty execution-error">${escapeHTML(error.message)}</div>`;
    runMessage.textContent = error.message;
    byId("console-message").textContent = "The repository run did not complete.";
    byId("console-state").innerHTML = "<i></i> ERROR";
  } finally {
    byId("console-state").classList.remove("running");
    button.disabled = !state.apiAvailable || !state.scenarios.length;
    button.innerHTML = '<span class="play-icon">▶</span> Run scenario <span class="button-arrow">↗</span>';
  }
}

async function evaluateLog(file) {
  const note = byId("judge-note");
  const selectedScenario = byId("evaluation-scenario").value;
  if (!selectedScenario) throw new Error("Choose a repository scenario with an evaluation reference first.");
  const contents = await file.text();
  let record;
  if (/\.jsonl$/i.test(file.name)) {
    const records = contents.split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line));
    record = records.find((item) => String(item.scenario_id || "") === selectedScenario) || (records.length === 1 ? records[0] : null);
  } else {
    record = JSON.parse(contents);
  }
  if (!record || typeof record !== "object" || Array.isArray(record)) throw new Error("No matching trajectory record found in the file.");
  note.innerHTML = '<span>◌</span><span><strong>Scoring with the repository LLM-As-Judge…</strong><br>The configured model is reviewing the selected scenario and trajectory.</span>';
  const result = await api("/api/evaluate-log", {
    method: "POST",
    body: JSON.stringify({ scenario_id: selectedScenario, record })
  });
  note.classList.add("file-selected");
  note.innerHTML = `<span>${result.passed ? "✓" : "!"}</span><span><strong>Repository LLM-As-Judge · ${result.score === undefined ? "no score" : `${Math.round(result.score * 100)}%`}</strong><br>${escapeHTML(result.rationale || JSON.stringify(result.details || {}))}</span>`;
}

byId("scenario-search").addEventListener("input", (event) => {
  state.search = event.target.value;
  renderScenarios();
});
document.querySelector(".filter-panel").addEventListener("click", (event) => {
  const button = event.target.closest("[data-filter]");
  if (!button) return;
  const selected = state.filters[button.dataset.filter];
  selected.has(button.dataset.value) ? selected.delete(button.dataset.value) : selected.add(button.dataset.value);
  refreshScenarioView();
});
byId("scenario-list").addEventListener("click", (event) => {
  const runButton = event.target.closest("[data-run-id]");
  if (runButton) {
    event.stopPropagation();
    byId("scenario-run-select").value = runButton.dataset.runId;
    byId("runner-select").focus();
    byId("demo").scrollIntoView({ behavior: "smooth", block: "start" });
    runScenario(runButton.dataset.runId);
    return;
  }
  const toggle = event.target.closest("[data-scenario]");
  if (!toggle) return;
  const id = toggle.dataset.scenario;
  state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id);
  renderScenarios();
});
byId("scenario-sort").addEventListener("change", renderScenarios);
byId("refresh-repo").addEventListener("click", refreshRepository);
byId("clear-filters").addEventListener("click", () => {
  state.search = "";
  Object.values(state.filters).forEach((set) => set.clear());
  byId("scenario-search").value = "";
  refreshScenarioView();
});
byId("teach-toggle").checked = localStorage.getItem("aob-teach-mode") === "on";
document.body.classList.toggle("teach-mode", byId("teach-toggle").checked);
byId("teach-toggle").addEventListener("change", (event) => {
  document.body.classList.toggle("teach-mode", event.target.checked);
  localStorage.setItem("aob-teach-mode", event.target.checked ? "on" : "off");
});

document.querySelectorAll(".paradigm-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".paradigm-tab").forEach((other) => {
      const active = other === tab;
      other.classList.toggle("active", active);
      other.setAttribute("aria-selected", String(active));
      const panel = byId(other.getAttribute("aria-controls"));
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
  });
});

document.querySelectorAll(".agent-node").forEach((node) => {
  const setActive = (active) => {
    node.classList.toggle("is-active", active);
    document.querySelector(`[data-agent-card="${node.dataset.agent}"]`)?.classList.toggle("is-active", active);
  };
  node.addEventListener("mouseenter", () => setActive(true));
  node.addEventListener("mouseleave", () => setActive(false));
  node.addEventListener("focus", () => setActive(true));
  node.addEventListener("blur", () => setActive(false));
});

byId("run-scenario").addEventListener("click", () => runScenario(byId("scenario-run-select").value));
byId("scenario-run-select").addEventListener("change", (event) => { byId("run-scenario").disabled = !event.target.value || !state.apiAvailable; });

const fileInput = byId("log-upload");
const fileDrop = byId("file-drop");
const judgeNote = byId("judge-note");
async function showFile(file) {
  if (!file) return;
  if (!/\.(json|jsonl)$/i.test(file.name)) {
    judgeNote.textContent = "Unsupported file type. Choose a JSON or JSONL trajectory log.";
    fileInput.value = "";
    return;
  }
  try {
    await evaluateLog(file);
  } catch (error) {
    judgeNote.textContent = `Evaluation failed: ${error.message}`;
  }
}
fileInput.addEventListener("change", () => showFile(fileInput.files[0]));
fileDrop.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach((eventName) => fileDrop.addEventListener(eventName, (event) => {
  event.preventDefault();
  fileDrop.classList.add("drag-over");
}));
["dragleave", "drop"].forEach((eventName) => fileDrop.addEventListener(eventName, (event) => {
  event.preventDefault();
  fileDrop.classList.remove("drag-over");
}));
fileDrop.addEventListener("drop", (event) => showFile(event.dataTransfer.files[0]));

async function initializeDashboard() {
  try {
    const status = await api("/api/status");
    state.apiAvailable = true;
    setBackendStatus("REPOSITORY API", true);
    byId("run-message").textContent = status.ready
      ? `Ready · ${status.model} · live model/MCP calls may use external services`
      : `Run setup needed · configure ${status.missing_environment.join(", ") || status.agent_import_error || "repository dependencies"}`;
    document.querySelectorAll(".paradigm-tab").length && (byId("runner-select").innerHTML = status.runners.map((runner) => `<option value="${escapeHTML(runner.id)}">${escapeHTML(runner.label)}</option>`).join(""));
    const [scenarioResponse, overview] = await Promise.all([api("/api/scenarios"), api("/api/overview")]);
    state.scenarios = scenarioResponse.scenarios;
    renderOverview(overview);
    renderScenarioOptions();
    refreshScenarioView();
    await loadResults();
    byId("console-message").textContent = `Loaded ${state.scenarios.length} repository scenario files; data values read from checked-in samples.`;
  } catch (error) {
    state.apiAvailable = false;
    setBackendStatus("API OFFLINE");
    renderScenarioOptions();
    byId("scenario-list").innerHTML = `<div class="scenario-empty">Cannot read repository files because the dashboard API is unavailable. Start it with <code>uv run python dashboard/server.py</code>. ${escapeHTML(error.message)}</div>`;
    byId("scenario-summary").textContent = "Repository API unavailable";
    byId("run-message").textContent = error.message;
    byId("console-message").textContent = "Start the repository-backed dashboard server.";
  }
}

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll(".section-block, .demo-section").forEach((section) => {
    section.classList.add("reveal-section");
    revealObserver.observe(section);
  });
}

initializeDashboard();
