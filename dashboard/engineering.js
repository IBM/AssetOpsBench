const $ = (id) => document.getElementById(id);
const API_ROOT = "/api";
let statusInfo = null;
let toolsByName = new Map();
let profiles = [];

async function api(path, options = {}) {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function showError(target, error) {
  target.textContent = error instanceof Error ? error.message : String(error);
  target.classList.add("is-error");
}

function setTeachMode(enabled) {
  document.body.classList.toggle("teach-mode", enabled);
  localStorage.setItem("aob-teach-mode", enabled ? "on" : "off");
}

function renderRun(run, compact = false) {
  const article = document.createElement("article");
  article.className = compact ? "run-card compact-run-card" : "run-card";
  const heading = document.createElement("div");
  heading.className = "run-card-heading";
  const title = document.createElement("h3");
  title.textContent = run.scenario_title || `Scenario ${run.scenario_id || "unknown"}`;
  const meta = document.createElement("span");
  meta.textContent = `${run.runner || "Repository run"} · ${run.model || "model unavailable"}`;
  heading.append(title, meta);
  article.append(heading);
  if (run.evaluation) {
    const evaluation = document.createElement("p");
    evaluation.className = "run-evaluation";
    const score = run.evaluation.score;
    evaluation.textContent = score === undefined
      ? `LLM judge: ${run.evaluation.error || run.evaluation.rationale || "No score returned"}`
      : `LLM judge score: ${score}${run.evaluation.passed === true ? " · passed" : run.evaluation.passed === false ? " · not passed" : ""}`;
    article.append(evaluation);
  }
  const answer = document.createElement("p");
  answer.className = "run-answer";
  answer.textContent = run.answer || "No answer was persisted for this record.";
  article.append(answer);
  const replay = document.createElement("div");
  replay.innerHTML = window.AssetOpsReplay.render(run);
  article.append(replay.firstElementChild);
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Inspect raw run record";
  const pre = document.createElement("pre");
  pre.className = "lab-output history-output";
  pre.textContent = JSON.stringify({
    question: run.question,
    plan: run.plan,
    trajectory: run.trajectory,
    evaluation: run.evaluation,
    run_id: run.run_id,
  }, null, 2);
  details.append(summary, pre);
  article.append(details);
  return article;
}

function renderRunCollection(container, runs, failures = []) {
  container.replaceChildren();
  if (!runs.length && !failures.length) {
    const empty = document.createElement("p");
    empty.className = "execution-empty";
    empty.textContent = "No local scenarios were run.";
    container.append(empty);
    return;
  }
  if (failures.length) {
    const errors = document.createElement("div");
    errors.className = "lab-warning";
    errors.textContent = failures.map((item) => `${item.runner || item.scenario_id}: ${item.error}`).join(" · ");
    container.append(errors);
  }
  runs.forEach((run) => container.append(renderRun(run, true)));
}

async function loadTools(serverName) {
  const select = $("tool-name");
  select.disabled = true;
  $("call-tool").disabled = true;
  $("tool-description").textContent = "Discovering tool definitions from the repository MCP server…";
  $("tool-result").textContent = "Waiting for the repository MCP server response…";
  $("tool-result").classList.remove("is-error");
  try {
    const response = await api(`/tools?server=${encodeURIComponent(serverName)}`);
    toolsByName = new Map(response.tools.map((tool) => [tool.name, tool]));
    select.replaceChildren();
    response.tools.forEach((tool) => {
      const option = document.createElement("option");
      option.value = tool.name;
      option.textContent = tool.name;
      select.append(option);
    });
    select.disabled = response.tools.length === 0;
    $("call-tool").disabled = response.tools.length === 0;
    if (response.tools.length) updateToolDescription();
    else $("tool-description").textContent = "No tools are enabled by the dashboard's read-only tool policy.";
  } catch (error) {
    select.replaceChildren(new Option("Unable to load tools", ""));
    $("tool-description").textContent = "The MCP server could not be reached or initialized.";
    showError($("tool-result"), error);
  }
}

function updateToolDescription() {
  const tool = toolsByName.get($("tool-name").value);
  if (!tool) return;
  $("tool-description").textContent = tool.description || "No description is provided by this repository tool.";
  const required = tool.parameters.filter((parameter) => parameter.required);
  $("tool-arguments").value = required.length
    ? JSON.stringify(Object.fromEntries(required.map(({ name, type }) => [name, type === "integer" || type === "number" ? 0 : type === "boolean" ? false : type === "object" || type === "array" ? (type === "array" ? [] : {}) : ""])), null, 2)
    : "{}";
}

async function loadScenarios() {
  try {
    const { scenarios } = await api("/scenarios");
    for (const id of ["compare-scenario"]) {
      const select = $(id);
      select.replaceChildren();
      scenarios.forEach((scenario) => {
        const option = document.createElement("option");
        option.value = scenario.id;
        option.textContent = `${scenario.id} · ${scenario.title}`;
        select.append(option);
      });
    }
  } catch (error) {
    $("compare-scenario").replaceChildren(new Option("Scenario API unavailable", ""));
    $("compare-status").textContent = error.message;
  }
}

function selectedRunners() {
  return [...document.querySelectorAll(".runner-options input:checked")].map((input) => input.value);
}

async function loadProfiles() {
  const select = $("benchmark-profile");
  try {
    const response = await api("/benchmark/profiles");
    profiles = response.profiles;
    select.replaceChildren();
    profiles.forEach((profile) => select.append(new Option(profile.id, profile.id)));
    updateBenchmarkDetail();
  } catch (error) {
    select.replaceChildren(new Option("Benchmark profiles unavailable", ""));
    showError($("benchmark-detail"), error);
  }
}

function updateBenchmarkDetail() {
  const profile = profiles.find((item) => item.id === $("benchmark-profile").value);
  const detail = $("benchmark-detail");
  if (!profile) return;
  detail.classList.remove("is-error");
  const localMatches = profile.locally_available.length;
  detail.textContent = `${profile.id.toUpperCase()} · ${profile.scenario_count} IDs in ${profile.source} · ${localMatches} locally runnable scenario${localMatches === 1 ? "" : "s"} · ${profile.scenario_count - localMatches} require the external benchmark scenario loader.`;
  $("run-benchmark").disabled = !statusInfo?.ready || localMatches === 0 || localMatches > 20;
  $("benchmark-status").textContent = !statusInfo?.ready
    ? `Configure credentials for ${statusInfo?.missing_environment?.join(", ") || "the selected model"} before running.`
    : localMatches > 20
      ? "This profile exceeds the dashboard's 20-scenario batch limit."
      : localMatches === 0
        ? "No profile scenario IDs match scenario definitions in this checkout."
        : `Runs ${localMatches} actual scenario${localMatches === 1 ? "" : "s"}; model and judge calls may incur costs.`;
}

async function refreshHistory() {
  const count = $("history-count");
  const container = $("run-history-list");
  count.textContent = "Loading saved runs…";
  try {
    const response = await api("/history");
    container.replaceChildren();
    response.runs.forEach((run) => container.append(renderRun(run)));
    renderEvaluationSummary(response.runs);
    count.textContent = `${response.runs.length} saved repository run${response.runs.length === 1 ? "" : "s"}`;
    if (!response.runs.length) {
      const empty = document.createElement("p");
      empty.className = "execution-empty";
      empty.textContent = "No persisted dashboard trajectories yet. Run an agent scenario to create one.";
      container.append(empty);
    }
  } catch (error) {
    count.textContent = "History unavailable";
    container.replaceChildren();
    showError(container, error);
  }
}

function renderEvaluationSummary(runs) {
  const container = $("evaluation-summary");
  container.replaceChildren();
  const scored = runs.filter((run) => typeof run.evaluation?.score === "number");
  const mean = scored.length
    ? scored.reduce((total, run) => total + run.evaluation.score, 0) / scored.length
    : null;
  const metrics = [
    ["Saved trajectories", String(runs.length)],
    ["Repository-judge scores", String(scored.length)],
    ["Mean score", mean === null ? "—" : `${Math.round(mean * 100)}%`],
  ];
  metrics.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "evaluation-metric";
    const name = document.createElement("span");
    name.textContent = label;
    const number = document.createElement("strong");
    number.textContent = value;
    card.append(name, number);
    container.append(card);
  });
  const byRunner = new Map();
  scored.forEach((run) => {
    const group = byRunner.get(run.runner) || [];
    group.push(run.evaluation.score);
    byRunner.set(run.runner, group);
  });
  if (byRunner.size) {
    const comparison = document.createElement("div");
    comparison.className = "runner-score-summary";
    [...byRunner.entries()].sort(([left], [right]) => left.localeCompare(right)).forEach(([runner, scores]) => {
      const item = document.createElement("span");
      const average = scores.reduce((sum, score) => sum + score, 0) / scores.length;
      item.textContent = `${runner}: ${Math.round(average * 100)}% mean · ${scores.length} scored`;
      comparison.append(item);
    });
    container.append(comparison);
  }
}

$("teach-toggle").checked = localStorage.getItem("aob-teach-mode") === "on";
setTeachMode($("teach-toggle").checked);
$("teach-toggle").addEventListener("change", (event) => setTeachMode(event.target.checked));
$("tool-server").addEventListener("change", (event) => loadTools(event.target.value));
$("tool-name").addEventListener("change", updateToolDescription);
$("call-tool").addEventListener("click", async () => {
  const output = $("tool-result");
  $("call-tool").disabled = true;
  $("tool-result-title").textContent = `${$("tool-name").value} · calling…`;
  output.classList.remove("is-error");
  try {
    const args = JSON.parse($("tool-arguments").value || "{}");
    if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("Tool arguments must be a JSON object.");
    const result = await api("/tool-call", {
      method: "POST",
      body: JSON.stringify({ server: $("tool-server").value, tool: $("tool-name").value, arguments: args }),
    });
    $("tool-result-title").textContent = `${result.server} / ${result.tool}`;
    output.textContent = typeof result.output === "string" ? result.output : JSON.stringify(result.output, null, 2);
  } catch (error) {
    $("tool-result-title").textContent = "Tool call failed";
    showError(output, error);
  } finally {
    $("call-tool").disabled = !$("tool-name").value;
  }
});
$("compare-runners").addEventListener("click", async () => {
  const button = $("compare-runners");
  const runners = selectedRunners();
  button.disabled = true;
  $("compare-status").textContent = "Executing selected repository agent runners…";
  try {
    const model = $("compare-model").value.trim();
    const result = await api("/compare", {
      method: "POST",
      body: JSON.stringify({ scenario_id: $("compare-scenario").value, runners, ...(model ? { model } : {}) }),
    });
    renderRunCollection($("compare-results"), result.runs, result.failures);
    $("compare-status").textContent = `${result.runs.length} runner${result.runs.length === 1 ? "" : "s"} completed; ${result.failures.length} failed.`;
    refreshHistory();
  } catch (error) {
    $("compare-status").textContent = error.message;
    showError($("compare-results"), error);
  } finally {
    button.disabled = !statusInfo?.ready || selectedRunners().length < 2 || selectedRunners().length > 4;
  }
});
document.querySelectorAll(".runner-options input").forEach((input) => input.addEventListener("change", () => {
  const count = selectedRunners().length;
  $("compare-runners").disabled = !statusInfo?.ready || count < 2 || count > 4 || !$("compare-scenario").value;
}));
$("compare-scenario").addEventListener("change", () => {
  $("compare-runners").disabled = !statusInfo?.ready || selectedRunners().length < 2 || !$("compare-scenario").value;
});
$("benchmark-profile").addEventListener("change", updateBenchmarkDetail);
$("run-benchmark").addEventListener("click", async () => {
  const button = $("run-benchmark");
  button.disabled = true;
  $("benchmark-status").textContent = "Running repository scenarios sequentially; this may take time…";
  $("benchmark-results").replaceChildren();
  try {
    const model = $("benchmark-model").value.trim();
    const result = await api("/benchmark/run", {
      method: "POST",
      body: JSON.stringify({
        profile: $("benchmark-profile").value,
        runner: $("benchmark-runner").value,
        ...(model ? { model } : {}),
      }),
    });
    renderRunCollection($("benchmark-results"), result.runs, result.failures);
    if (result.skipped_scenario_ids.length) {
      const skipped = document.createElement("p");
      skipped.className = "lab-help";
      skipped.textContent = `Not run (not present as local scenario definitions): ${result.skipped_scenario_ids.join(", ")}`;
      $("benchmark-results").append(skipped);
    }
    $("benchmark-status").textContent = `${result.runs.length} completed, ${result.failures.length} failed; no data reset performed.`;
    refreshHistory();
  } catch (error) {
    $("benchmark-status").textContent = error.message;
  } finally {
    updateBenchmarkDetail();
  }
});
$("refresh-history").addEventListener("click", refreshHistory);

async function initialize() {
  try {
    statusInfo = await api("/status");
    $("lab-connection").textContent = statusInfo.ready
      ? `Connected · ${statusInfo.scenarios} repository scenarios · model ${statusInfo.model}`
      : `Repository API connected · ${statusInfo.scenarios} scenarios · missing credentials: ${statusInfo.missing_environment.join(", ")}`;
    $("lab-connection").classList.toggle("is-warning", !statusInfo.ready);
    $("compare-runners").disabled = !statusInfo.ready;
  } catch (error) {
    $("lab-connection").textContent = `Local repository API unavailable: ${error.message}. Start it with uv run python dashboard/server.py.`;
    $("lab-connection").classList.add("is-warning");
  }
  await Promise.all([loadTools($("tool-server").value), loadScenarios(), loadProfiles(), refreshHistory()]);
  if (statusInfo) {
    $("compare-runners").disabled = !statusInfo.ready || selectedRunners().length < 2 || !$("compare-scenario").value;
  }
}

initialize();
