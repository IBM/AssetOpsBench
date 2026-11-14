(() => {
  const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  const jsonText = (value) => typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "No output recorded.";

  function eventMarkup(event, index) {
    const evidence = event.kind === "evidence";
    const detailMarkup = evidence
      ? `<details class="replay-evidence"><summary>Inspect tool input and exact returned output</summary>
          ${event.input !== undefined ? `<span class="detail-label">TOOL INPUT · ${escapeHTML(event.tool)}</span><pre>${escapeHTML(jsonText(event.input))}</pre>` : ""}
          <span class="detail-label">TOOL RESPONSE · ${escapeHTML(event.tool)}</span><pre>${escapeHTML(event.output)}</pre>
        </details>`
      : event.detail
        ? `<details class="replay-evidence"><summary>${escapeHTML(event.detailLabel || "Inspect details")}</summary><pre>${escapeHTML(jsonText(event.detail))}</pre></details>`
        : "";
    return `<li class="replay-step${evidence ? " replay-evidence-step" : ""}${index === 0 ? " is-current" : ""}" data-replay-step data-replay-index="${index}">
      <div class="replay-step-marker" aria-hidden="true">${String(index + 1).padStart(2, "0")}</div>
      <div class="replay-step-content">
        <button class="replay-select" type="button" data-replay-select="${index}"${index === 0 ? ' aria-current="step"' : ""}><span class="replay-step-label">${escapeHTML(event.label)}</span><span class="replay-step-title">${escapeHTML(event.title)}</span></button>
        ${event.description ? `<p class="replay-step-description">${escapeHTML(event.description)}</p>` : ""}
        ${detailMarkup}
      </div>
    </li>`;
  }

  function eventsFor(run) {
    const events = [{
      kind: "scenario",
      label: "SCENARIO",
      title: run.question || `Scenario ${run.scenario_id || ""}`,
      description: run.scenario_id ? `Repository scenario ${run.scenario_id}` : "Scenario prompt",
    }];
    if (run.plan) {
      events.push({
        kind: "plan",
        label: "EXECUTION PLAN",
        title: "Plan returned by the runner",
        detailLabel: "RECORDED PLAN",
        detail: run.plan,
      });
    }

    const trajectory = run.trajectory;
    let evidenceCount = 0;
    if (Array.isArray(trajectory)) {
      trajectory.forEach((step, index) => {
        if (step.task) {
          events.push({ kind: "note", label: `AGENT NOTE · STEP ${index + 1}`, title: step.task });
        }
        const tool = [step.server, step.tool].filter(Boolean).join(" · ") || `Agent step ${index + 1}`;
        events.push({
          kind: "evidence",
          label: step.error ? "TOOL ERROR" : "RETRIEVED EVIDENCE",
          title: tool,
          tool,
          input: step.tool_args,
          output: jsonText(step.error || step.response || "No tool response was recorded."),
        });
        evidenceCount += 1;
      });
    } else if (Array.isArray(trajectory?.turns)) {
      trajectory.turns.forEach((turn, index) => {
        if (typeof turn.text === "string" && turn.text.trim()) {
          events.push({
            kind: "note",
            label: `AGENT NOTE · TURN ${Number(turn.index ?? index) + 1}`,
            title: turn.text,
          });
        }
        (Array.isArray(turn.tool_calls) ? turn.tool_calls : []).forEach((call) => {
          const tool = call.name || "Unnamed tool";
          events.push({
            kind: "evidence",
            label: call.error ? "TOOL ERROR" : "RETRIEVED EVIDENCE",
            title: tool,
            tool,
            input: call.input,
            output: call.error || (call.output === undefined || call.output === null
              ? "No tool response was recorded."
              : jsonText(call.output)),
          });
          evidenceCount += 1;
        });
      });
    } else if (trajectory !== undefined && trajectory !== null) {
      events.push({
        kind: "note",
        label: "RECORDED TRAJECTORY",
        title: "This runner returned an unstructured trajectory",
        detailLabel: "RAW TRAJECTORY",
        detail: trajectory,
      });
    }

    events.push({
      kind: "answer",
      label: "FINAL RECOMMENDATION",
      title: run.answer || "The runner returned no final answer.",
    });
    if (run.evaluation) {
      const evaluation = run.evaluation;
      const score = typeof evaluation.score === "number" ? ` · ${Math.round(evaluation.score * 100)}%` : "";
      events.push({
        kind: "evaluation",
        label: "REPOSITORY EVALUATION",
        title: `${evaluation.passed === true ? "Passed" : evaluation.passed === false ? "Needs review" : "Evaluation returned"}${score}`,
        description: evaluation.rationale || evaluation.error || "",
        detail: evaluation.details,
        detailLabel: "EVALUATION DETAILS",
      });
    }
    return { events, evidenceCount };
  }

  function render(run) {
    const { events, evidenceCount } = eventsFor(run);
    const runLabel = run.run_id ? `RUN ${run.run_id}` : `${run.runner || "AGENT"} · ${run.model || "MODEL"}`;
    return `<section class="evidence-replay" aria-label="Evidence-linked run replay">
      <div class="replay-heading"><div><span class="detail-label">EVIDENCE-LINKED REPLAY</span><h3>Trace the run, step by step.</h3></div><span class="replay-count">${evidenceCount} tool response${evidenceCount === 1 ? "" : "s"}</span></div>
      <p class="replay-disclaimer">Ordered record of the prompt, agent notes, tool inputs and returned outputs, recommendation, and evaluation. Tool outputs are shown as returned; their presence does not prove the agent relied on them.</p>
      <div class="replay-controls">
        <button type="button" class="replay-control" data-replay-action="prev" aria-label="Previous replay step">←</button>
        <button type="button" class="replay-control replay-play" data-replay-action="play">▶ Play replay</button>
        <button type="button" class="replay-control" data-replay-action="next" aria-label="Next replay step">→</button>
        <span class="replay-progress" data-replay-status role="status">Step 1 of ${events.length} · ${escapeHTML(events[0].title)}</span>
      </div>
      <ol class="replay-timeline" data-replay-list>${events.map(eventMarkup).join("")}</ol>
      <div class="replay-source">${escapeHTML(runLabel)}${run.created_at ? ` · ${escapeHTML(run.created_at)}` : ""}</div>
    </section>`;
  }

  const timers = new WeakMap();
  function selectStep(panel, index) {
    const steps = [...panel.querySelectorAll("[data-replay-step]")];
    if (!steps.length) return;
    const selected = Math.max(0, Math.min(index, steps.length - 1));
    steps.forEach((step, position) => {
      const active = position === selected;
      step.classList.toggle("is-current", active);
      const button = step.querySelector("[data-replay-select]");
      if (active) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    const step = steps[selected];
    const status = panel.querySelector("[data-replay-status]");
    status.textContent = `Step ${selected + 1} of ${steps.length} · ${step.querySelector(".replay-step-title").textContent}`;
    const details = step.querySelector("details");
    if (details) details.open = true;
  }

  function stop(panel) {
    const timer = timers.get(panel);
    if (timer) clearInterval(timer);
    timers.delete(panel);
    const play = panel.querySelector('[data-replay-action="play"]');
    if (play) play.textContent = "▶ Play replay";
  }

  document.addEventListener("click", (event) => {
    const panel = event.target.closest(".evidence-replay");
    if (!panel) return;
    const select = event.target.closest("[data-replay-select]");
    const action = event.target.closest("[data-replay-action]");
    if (select) {
      stop(panel);
      selectStep(panel, Number(select.dataset.replaySelect));
      return;
    }
    if (!action) return;
    const steps = [...panel.querySelectorAll("[data-replay-step]")];
    const current = Math.max(0, steps.findIndex((step) => step.classList.contains("is-current")));
    if (action.dataset.replayAction === "prev" || action.dataset.replayAction === "next") {
      stop(panel);
      selectStep(panel, current + (action.dataset.replayAction === "prev" ? -1 : 1));
    } else if (timers.has(panel)) {
      stop(panel);
    } else {
      action.textContent = "Ⅱ Pause replay";
      timers.set(panel, setInterval(() => {
        const active = [...panel.querySelectorAll("[data-replay-step]")].findIndex((step) => step.classList.contains("is-current"));
        if (active >= steps.length - 1) {
          stop(panel);
        } else {
          selectStep(panel, active + 1);
        }
      }, 1500));
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".evidence-replay").forEach((panel) => selectStep(panel, 0));
  });
  window.AssetOpsReplay = { render };
})();
