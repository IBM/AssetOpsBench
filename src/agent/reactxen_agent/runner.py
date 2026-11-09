"""ReActXen: actor, workspace-only ReAct review, and ReAct reflection.

Only the actor attaches domain MCP servers. All phases share one live execution
environment. Review verdicts control retries; benchmark scoring stays external.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path

from llm import LLMBackend
from observability.runspan import scoped_run_context

from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult
from ..runner import AgentRunner
from ..stirrup_agent import StirrupAgentRunner
from .candidates import CandidateComparison, CandidateLedger
from .distill import clarification_block, distill_question, needs_distillation
from .models import Outcome, ReActXenTrajectory, ReviewVerdict, RoundRecord
from .prompts import REFLECT_REACT_SYSTEM, REVIEW_REACT_SYSTEM, actor_round_notice
from .reflect import as_prompt_block, reflect
from .reflection_plan import (
    ReflectionPlan,
    build_reflection_finish_tool,
    format_guidance,
    validate_selection,
)
from .review import _render_preconditions, check_preconditions, review
from .review_plan import ReviewFinishParams, build_review_finish_tool
from .self_ask import SelfAskTool
from .workspace import shared_workspace

_log = logging.getLogger(__name__)

MAX_ROUNDS = 3

_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


class ReActXenAgentRunner(AgentRunner):
    """Stirrup as ReAct, wrapped in self-ask, review/critique and reflect.

    Args:
        llm: Accepted for :class:`AgentRunner` interface parity.
        server_paths: MCP server specs (defaults to all registered servers).
        model: Router prefix or native ``<provider>/<model>`` for the ReAct core.
        critic_model: Model for self-ask, review and reflect.  Defaults to
            ``model``.  Set it to a different model only if you intend to
            report a two-model configuration; a stronger critic than actor
            is a confound, not a free win.
        max_rounds: Loop bound, capped at :data:`MAX_ROUNDS`.
        self_ask_enabled: Expose the self-ask tool to the core.  The runner
            counts invocations either way, so a zero count with the flag on
            is a reportable finding rather than a silent no-op.
        review_enabled: Run review and reflect.  With both flags off this
            runner reduces exactly to ``stirrup-agent``, which is the
            baseline arm of the ablation.
        declared_evidence: Evidence files the scenario manifest declares, used
            by the ``declared_evidence_consulted`` precondition.
        expect_json: Whether the family's criterion expects a JSON answer.
        code_enabled: Enable the shared code workspace (required for ReAct critics).
        critic_mode: Workspace-only Stirrup ReAct critics, or legacy completion.
        review_max_turns: Review session turn budget.
        reflect_max_turns: Reflect session turn budget.
        **stirrup_kwargs: Forwarded to :class:`StirrupAgentRunner`.
    """

    def __init__(
        self,
        llm: LLMBackend | None = None,
        server_paths: dict[str, Path | str] | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        critic_model: str | None = None,
        max_rounds: int = MAX_ROUNDS,
        self_ask_enabled: bool = True,
        review_enabled: bool = True,
        declared_evidence: Sequence[str] = (),
        expect_json: bool = False,
        code_enabled: bool = True,
        critic_mode: str = "react",
        review_max_turns: int = 10,
        reflect_max_turns: int = 8,
        reflect_finish_repair_attempts: int = 2,
        review_finish_repair_attempts: int = 2,
        distill_question_enabled: bool = True,
        **stirrup_kwargs,
    ) -> None:
        super().__init__(llm, server_paths)
        self._model = model
        self._critic_model = critic_model or model
        self._max_rounds = max(1, min(max_rounds, MAX_ROUNDS))
        self._self_ask_enabled = self_ask_enabled
        self._review_enabled = review_enabled
        self._declared_evidence = list(declared_evidence)
        self._expect_json = expect_json
        self._code_enabled = code_enabled
        if code_enabled and stirrup_kwargs.get("code_backend", "docker") != "docker":
            raise ValueError(
                "ReActXen uses a network-disabled Docker sandbox; local execution is not supported"
            )
        if critic_mode not in {"react", "completion"}:
            raise ValueError("critic_mode must be 'react' or 'completion'")
        if critic_mode == "react" and review_enabled and not code_enabled:
            raise ValueError(
                "ReAct critics require code_enabled=True; use --critic-mode completion or --no-review with --no-code"
            )
        if review_max_turns < 1 or reflect_max_turns < 1:
            raise ValueError("critic turn limits must be positive")
        self._critic_mode = critic_mode
        self._review_max_turns = review_max_turns
        self._reflect_max_turns = reflect_max_turns
        if not 0 <= reflect_finish_repair_attempts <= 3:
            raise ValueError("reflect_finish_repair_attempts must be between 0 and 3")
        self._reflect_finish_repair_attempts = reflect_finish_repair_attempts
        if not 0 <= review_finish_repair_attempts <= 3:
            raise ValueError("review_finish_repair_attempts must be between 0 and 3")
        self._review_finish_repair_attempts = review_finish_repair_attempts
        self._distill_question_enabled = distill_question_enabled
        self._stirrup_kwargs = stirrup_kwargs

    # -- the critic channel ------------------------------------------------
    #
    # Self-ask and the optional legacy completion-only critic mode.

    async def _complete(
        self, system: str, user: str, *, max_tokens=None, json_only=False
    ) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)`` from the critic model."""
        import litellm

        from llm.routers import resolve_model, resolve_router_creds

        resolved = resolve_model(self._critic_model)
        creds = resolve_router_creds(self._critic_model)
        kwargs = {}
        if creds is not None:
            resolved = "openai/" + resolved
            kwargs = {"api_base": creds.base_url, "api_key": creds.api_key}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_only:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["timeout"] = 90
        response = await litellm.acompletion(
            model=resolved,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            **kwargs,
        )
        usage = getattr(response, "usage", None)
        return (
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    async def _distill_complete(self, system, user):
        return await self._complete(system, user, max_tokens=3000, json_only=True)

    def _core(self, system_prompt: str, self_ask: SelfAskTool | None):
        """Build the Stirrup core for one round."""
        kwargs = dict(self._stirrup_kwargs)
        kwargs["network_disabled"] = True
        kwargs["require_finish"] = True
        if self_ask is not None:
            # Adapt the optional self-ask callable to Stirrup's typed tools.
            kwargs["extra_tools"] = [
                *kwargs.get("extra_tools", []),
                self_ask.as_stirrup_tool(),
            ]
        return StirrupAgentRunner(
            self._llm,
            self._server_paths,
            model=self._model,
            code_enabled=self._code_enabled,
            system_prompt=system_prompt,
            **kwargs,
        )

    def _critic(self, role: str, workspace, alternative_ids=()):
        """Critics have code/file helpers and explicitly NO MCP server namespace."""
        ledger = getattr(workspace, "candidate_ledger", None)
        ledger = ledger if isinstance(ledger, CandidateLedger) else None
        return StirrupAgentRunner(
            self._llm,
            server_paths={},
            model=self._critic_model,
            code_enabled=True,
            code_provider=workspace,
            code_backend=self._stirrup_kwargs.get("code_backend", "docker"),
            system_prompt=REVIEW_REACT_SYSTEM
            if role == "review"
            else REFLECT_REACT_SYSTEM,
            max_turns=self._review_max_turns
            if role == "review"
            else self._reflect_max_turns,
            temperature=self._stirrup_kwargs.get("temperature"),
            reasoning_effort=self._stirrup_kwargs.get("reasoning_effort"),
            runner_name=f"reactxen-{role}",
            persist_result=False,
            network_disabled=True,
            require_finish=True,
            finish_repair_attempts=self._reflect_finish_repair_attempts
            if role == "reflect"
            else self._review_finish_repair_attempts,
            finish_tool=build_reflection_finish_tool(alternative_ids, ledger)
            if role == "reflect"
            else build_review_finish_tool(ledger),
            extra_tools=[ledger.registration_tool()] if ledger is not None else [],
        )

    async def _save_context(self, workspace, index: int, role: str, payload) -> str:
        path = f"context/round-{index:03d}/{role}.json"
        await workspace.write_file_bytes(
            path, json.dumps(payload, default=str, ensure_ascii=False).encode()
        )
        return path

    async def _react_review(self, question, round_record, workspace):
        ledger = getattr(workspace, "candidate_ledger", None)
        ledger = ledger if isinstance(ledger, CandidateLedger) else None
        if ledger is not None:
            ledger.phase = f"round-{round_record.index:03d}-review"
        checks = check_preconditions(
            question=question,
            answer=round_record.answer,
            trajectory=round_record.trajectory,
            declared_evidence=self._declared_evidence,
            expect_json=self._expect_json,
        )
        context = await self._save_context(
            workspace,
            round_record.index,
            "actor",
            {
                "question": question,
                "answer": round_record.answer,
                "trajectory": dataclasses.asdict(round_record.trajectory),
                "declared_evidence": self._declared_evidence,
                "preconditions": [dataclasses.asdict(p) for p in checks],
                "actor_round_notice": round_record.actor_round_notice,
                "final_actor_attempt": round_record.final_actor_attempt,
                "advisory_clarification_used": round_record.advisory_clarified_question,
            },
        )
        evidence_index = []
        for turn_index, turn in enumerate(round_record.trajectory.turns):
            for call_index, call in enumerate(turn.tool_calls):
                entry = {
                    "tool": call.name,
                    "input": call.input,
                    "output_location": f"{context}#trajectory.turns[{turn_index}].tool_calls[{call_index}].output",
                }
                try:
                    output = (
                        json.loads(call.output)
                        if isinstance(call.output, str)
                        else call.output
                    )
                    if (
                        isinstance(output, dict)
                        and output.get("artifact_type") == "mcp_result"
                    ):
                        entry["snapshot"] = output
                except (ValueError, TypeError):
                    pass
                evidence_index.append(entry)
        index_path = await self._save_context(
            workspace, round_record.index, "evidence-index", evidence_index
        )
        task = (
            f"Verify the actor's answer to this original request:\n{question}\n\n"
            f"Candidate answer:\n{round_record.answer}\n\n"
            f"Start with the compact evidence index: {index_path}; avoid rediscovering files.\n"
            f"Full collected evidence/tool trace: {context}\n"
            f"Shared snapshots: mcp_results/ (inspect existing files; do not fetch new data).\n"
            f"Write scratch files under review/round-{round_record.index:03d}/.\n"
            f"Mechanical checks:\n{_render_preconditions(checks)}\n"
            "Actively seek plausible competing interpretations and answers, especially different "
            "populations, filters, exclusions, units or methods that could change the result. "
            "Read candidate-ledger/index.json and compare EVERY retained candidate_id in "
            "candidate_comparisons, including every labelled calculation table even if it is not "
            "in your five new alternatives. For each, derive its candidate answer where possible, "
            "give evidence, and mark supported/plausible/ruled_out/not_answer. Never silently drop "
            "a previously computed result, or reject it merely as an outlier. Register newly "
            "discovered answers with retain_candidate if not already retained automatically. "
            "Check the generated answer against the ORIGINAL question, including exact objective, "
            "scope, exclusions, units, requested fields and output format. A question clarification "
            "is advisory only: an answer matching a revision but not the original is misaligned. "
            "Report question_alignment_verified and alignment_findings explicitly. "
            "Test meaningful alternatives and preserve unresolved ones. Verify requested population "
            "semantics separately from arithmetic; reproducing a sum does not justify its filter. "
            "Call finish with the typed outcome, arithmetic_verified, population_verified, critique, "
            "alternative_search_summary, alternatives and evidence_paths shown in its schema."
        )
        try:
            with scoped_run_context(f"round-{round_record.index:03d}-review"):
                result = await self._critic("review", workspace).run(task)
        except Exception as exc:
            _log.warning("ReAct review failed", exc_info=True)
            round_record.phase_errors["review"] = f"{type(exc).__name__}: {exc}"
            return ReviewVerdict(
                outcome=Outcome.PARTIAL,
                preconditions=checks,
                critique="Reviewer unavailable; answer is unverified.",
            )
        round_record.review_trajectory = result.trajectory
        if not result.completed:
            round_record.phase_errors["review"] = (
                "Reviewer did not return a completed finish.answer"
            )
            return ReviewVerdict(
                outcome=Outcome.PARTIAL,
                preconditions=checks,
                critique="Review incomplete; no completed verification report was produced.",
                input_tokens=result.trajectory.total_input_tokens,
                output_tokens=result.trajectory.total_output_tokens,
            )
        round_record.review_report = result.answer
        try:
            report = ReviewFinishParams.model_validate_json(result.answer)
            if ledger is not None:
                ledger.require_coverage(report.candidate_comparisons)
        except ValueError as exc:
            round_record.phase_errors["review"] = f"Invalid typed review report: {exc}"
            return ReviewVerdict(
                outcome=Outcome.PARTIAL,
                preconditions=checks,
                critique="Reviewer did not produce a valid typed verification report.",
                input_tokens=result.trajectory.total_input_tokens,
                output_tokens=result.trajectory.total_output_tokens,
            )
        if ledger is not None:
            compared = {c.candidate_id for c in report.candidate_comparisons}
            for alternative in report.alternatives:
                ident = ledger.retain(
                    {
                        "kind": "registered",
                        "interpretation": alternative.interpretation,
                        "candidate_answer": alternative.candidate_answer,
                    },
                    {
                        "phase": ledger.phase,
                        "source": f"context/round-{round_record.index:03d}/review.json",
                    },
                )
                if ident not in compared:
                    # The report itself already contains this new option's assessment.
                    report.candidate_comparisons.append(
                        CandidateComparison(
                            candidate_id=ident,
                            status=alternative.status,
                            candidate_answer=alternative.candidate_answer,
                            interpretation=alternative.interpretation,
                            supporting_evidence=alternative.supporting_evidence,
                            counterevidence_or_gap=alternative.counterevidence_or_gap,
                            evidence_paths=alternative.evidence_paths,
                        )
                    )
                    compared.add(ident)
            await ledger.flush()
        round_record.review_plan = report.model_dump(exclude={"paths"})
        round_record.review_report = report.answer
        outcome, critique = report.outcome, report.critique
        alternatives = [a.model_dump_json() for a in report.alternatives]
        # A tool-using reviewer must actually inspect evidence before accepting.
        if outcome is Outcome.ACCOMPLISHED and (
            any(not p.passed for p in checks)
            or not any(c.name == "code_exec" for c in result.trajectory.all_tool_calls)
        ):
            outcome = Outcome.PARTIAL
            critique += " Independent evidence verification remains incomplete."
        return ReviewVerdict(
            outcome=outcome,
            preconditions=checks,
            critique=critique,
            alternatives=alternatives,
            input_tokens=result.trajectory.total_input_tokens,
            output_tokens=result.trajectory.total_output_tokens,
        )

    async def _react_reflect(self, question, round_record, workspace):
        ledger = getattr(workspace, "candidate_ledger", None)
        ledger = ledger if isinstance(ledger, CandidateLedger) else None
        if ledger is not None:
            ledger.phase = f"round-{round_record.index:03d}-reflect"
        context = await self._save_context(
            workspace,
            round_record.index,
            "review",
            {
                "question": question,
                "actor_context": f"context/round-{round_record.index:03d}/actor.json",
                "verdict": dataclasses.asdict(round_record.review),
                "report": round_record.review_report,
                "structured_review": round_record.review_plan,
                "candidate_ledger": "candidate-ledger/index.json",
                "trajectory": dataclasses.asdict(round_record.review_trajectory)
                if round_record.review_trajectory
                else None,
            },
        )
        alternatives = [
            {"alternative_id": f"alternative_{i + 1}", "description": value}
            for i, value in enumerate(round_record.review.alternatives)
        ]
        alternative_ids = [a["alternative_id"] for a in alternatives]
        task = (
            f"Investigate the review and select an evidence-backed correction for:\n{question}\n"
            f"Review verdict, full report and verification trace: {context}\n"
            "Actor trace and original results are linked there; snapshots are in mcp_results/.\n"
            f"Use context/round-{round_record.index:03d}/evidence-index.json to locate relevant outputs.\n"
            f"Write scratch files under reflect/round-{round_record.index:03d}/.\n"
            "Inspect the existing evidence with code tools. Do not call services or fetch data. "
            "Assess every offered alternative, explicitly stating why you select, reject or leave it unresolved. "
            "Also read candidate-ledger/index.json: candidate_assessments must account for EVERY "
            "retained candidate_id, independently of the shortlist. Explicitly select, reject with "
            "counterevidence, leave unresolved, or mark not_answer with a reason. Include selected "
            "ledger IDs in selected_candidate_ids. Do not omit a computed candidate just because "
            "Review did not shortlist it. Register genuinely new answers with retain_candidate "
            "before selecting their IDs. Never replace a supported candidate with a more complex "
            "filter without evidence distinguishing them.\n"
            f"Offered alternatives and required IDs: {json.dumps(alternatives, ensure_ascii=False)}\n"
            "Call finish with the typed fields in its schema: selected_alternative (one of those IDs, "
            "'new', or 'none'), selection_reason, alternative_assessments (one per offered ID, each with "
            "alternative_id, decision=selected/rejected/unresolved, reason, evidence_paths), evidence_paths, "
            "candidate_answer, correction_steps (1-5 steps). Reserve your last turn for finish; "
            "conclude with the evidence you have instead of starting another exploration. "
            "Keep selection_reason under 2000 characters and each alternative reason under 1500. "
            "Do not dismiss a plausible alternative just because it differs from the actor; "
            "mark it unresolved and request discriminating evidence when the workspace cannot decide."
        )
        try:
            with scoped_run_context(f"round-{round_record.index:03d}-reflect"):
                result = await self._critic(
                    "reflect", workspace, alternative_ids=alternative_ids
                ).run(task)
        except Exception as exc:
            _log.warning("ReAct reflection failed", exc_info=True)
            round_record.phase_errors["reflect"] = f"{type(exc).__name__}: {exc}"
            return "", 0, 0
        round_record.reflect_trajectory = result.trajectory
        in_tok, out_tok = (
            result.trajectory.total_input_tokens,
            result.trajectory.total_output_tokens,
        )
        try:
            if not result.completed:
                raise ValueError("Reflect did not return a completed finish.answer")
            plan = ReflectionPlan.model_validate_json(result.answer)
            validate_selection(plan, alternative_ids)
            if ledger is not None:
                ledger.require_coverage(plan.candidate_assessments)
            round_record.reflection_plan = plan.model_dump()
            guidance = format_guidance(plan)
        except ValueError as exc:
            round_record.phase_errors["reflect"] = str(exc)
            await self._save_context(
                workspace,
                round_record.index,
                "reflect",
                {
                    "completed": False,
                    "error": str(exc),
                    "guidance": "",
                    "trajectory": dataclasses.asdict(result.trajectory),
                },
            )
            return "", in_tok, out_tok
        await self._save_context(
            workspace,
            round_record.index,
            "reflect",
            {
                "completed": True,
                "plan": round_record.reflection_plan,
                "guidance": guidance,
                "trajectory": dataclasses.asdict(result.trajectory),
            },
        )
        return guidance, in_tok, out_tok

    def _persist(self, question, answer, record):
        from observability import persist_trajectory

        payload = dataclasses.asdict(record)
        payload["turns"] = [dataclasses.asdict(t) for t in record.turns]
        payload["total_input_tokens"] = record.total_input_tokens
        payload["total_output_tokens"] = record.total_output_tokens
        persist_trajectory(
            runner_name="reactxen-agent",
            model=self._model,
            question=question,
            answer=answer,
            trajectory=payload,
        )

    async def run(self, question: str) -> AgentResult:
        record = ReActXenTrajectory(
            started_at=_dt.datetime.now(_dt.UTC).isoformat(),
            phase_configuration={
                "actor_model": self._model,
                "critic_model": self._critic_model,
                "critic_mode": self._critic_mode,
                "review_max_turns": self._review_max_turns,
                "reflect_max_turns": self._reflect_max_turns,
                "reflect_finish_repair_attempts": self._reflect_finish_repair_attempts,
                "review_finish_repair_attempts": self._review_finish_repair_attempts,
                "critic_mcp_servers": [],
                "sandbox_network": "none",
                "strict_finish": True,
                "candidate_retention": "append_only_live_ledger",
                "distill_question_enabled": self._distill_question_enabled,
            },
        )
        try:
            if self._critic_mode == "react" and self._review_enabled:
                configured = self._stirrup_kwargs.get("workspace_dir")
                if configured is None:
                    parent = os.environ.get("AGENT_TRAJECTORY_DIR")
                    base = (
                        Path(parent).resolve().parent / "workspaces"
                        if parent
                        else Path(tempfile.gettempdir()) / "reactxen-workspaces"
                    )
                else:
                    base = Path(configured).expanduser().resolve()
                workdir = base / f"run-{uuid.uuid4().hex}"
                workdir.mkdir(parents=True, exist_ok=False)
                record.workspace_dir = str(workdir.resolve())
                prototype = StirrupAgentRunner(
                    self._llm,
                    self._server_paths,
                    code_enabled=True,
                    code_backend=self._stirrup_kwargs.get("code_backend", "docker"),
                    workspace_dir=workdir,
                    preserve_workspace=True,
                    network_disabled=True,
                )
                async with shared_workspace(
                    prototype._build_code_provider()
                ) as workspace:
                    ledger = CandidateLedger(workspace, record.candidate_ledger)
                    workspace.candidate_ledger = ledger
                    workspace.tool = ledger.wrap_code_tool(workspace.tool)
                    await ledger.flush()
                    try:
                        return await self._run_loop(question, record, workspace)
                    finally:
                        await ledger.flush()
            return await self._run_loop(question, record, None)
        except BaseException:
            record.stopped_because = "interrupted_or_error"
            raise
        finally:
            answer = record.final_round.answer if record.final_round else ""
            self._persist(question, answer, record)

    async def _run_loop(self, question, record, workspace):
        self_ask = (
            SelfAskTool(complete=self._complete) if self._self_ask_enabled else None
        )
        guidance = ""
        clarified_question = ""
        answer = ""
        for index in range(self._max_rounds):
            ledger = (
                getattr(workspace, "candidate_ledger", None)
                if workspace is not None
                else None
            )
            ledger = ledger if isinstance(ledger, CandidateLedger) else None
            if ledger is not None:
                ledger.phase = f"round-{index:03d}-actor"
            notice = actor_round_notice(
                index, self._max_rounds, review_enabled=self._review_enabled
            )
            final_attempt = not self._review_enabled or index == self._max_rounds - 1
            prompt = (
                AGENT_SYSTEM_PROMPT
                + as_prompt_block(guidance)
                + clarification_block(question, clarified_question)
                + notice
            )
            if workspace is not None:
                prompt += (
                    "\nThis workspace persists across actor, review and reflect phases. "
                    "Reuse mcp_results/ snapshots and prior scripts when valid. "
                    "Review/Reflect cannot retrieve new data; collect any missing evidence "
                    "they identify with your MCP tools. Preserve original snapshots. "
                    "Retain every plausible computed answer: labelled entity/count/value tables "
                    "and scalar JSON answers are retained automatically in candidate-ledger/index.json. "
                    "Use retain_candidate for other answer formats, naming the filter/interpretation "
                    "and evidence. Revisit prior ledger candidates and the review/reflection decisions "
                    "before choosing a final answer; do not silently replace earlier results."
                )
            core = self._core(prompt, self_ask)
            if workspace is not None:
                core._code_provider = workspace
            if ledger is not None:
                core._extra_tools.append(ledger.registration_tool())
            core._persist_result = False
            with scoped_run_context(f"round-{index:03d}-actor"):
                result = await core.run(question)
            answer = result.answer
            if ledger is not None:
                # Capture non-code tool outputs and the actual submitted answer too.
                for call in result.trajectory.all_tool_calls:
                    if call.name != "code_exec":
                        await ledger.observe(
                            call.output, tool=call.name, arguments=call.input
                        )
                await ledger.observe(answer, tool="actor_final_answer")
            r = RoundRecord(
                index=index,
                answer=answer,
                trajectory=result.trajectory,
                actor_completed=result.completed,
                actor_round_notice=notice,
                final_actor_attempt=final_attempt,
                advisory_clarified_question=clarified_question,
                self_ask_calls=self_ask.reset_round() if self_ask else 0,
                self_ask_input_tokens=self_ask.input_tokens if self_ask else 0,
                self_ask_output_tokens=self_ask.output_tokens if self_ask else 0,
            )
            if self_ask:
                self_ask.input_tokens = self_ask.output_tokens = 0
            record.rounds.append(r)
            self._persist(question, answer, record)
            if not result.completed:
                record.stopped_because = "actor_incomplete"
                break
            if not self._review_enabled:
                record.stopped_because = "review_disabled"
                break
            if workspace is not None:
                r.review = await self._react_review(question, r, workspace)
            else:
                r.review = await review(
                    complete=self._complete,
                    question=question,
                    answer=answer,
                    trajectory=result.trajectory,
                    declared_evidence=self._declared_evidence,
                    expect_json=self._expect_json,
                )
            self._persist(question, answer, record)
            _log.info("round %d: %s", index, r.review.outcome.value)
            if "review" in r.phase_errors:
                record.stopped_because = "review_incomplete"
                break
            if r.review.outcome is Outcome.ACCOMPLISHED:
                record.stopped_because = "accomplished"
                break
            if index == self._max_rounds - 1:
                record.stopped_because = "max_rounds"
                break
            if workspace is not None:
                guidance, in_tok, out_tok = await self._react_reflect(
                    question, r, workspace
                )
            else:
                guidance, in_tok, out_tok = await reflect(
                    complete=self._complete,
                    question=question,
                    answer=answer,
                    trajectory=result.trajectory,
                    verdict=r.review,
                )
            r.reflection = guidance
            r.reflect_input_tokens = in_tok
            r.reflect_output_tokens = out_tok
            if not guidance:
                record.stopped_because = (
                    "reflection_incomplete"
                    if "reflect" in r.phase_errors
                    else "no_progress"
                )
                break
            if (
                self._distill_question_enabled
                and index == 0
                and workspace is not None
                and needs_distillation(r.review_plan, r.reflection_plan)
            ):
                signals = {
                    "population_verified": r.review_plan.get("population_verified"),
                    "question_alignment_verified": r.review_plan.get(
                        "question_alignment_verified"
                    ),
                    "alignment_findings": r.review_plan.get("alignment_findings", ""),
                    "unresolved_interpretations": list(
                        dict.fromkeys(
                            a.get("interpretation", "")
                            for a in (
                                r.review_plan.get("alternatives", [])
                                + r.review_plan.get("candidate_comparisons", [])
                            )
                            if a.get("status") == "plausible"
                        )
                    ),
                }
                with scoped_run_context("round-000-distill"):
                    r.distillation = await distill_question(
                        self._distill_complete, question, signals
                    )
                r.distill_input_tokens = r.distillation.get("input_tokens", 0)
                r.distill_output_tokens = r.distillation.get("output_tokens", 0)
                if r.distillation.get("applied"):
                    clarified_question = r.distillation["clarified_question"]
                await self._save_context(workspace, index, "distill", r.distillation)
            if workspace is not None:
                guidance = (
                    f"Review report and alternatives: context/round-{index:03d}/review.json\n"
                    f"Structured reflection decision: context/round-{index:03d}/reflect.json\n"
                    + guidance
                )
                await self._save_context(
                    workspace,
                    index,
                    "actor-handoff",
                    {
                        "appended_guidance": as_prompt_block(guidance),
                        "advisory_question_block": clarification_block(
                            question, clarified_question
                        ),
                        "original_question_unchanged": question,
                        "review_context": f"context/round-{index:03d}/review.json",
                        "reflect_context": f"context/round-{index:03d}/reflect.json",
                    },
                )
            r.reflection = guidance
            self._persist(question, answer, record)
        return AgentResult(
            question=question,
            answer=answer,
            trajectory=record,
            completed=record.stopped_because
            not in {
                "actor_incomplete",
                "review_incomplete",
                "reflection_incomplete",
                "interrupted_or_error",
            },
        )
