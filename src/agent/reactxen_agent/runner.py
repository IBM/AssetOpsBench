"""AgentRunner implementing the ReActXen agent-family loop over Stirrup.

Stirrup is the ReAct core, unmodified: the same client, the same MCP servers,
the same system prompt as :class:`~agent.stirrup_agent.StirrupAgentRunner`.
Everything this runner adds sits outside that loop, so a comparison against
the stirrup-agent row is a clean ablation rather than a comparison of two
different agents.

Per round: run Stirrup, review the result, reflect into guidance, run again.
Bounded at three rounds, which is where ReActXen's own reflection curve
flattens.

Two invariants this file is written to protect:

* **Review never scores.**  Pass or fail comes from the benchmark's
  program-computed reference.  The verdict here only decides whether to spend
  another round, which is what keeps "no language model judges an outcome"
  true of the benchmark.
* **The budget is reported.**  Three rounds is roughly four model passes, so
  the honest comparator is Stirrup best-of-4 at matched cost, not Stirrup
  pass@1.  ``ReActXenTrajectory`` totals tokens across rounds so that
  comparison is computable from the saved trajectory.

Model routing goes through ``llm.routers.resolve_model`` exactly as the
Stirrup runner does, so router prefixes and native ``<provider>/<model>``
LiteLLM identifiers both work.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Sequence
from pathlib import Path

from llm import LLMBackend

from .._prompts import AGENT_SYSTEM_PROMPT
from ..models import AgentResult
from ..runner import AgentRunner
from ..stirrup_agent import StirrupAgentRunner
from .models import Outcome, ReActXenTrajectory, RoundRecord
from .reflect import as_prompt_block, reflect
from .review import review
from .self_ask import SelfAskTool

_log = logging.getLogger(__name__)

MAX_ROUNDS = 3
"""ReActXen reports most models saturating within five reflection steps and
its own auto-generation loop runs at one. Three is the point past which we
measured no further gain; raise it only with the cost curve in hand."""

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
        code_enabled: Pass through to Stirrup's code track.
        **stirrup_kwargs: Forwarded to :class:`StirrupAgentRunner`.
    """

    def __init__(
        self,
        llm: LLMBackend,
        server_paths: dict[str, Path | str] | None = None,
        *,
        model: str = _DEFAULT_MODEL,
        critic_model: str | None = None,
        max_rounds: int = MAX_ROUNDS,
        self_ask_enabled: bool = True,
        review_enabled: bool = True,
        declared_evidence: Sequence[str] = (),
        expect_json: bool = False,
        code_enabled: bool = False,
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
        self._stirrup_kwargs = stirrup_kwargs

    # -- the critic channel ------------------------------------------------
    #
    # One short, non-streaming completion. Kept behind a single method so the
    # loop stays free of provider specifics and so a test can substitute a
    # deterministic stub.

    async def _complete(self, system: str, user: str) -> tuple[str, int, int]:
        """Return ``(text, input_tokens, output_tokens)`` from the critic model."""
        import litellm
        from llm.routers import resolve_model, resolve_router_creds

        resolved = resolve_model(self._critic_model)
        creds = resolve_router_creds(self._critic_model) or {}
        response = await litellm.acompletion(
            model=resolved,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            **creds,
        )
        usage = getattr(response, "usage", None)
        return (
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )

    def _core(self, system_prompt: str, self_ask: SelfAskTool | None):
        """Build the Stirrup core for one round."""
        kwargs = dict(self._stirrup_kwargs)
        if self_ask is not None:
            # Stirrup builds its own tool set, so the self-ask tool is handed
            # in through the extra_tools passthrough added in the companion
            # one-line change to StirrupAgentRunner.  With that change absent,
            # pass self_ask_enabled=False and report the arm as unavailable
            # rather than silently running without the tool.
            kwargs["extra_tools"] = [*kwargs.get("extra_tools", []), self_ask]
        return StirrupAgentRunner(
            self._llm,
            self._server_paths,
            model=self._model,
            code_enabled=self._code_enabled,
            system_prompt=system_prompt,
            **kwargs,
        )

    async def run(self, question: str) -> AgentResult:
        record = ReActXenTrajectory(started_at=_dt.datetime.now(_dt.UTC).isoformat())
        self_ask = (
            SelfAskTool(complete=self._complete) if self._self_ask_enabled else None
        )
        guidance = ""
        answer = ""

        for index in range(self._max_rounds):
            core = self._core(AGENT_SYSTEM_PROMPT + as_prompt_block(guidance), self_ask)
            result = await core.run(question)
            answer = result.answer

            round_record = RoundRecord(
                index=index,
                answer=answer,
                trajectory=result.trajectory,
                self_ask_calls=self_ask.reset_round() if self_ask else 0,
            )
            record.rounds.append(round_record)

            if not self._review_enabled:
                record.stopped_because = "review_disabled"
                break

            verdict = await review(
                complete=self._complete,
                question=question,
                answer=answer,
                trajectory=result.trajectory,
                declared_evidence=self._declared_evidence,
                expect_json=self._expect_json,
            )
            round_record.review = verdict
            _log.info(
                "round %d: %s (%d precondition failure(s), %d alternative(s))",
                index,
                verdict.outcome.value,
                len(verdict.failed_preconditions),
                len(verdict.alternatives),
            )

            if verdict.outcome is Outcome.ACCOMPLISHED:
                record.stopped_because = "accomplished"
                break
            if index == self._max_rounds - 1:
                record.stopped_because = "max_rounds"
                break

            guidance, in_tok, out_tok = await reflect(
                complete=self._complete,
                question=question,
                answer=answer,
                trajectory=result.trajectory,
                verdict=verdict,
            )
            round_record.reflect_input_tokens = in_tok
            round_record.reflect_output_tokens = out_tok

            if not guidance:
                # Nothing actionable came back; another identical round would
                # only spend budget.
                record.stopped_because = "no_progress"
                break

        return AgentResult(question=question, answer=answer, trajectory=record)
