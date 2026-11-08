# `reactxen_agent`

Stirrup as the ReAct core, wrapped in the ReActXen agent-family loop: a
self-ask tool, a Review/Critique agent, and a Reflect agent, bounded at three
rounds.

```
question ──► [ Stirrup / ReAct + MCP tools + self_ask ] ──► answer
                         ▲                                    │
                         │                                    ▼
                   guidance ◄── [ Reflect ] ◄── [ Review / Critique ]
                                                   preconditions first,
                                                   anchored critique second
          ≤ 3 rounds, stops early on accomplished or no guidance
```

## Why it is built this way

**Stirrup is unmodified.** Same client, same MCP servers, same system prompt
as `stirrup_agent`. Running `--no-self-ask --no-review` reduces this runner
exactly to `stirrup-agent`, so the comparison is an ablation of the loop
rather than a comparison of two agents.

**Review does not score.** The verdict decides whether to spend another round.
Pass or fail comes from the benchmark's program-computed reference. Keeping
these separate is what keeps "no language model judges an outcome" true.

**Review runs mechanical checks before it critiques.** Unaided self-critique
is unreliable (Huang et al., *LLMs Cannot Self-Correct Reasoning Yet*, ICLR
2024); anchored critique is the usable form. The deterministic checks come
from ReActXen's own Limitations section — is the result well formed, does the
named asset or sensor appear in what the tools returned, was the scenario's
declared evidence consulted — plus the repetition signal from our failure
analysis. A failed mechanical check overrides an optimistic model verdict.

**Critique proposes alternatives, not revisions.** The diagnosed failure is
method selection, not malformed output, so the critic names the construction
defect and offers up to three candidate approaches to select among.
Comparative judgement is easier for a model than absolute judgement.

**Self-ask is a tool, and invocations are counted.** As a tool rather than a
forced pre-action step, a model may never call it. `ReActXenTrajectory.self_ask_rate()`
reports calls per ReAct step. Report it per model: a rate near zero means the
condition did not run, whatever the flag said.

## Running

```bash
# full loop
uv run python -m agent.cli reactxen --scenario <id> \
    --model litellm_proxy/openai/gpt-4o --max-rounds 3

# ablation arms
... --no-review                 # self-ask only
... --no-self-ask               # review + reflect only
... --no-self-ask --no-review   # equivalent to stirrup-agent
```

Model routing uses `llm.routers.resolve_model`, so router prefixes and native
`<provider>/<model>` LiteLLM identifiers both work. `--critic-model` defaults
to `--model`; set it differently only if you intend to report a two-model
configuration, since a stronger critic than actor is a confound.

## The comparison to report

Three rounds is roughly four model passes. The honest comparator is **Stirrup
best-of-4 at matched budget**, not Stirrup pass@1 — MLE-bench reports
o1-preview going from 16.9% at pass@1 to 34.1% at pass@8 with no scaffold at
all. `ReActXenTrajectory` totals tokens across rounds so cost-normalised
numbers are computable from the saved trajectory.

Report per family, not only in aggregate. The loop supplies decomposition and
validation; it does not supply domain method. Gains concentrated in the
families whose failures were validation failures, with the method-selection
families flat, is the result that confirms the root-cause attribution. A
uniform lift across all families would disconfirm it.

## One upstream change this needs

`StirrupAgentRunner` builds its own tool set and takes no `extra_tools` or
`system_prompt` argument. This package passes both. The companion commit adds
them as optional passthroughs with existing behaviour unchanged when omitted.
Until that lands, run with `self_ask_enabled=False` and report the self-ask
arm as unavailable rather than silently running without the tool.

## Files

| file | contents |
|---|---|
| `runner.py` | `ReActXenAgentRunner`, the round loop, the critic channel |
| `review.py` | deterministic preconditions, anchored critique, verdict parsing |
| `reflect.py` | review to next-round guidance |
| `self_ask.py` | the self-ask tool and its invocation counter |
| `models.py` | `RoundRecord`, `ReviewVerdict`, `ReActXenTrajectory` |
| `prompts.py` | all prompt text, with the reasoning for each |
