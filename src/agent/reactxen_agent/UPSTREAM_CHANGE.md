# Companion change to `stirrup_agent/runner.py`

`reactxen_agent` wraps `StirrupAgentRunner` and needs two optional
passthroughs it does not currently expose. Both default to existing
behaviour, so no caller changes and no behavioural change to the
`stirrup-agent` leaderboard row.

## 1. `system_prompt`

The loop appends review guidance to the system prompt between rounds. Today
the prompt is hard-wired to `AGENT_SYSTEM_PROMPT`.

```python
 def __init__(
     self,
     llm: LLMBackend,
     server_paths: dict[str, Path | str] | None = None,
     *,
     model: str = _DEFAULT_MODEL,
     code_enabled: bool = False,
     ...
+    system_prompt: str | None = None,
 ) -> None:
     ...
+    self._system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
```

Then use `self._system_prompt` where `AGENT_SYSTEM_PROMPT` is read today.

## 2. `extra_tools`

Self-ask is exposed to the model as a tool, so it has to join the tool set
Stirrup builds from the MCP servers and the finish tool.

```python
+    extra_tools: Sequence[object] = (),
     ...
+    self._extra_tools = list(extra_tools)
```

and where the tool list is assembled, append `self._extra_tools` after the
MCP providers and before `ASSETOPS_FINISH_TOOL`.

**Ordering matters.** The finish tool must stay last, or a model that calls
self-ask late in a turn can end up finishing on the sub-question instead of
the request.

## Until this lands

Run with `self_ask_enabled=False` and report the self-ask arm as unavailable.
Do not let it run without the tool while the flag reads true: the counter
would record zero invocations and the arm would look like a measured null
when it was never wired up.

## Verifying it did not change anything

`reactxen-agent --no-self-ask --no-review` should reproduce `stirrup-agent`
run for run on the same seed and model. That equivalence is the baseline arm
of the ablation, so it is worth asserting in CI rather than assuming.
