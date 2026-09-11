# Running multi-turn dialogs

Stirrup runs one question. This page covers the outer loop that makes it run a
dialog, the on-disk format for authoring one, and the arms you compare.

## What Stirrup does and does not give you

`Agent.run()` rebuilds the conversation on every call. It appends a fresh
`SystemMessage` and resets `full_msg_history` to `[]`, so nothing carries from
one `run()` to the next (`src/stirrup/core/agent.py`, in the `if not resumed:`
branch). The `resume=True` path restores an interrupted run of the *same* task,
keyed by `compute_task_hash(init_msgs)`; it does not continue a conversation.

What the session *does* keep is the execution environment. `__aenter__` builds
the exec env, the MCP connections and the tool set once per session rather than
once per run.

So continuity has to come from somewhere. This implementation puts it on the
filesystem rather than in the message history.

## Why the filesystem and not message replay

Replaying prior messages as `init_msgs` is the obvious alternative. Three
things argue against it.

`Agent._get_turn_count` counts `AssistantMessage` instances across the history,
and the agent loop guard is `while _get_turn_count(...) < max_turns`. Every
replayed assistant message spends the working budget before the agent does any
new work. At `max_turns=30`, a full-trace replay can exhaust the budget by turn
four.

Replay also puts the whole prior trace in the prompt whether the turn needs it
or not. That is the context growth the dialog paper's baseline suffers from:
response length climbing from 1,100 to 6,700 characters across a dialog.

And replay is invisible. You cannot tell from a trajectory whether the agent
used turn 2's evidence or ignored it.

Mounting makes retrieval an action. The agent reads a router, decides which
earlier turn matters, and opens it. That decision lands in the trajectory as a
`code_exec` call naming a path, so cross-turn reuse becomes something you
measure rather than something you assume.

## Authoring a dialog

The format extends the scenario directory rather than replacing it:

```
scenario_1/
  question.txt            turn 1, exactly as today
  groundtruth.txt         turn 1's expected answer, exactly as today
  manifest.json
  turns/
    02/question.txt       turn 2
    02/groundtruth.txt    optional
    02/depends_on.txt     optional, e.g. "1"
    03/question.txt       turn 3
```

A scenario with no `turns/` directory loads as a one-turn dialog. The entire
existing suite is therefore already a valid set of dialogs, and no file needs
editing to keep working.

Turn 1 lives in `question.txt` and nowhere else. A `turns/01/` directory is
rejected, because two sources for the same turn drift apart.

## Running one

```bash
# m1: earlier turns mounted behind a router
uv run python -m agent.stirrup_agent.cli_dialog \
  --scenario-dir src/couchdb/scenarios_data/scenario_1 \
  --dialog-root ./dlg-1 --m-level m1

# m0: the unaided arm, every turn as if it were the first
uv run python -m agent.stirrup_agent.cli_dialog \
  --scenario-dir src/couchdb/scenarios_data/scenario_1 \
  --dialog-root ./dlg-1-m0 --m-level m0

# an ad-hoc dialog, no scenario directory
uv run python -m agent.stirrup_agent.cli_dialog --dialog-root ./dlg-ad-hoc \
  --turn "What sensors are on Chiller 6?" \
  --turn "Which of those has drifted this month?"
```

## The M levels

| Level | Mounts | Measures |
| --- | --- | --- |
| `m0` | nothing | How much the dialog actually depends on its history |
| `m1` | earlier turns, behind a router | The treatment |
| `m1-full` | the same tree, no routing discipline | Routing, separated from mere availability |

`m0` is to dialogs what `k0` is to skills: it mounts nothing and appends
nothing, so it is the honest baseline. Record the M level and the K level in
every results row. They move the leaderboard the way a model change does.

## What a run leaves behind

```
dlg-1/
  turn-01/              turn 1's preserved workspace
  turn-02/              turn 2's, with turn 1 mounted during the run
  turn-03/
  _staged/turn-02/      exactly what turn 2 was given
  _staged/turn-03/      exactly what turn 3 was given
  dialog.json           ask, answer, duration, tool calls, per turn
```

`_staged/` is the audit trail. It is the mounted tree as the agent saw it, kept
after the run, so a claim about what turn 3 could have known is checkable rather
than argued.

## Checking the mount reached the agent

```bash
ls dlg-1/_staged/turn-02/turn-router/SKILL.md     # the router turn 2 was given
ls dlg-1-m0/_staged 2>/dev/null                   # must not exist under m0
grep -l "turns/turn-" dlg-1/turn-0*/              # turns the agent actually opened
```

The third command is the one that matters. It separates a dialog where the
agent consulted its history from one where the history merely sat there.

## Per-turn cost

`dialog.json` records duration and tool calls per turn, which is what the
paper's per-turn cost table is built from. The claim to test is that turns 2
onward run faster than turn 1 because evidence is reused rather than
re-gathered. Under `m0` that speedup should disappear; if it does not, the
dialog did not need its history and the scenario is mis-authored.
