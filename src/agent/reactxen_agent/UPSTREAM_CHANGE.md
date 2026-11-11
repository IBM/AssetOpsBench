# Stirrup integration

The Actor, Review and Reflect runners share Stirrup's model client, loop and
trajectory adapter. Optional constructor hooks provide a system prompt, extra
tools, an injected code provider, a role name, a typed finish tool and bounded
finish-repair steps. Existing standalone Stirrup defaults remain available.

ReActXen owns one offline Docker provider for its whole run. BorrowedWorkspace
lets each sequential session use it without closing the container at phase exit.
The actor retains its configured domain MCP servers. Review and Reflect receive
an empty server map and workspace-only tools. Their actual trajectories and
usage are embedded in the outer record rather than separate evaluator inputs.

The ReActXen CLI and scenario suite registration expose budgets, ablations,
completion-only legacy critics, optional question distillation and per-scenario
suite timeouts. The README documents setup and full benchmark execution.
