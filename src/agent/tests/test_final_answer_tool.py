"""The uniform final-result extraction: read the answer from the record_final_answer tool call.

Every runner records tool calls the same way, so scanning the trajectory for the last
record_final_answer call is a framework-agnostic way to get the answer - independent of Stirrup's
finish_params, OpenCode's event stream, etc.
"""

from agent.models import (
    FINAL_ANSWER_TOOL,
    ToolCall,
    Trajectory,
    TurnRecord,
    final_answer_from_trajectory,
)


def _turn(index: int, *tool_calls: ToolCall) -> TurnRecord:
    return TurnRecord(index=index, text="", tool_calls=list(tool_calls))


def test_reads_mcp_prefixed_record_final_answer():
    traj = Trajectory(turns=[
        _turn(0, ToolCall(name="wo__get_work_order", input={"asset": "CWC04013"})),
        _turn(1, ToolCall(name="utilities__record_final_answer",
                          input={"answer": "There are 7 open work orders on CWC04013."})),
    ])
    assert final_answer_from_trajectory(traj) == "There are 7 open work orders on CWC04013."


def test_reads_bare_tool_name_too():
    """Non-MCP frameworks may record the tool without a server prefix."""
    traj = Trajectory(turns=[_turn(0, ToolCall(name=FINAL_ANSWER_TOOL, input={"answer": "42"}))])
    assert final_answer_from_trajectory(traj) == "42"


def test_returns_none_when_not_called():
    traj = Trajectory(turns=[_turn(0, ToolCall(name="wo__get_work_order", input={}))])
    assert final_answer_from_trajectory(traj) is None


def test_prefers_the_last_call():
    traj = Trajectory(turns=[
        _turn(0, ToolCall(name="utilities__record_final_answer", input={"answer": "draft"})),
        _turn(1, ToolCall(name="utilities__record_final_answer", input={"answer": "final answer"})),
    ])
    assert final_answer_from_trajectory(traj) == "final answer"


def test_ignores_empty_or_nonstring_result():
    traj = Trajectory(turns=[
        _turn(0, ToolCall(name="utilities__record_final_answer", input={"answer": "  "})),
        _turn(1, ToolCall(name="utilities__record_final_answer", input={})),
    ])
    assert final_answer_from_trajectory(traj) is None
