"""Codex traces must preserve tool evidence for the established evaluator."""
import asyncio
import json
from pathlib import Path

import pytest

from agent.codex_agent.runner import CodexAgentRunner
from evaluation.loader import load_trajectories
from evaluation.evaluator import _trajectory_to_text
from observability import set_run_context


def test_completed_tool_evidence_and_usage_survive_persistence(tmp_path, monkeypatch):
    fake = tmp_path / "codex"
    fake.write_text('''#!/usr/bin/env python3
import json, sys
from pathlib import Path
question = sys.stdin.read()
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('Transformer temperature is 82 C.')
items = [
{'type':'item.started','item':{'type':'mcp_tool_call','id':'call1','server':'iot','tool':'history','arguments':{'asset_id':'transformer'}}},
{'type':'item.completed','item':{'type':'mcp_tool_call','id':'call1','server':'iot','tool':'history','arguments':{'asset_id':'transformer'},'result':{'content':[{'type':'text','text':'82 C'}]},'error':None}},
{'type':'item.completed','item':{'type':'mcp_tool_call','id':'call2','server':'iot','tool':'missing','arguments':{},'result':None,'error':{'message':'unknown sensor'}}},
{'type':'item.completed','item':{'type':'agent_message','text':'Transformer temperature is 82 C.'}},
{'type':'turn.completed','usage':{'input_tokens':120,'output_tokens':35}}]
for item in items: print(json.dumps(item))
''')
    fake.chmod(0o700)
    monkeypatch.setenv('AGENT_TRAJECTORY_DIR', str(tmp_path / 'traces'))
    set_run_context(run_id='codex-test', scenario_id='temperature')
    try:
        result = asyncio.run(CodexAgentRunner(executable=str(fake), server_paths={}).run('Read temperature'))
    finally:
        set_run_context(run_id=None, scenario_id=None)
    assert len(result.trajectory.all_tool_calls) == 2  # started event must not duplicate completed call
    assert result.trajectory.total_input_tokens == 120
    assert result.trajectory.total_output_tokens == 35
    record, = load_trajectories(tmp_path / 'traces')
    assert record.scenario_id == 'temperature'
    assert record.model == 'gpt-6-astra'
    evidence = _trajectory_to_text(record)
    assert '82 C' in evidence and 'unknown sensor' in evidence and 'transformer' in evidence
    assert record.answer == result.answer


def test_failed_codex_turn_is_not_persisted_as_success(tmp_path, monkeypatch):
    fake = tmp_path / 'codex'
    fake.write_text('#!/usr/bin/env python3\nprint(\'{"type":"turn.failed","error":{"message":"model unavailable"}}\')\n')
    fake.chmod(0o700)
    monkeypatch.setenv('AGENT_TRAJECTORY_DIR', str(tmp_path / 'traces'))
    with pytest.raises(RuntimeError, match='model unavailable'):
        asyncio.run(CodexAgentRunner(executable=str(fake), server_paths={}).run('Read temperature'))
    assert not list((tmp_path / 'traces').glob('*.json'))
