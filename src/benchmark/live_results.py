"""Read measured scenario outcomes for live and offline comparison views."""
from datetime import datetime, timezone
import json
from pathlib import Path
from .measurement import observed_metrics, read_events, summarize, summarize_cases

NAMES = {'opus-5-5': 'Opus 5.5', 'gpt-6-astra': 'GPT-6 Astra',
         'glm-5-3-low': 'GLM 5.3', 'gpt-6-1-sol': 'GPT-6.1 Sol', 'fable-5-1': 'Fable 5.1'}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return default


def snapshot(suite, target):
    scenarios = []
    for name in ('scenarios.json', 'negative_scenarios.json'):
        batch = read_json(suite / name, [])
        if name == 'negative_scenarios.json':
            batch = [{**row, 'type':'negative'} for row in batch]
        scenarios.extend(batch)
    attempts = [read_json(p) for p in (target / 'measurements').glob('*.json')]
    attempts = [r for r in attempts if r]
    latest = {}
    for record in attempts:
        sid = record['scenario_id']
        if sid not in latest or record['attempt'] > latest[sid]['attempt']:
            latest[sid] = record
    rows = []
    for scenario in scenarios:
        sid = str(scenario['id'])
        record = latest.get(sid)
        grade = (record or {}).get('grading')
        result = grade.get('result') if grade else None
        status = (record or {}).get('status', 'pending')
        if grade and grade.get('status') == 'completed':
            status = 'pass' if result['score']['passed'] else 'fail'
        elif status == 'completed':
            status = 'grading' if grade and grade.get('status') == 'running' else 'executed'
        metrics = (record or {}).get('metrics')
        if record and metrics is None:
            metrics = observed_metrics(read_events(Path(record['trace_file'])))
        trace = read_json(target / 'trajectories' / f"{record['run_id']}.json") if record else None
        elapsed = (record or {}).get('execution_duration_ms')
        if record and status == 'running' and record.get('execution_start'):
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(record['execution_start'])).total_seconds() * 1000
        rows.append({'id': sid, 'type': scenario.get('type', 'negative'),
                     'question': scenario['text'], 'status': status,
                     'answer': (trace or {}).get('answer'), 'grade': result,
                     'duration_ms': elapsed, 'grading_ms': (grade or {}).get('duration_ms'),
                     'metrics': metrics, 'record': record})
    return {'rows': rows, 'summary': summarize_cases(list(latest.values()),assigned_cases=len(scenarios)),
            'attempt_summary': summarize(attempts), 'settings': read_json(target / 'settings.json')}
