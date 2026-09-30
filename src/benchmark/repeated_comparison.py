"""Average independent suite repetitions, retaining attempts and missing values."""
from __future__ import annotations

from collections import defaultdict
import math
import statistics
from .measurement import summarize


def mean_sd(values):
    observed = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return {'mean': statistics.mean(observed) if observed else None,
            'sd': statistics.stdev(observed) if len(observed) > 1 else None,
            'observed_repetitions': len(observed)}


def aggregate(repetitions, scenario_ids, *, require_complete=False):
    """Equal-weight repetition means; a retry is not an independent repetition."""
    expected = set(scenario_ids)
    if len({index for index, _ in repetitions}) != len(repetitions):
        raise ValueError('Repetition indices must be distinct')
    rows, all_attempts, per_repetition = [], [], []
    reference = None
    outcomes = defaultdict(list)
    for index, attempts in repetitions:
        latest = {}
        for record in attempts:
            if record['scenario_id'] not in expected:
                raise ValueError('Repetition contains an unexpected scenario')
            settings = record['settings']
            identity = tuple(settings.get(k) for k in ('model', 'provider', 'harness', 'reasoning_effort',
                              'temperature', 'token_limit', 'timeout_seconds', 'max_turns',
                              'suite_sha256', 'judge_model', 'rubric_sha256', 'concurrency_level',
                              'sdk_version', 'cli_version', 'grading_settings'))
            identity += ((settings.get('database_policy') or {}).get('snapshot_sha256'),)
            if reference is None: reference = identity
            if identity != reference:
                raise ValueError('Repetitions differ in model, settings, suite, rubric, judge or initial snapshot')
            sid = record['scenario_id']
            if sid not in latest or record['attempt'] > latest[sid]['attempt']:
                latest[sid] = record
        cases = sorted(latest.values(), key=lambda r: r['execution_index'])
        complete = set(latest) == expected and all(r['status'] == 'completed' and
                    (r.get('grading') or {}).get('status') == 'completed' for r in cases)
        if require_complete and not complete:
            raise ValueError(f'Repetition {index} is incomplete; refusing a final average')
        stats, reliability = summarize(cases), summarize(attempts)
        calls, errors = reliability['tool_call_count']['total'], reliability['tool_errors']['total']
        reliability['tool_error_rate'] = errors / calls if errors is not None and calls else None
        per_repetition.append({'index': index, 'complete': complete, 'cases': stats, 'attempts': reliability,
                               'assigned_cases': len(expected)})
        for r in cases:
            grade = (r.get('grading') or {})
            if grade.get('status') == 'completed':
                outcomes[r['scenario_id']].append({'repetition': index, 'passed': grade['result']['score']['passed'],
                                                  'score': grade['result']['score']['score'],
                                                  'execution_duration_ms': r.get('execution_duration_ms'),
                                                  'grading_duration_ms': grade.get('duration_ms'),
                                                  **{key: (r.get('metrics') or {}).get(key) for key in
                                                     ('tool_call_count', 'input_tokens', 'output_tokens')}})
        rows.extend(cases)
        all_attempts.extend(attempts)
    completed = [r for r in per_repetition if r['complete']]
    scalar_metrics = ('pass_rate', 'mean_score', 'median_execution_ms', 'p95_execution_ms',
                      'median_grading_ms', 'p95_grading_ms')
    means = {key: mean_sd([r['cases'][key] for r in completed]) for key in scalar_metrics}
    for metric in ('tool_call_count', 'input_tokens', 'output_tokens', 'reasoning_tokens',
                   'cache_read_tokens', 'cache_write_tokens', 'tool_errors'):
        means[metric] = {kind: mean_sd([r['cases'][metric][kind] for r in completed]) for kind in ('mean', 'total')}
    means['run_error_rate'] = mean_sd([r['attempts']['run_error_rate'] for r in completed])
    means['tool_error_rate'] = mean_sd([r['attempts']['tool_error_rate'] for r in completed])
    rubric_keys = sorted({k for r in completed for k in r['cases']['rubric_success_rates']})
    means['rubric_success_rates'] = {key: mean_sd([r['cases']['rubric_success_rates'].get(key, {}).get('success_rate')
                                               for r in completed]) for key in rubric_keys}
    return {'k': len(repetitions), 'complete_repetitions': len(completed),
            'per_repetition': per_repetition, 'means': means,
            'pooled_cases': summarize(rows), 'pooled_attempts': summarize(all_attempts),
            'per_scenario': {sid: {'outcomes': outcomes[sid],
                                  'pass_fraction': statistics.mean(o['passed'] for o in outcomes[sid]) if outcomes[sid] else None,
                                  'observed_judgments': len(outcomes[sid]),
                                  **{key: mean_sd([o.get(key) for o in outcomes[sid]]) for key in
                                     ('score', 'execution_duration_ms', 'grading_duration_ms',
                                      'tool_call_count', 'input_tokens', 'output_tokens')}} for sid in sorted(expected)}}


def snapshot(experiment, config):
    """Reuse the established scenario view while retaining repetition identity."""
    import json
    from pathlib import Path
    from .live_results import NAMES, snapshot as single_snapshot

    suite = Path(experiment['suite'])
    models = []
    for spec in config['targets']:
        rows, repeats, settings = [], [], []
        ids = None
        for repetition in experiment['repetitions']:
            target = Path(repetition['root']) / spec['name']
            single = single_snapshot(suite, target)
            if ids is None: ids = {r['id'] for r in single['rows']}
            records = [json.loads(p.read_text()) for p in (target / 'measurements').glob('*.json')]
            repeats.append((repetition['index'], records))
            settings.append({'repetition': repetition['index'], 'settings': single['settings']})
            rows.extend({**row, 'id': f"{repetition['index']}:{row['id']}",
                         'scenario_id': row['id'], 'repetition': repetition['index'],
                         'question': f"R{repetition['index']} · {row['question']}"} for row in single['rows'])
        result = aggregate(repeats, ids or set())
        models.append({'key': spec['name'], 'name': NAMES[spec['name']], 'model_id': spec['model_id'],
                       'rows': rows, 'summary': result['pooled_cases'], 'attempt_summary': result['pooled_attempts'],
                       'repeated': result, 'settings': settings})
    return {'models': models, 'judge': config['judge'], 'k': experiment['k']}


def load_experiment(path):
    import json
    from pathlib import Path
    path = Path(path).resolve()
    experiment = json.loads(path.read_text())
    experiment['suite'] = str((path.parent / experiment['suite']).resolve())
    for repetition in experiment['repetitions']:
        repetition['root'] = str((path.parent / repetition['root']).resolve())
    return experiment
