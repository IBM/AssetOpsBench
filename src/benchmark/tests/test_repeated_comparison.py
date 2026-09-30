"""Independent repeats must not be confused with retries or partial samples."""
from copy import deepcopy
import pytest
from benchmark.repeated_comparison import aggregate


def record(sid, passed, duration, *, attempt=1, tokens=100):
    return {'scenario_id': sid, 'execution_index': int(sid), 'attempt': attempt, 'status': 'completed',
            'settings': {'model': 'model-a', 'suite_sha256': 'suite', 'judge_model': 'judge',
                         'database_policy': {'snapshot_sha256': 'snapshot'}},
            'execution_duration_ms': duration,
            'metrics': {'input_tokens': tokens, 'tool_call_count': 2, 'tool_errors': 0},
            'grading': {'status': 'completed', 'duration_ms': 20,
                        'result': {'score': {'passed': passed, 'score': float(passed),
                                  'details': {'task_completion': passed, 'hallucinations': not passed}}}}}


def test_three_repeats_average_rates_and_sample_variation_not_any_success():
    result = aggregate([(1, [record('1', True, 10), record('2', True, 30)]),
                        (2, [record('1', False, 20), record('2', True, 60)]),
                        (3, [record('1', False, 30), record('2', False, 90)])], {'1', '2'}, require_complete=True)
    assert result['means']['pass_rate'] == {'mean': .5, 'sd': .5, 'observed_repetitions': 3}
    assert result['means']['median_execution_ms']['mean'] == 40
    assert result['means']['median_execution_ms']['sd'] == 20
    assert result['means']['p95_execution_ms']['mean'] == 60
    assert result['per_scenario']['1']['pass_fraction'] == pytest.approx(1 / 3)
    assert result['per_scenario']['1']['execution_duration_ms']['mean'] == 20
    assert result['means']['rubric_success_rates']['hallucinations']['mean'] == .5
    assert result['pooled_cases']['graded'] == 6


def test_retry_does_not_change_repetition_weight_or_hide_failure_cost():
    failed = record('1', False, 500)
    failed.update(status='failed', grading=None)
    result = aggregate([(1, [record('1', True, 10)]),
                        (2, [failed, record('1', False, 30, attempt=2)]),
                        (3, [record('1', False, 50)])], {'1'}, require_complete=True)
    assert result['means']['pass_rate']['mean'] == pytest.approx(1 / 3)
    assert result['means']['median_execution_ms']['mean'] == 30
    assert result['pooled_cases']['attempted'] == 3
    assert result['pooled_attempts']['attempted'] == 4
    assert result['pooled_attempts']['run_error_rate'] == .25
    assert result['means']['input_tokens']['total']['mean'] == 100


def test_partial_repeat_and_missing_tokens_are_not_zero_filled():
    result = aggregate([(1, [record('1', True, 10, tokens=None)]), (2, []), (3, [])], {'1'})
    assert result['complete_repetitions'] == 1
    assert result['means']['pass_rate']['mean'] == 1
    assert result['means']['pass_rate']['sd'] is None
    assert result['means']['input_tokens']['total']['mean'] is None
    assert result['per_scenario']['1']['observed_judgments'] == 1
    with pytest.raises(ValueError, match='incomplete'):
        aggregate([(1, [record('1', True, 10)]), (2, [])], {'1'}, require_complete=True)


@pytest.mark.parametrize('change', ['snapshot', 'suite', 'reasoning', 'judge'])
def test_different_experiment_conditions_cannot_be_averaged(change):
    a = record('1', True, 10); b = deepcopy(a)
    settings = b['settings']
    if change == 'snapshot': settings['database_policy']['snapshot_sha256'] = 'other'
    elif change == 'suite': settings['suite_sha256'] = 'other'
    elif change == 'reasoning': settings['reasoning_effort'] = 'max'
    else: settings['judge_model'] = 'other'
    with pytest.raises(ValueError, match='differ'):
        aggregate([(1, [a]), (2, [b])], {'1'})
