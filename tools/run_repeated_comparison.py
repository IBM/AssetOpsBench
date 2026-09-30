"""Keep repetition 1 and execute two fresh repetitions with the same snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark.measurement import suite_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'benchmarks/runs/2026-09-30-transformer/models')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'generated/comparisons/transformer-k3')
    args = parser.parse_args()
    baseline, output = args.baseline.resolve(), args.output_dir.resolve()
    if (output / 'experiment.json').exists():
        parser.error('Experiment already exists; use a fresh output directory.')
    config = json.loads((ROOT / 'benchmarks/generated-comparison.json').read_text())
    suite = ROOT / config['suite']
    digest = suite_hash([suite / name for name in ('scenarios.json', 'negative_scenarios.json')])
    snapshots = set()
    for spec in config['targets']:
        target = baseline / spec['name']
        settings = json.loads((target / 'settings.json').read_text())
        if settings['model'] != spec['model_id'] or settings['judge_model'] != config['judge']:
            parser.error('Reference model/judge does not match configuration.')
        if settings['suite_sha256'] != digest:
            parser.error('Configured suite differs from the reference repetition.')
        snapshots.add(settings['database_policy']['snapshot_sha256'])
    if len(snapshots) != 1:
        parser.error('Reference targets did not share one snapshot.')
    record = {'status': 'running', 'k': 3, 'baseline': str(baseline),
              'suite': str(ROOT / config['suite']), 'snapshot_sha256': snapshots.pop(),
              'start': datetime.now(timezone.utc).isoformat(), 'end': None,
              'repetitions': [{'index': 1, 'root': str(baseline), 'status': 'completed'},
                              *[{'index': i, 'root': str(output / f'repetition-{i}'), 'status': 'pending'} for i in (2, 3)]],
              'execution_policy': 'repetitions serial; five model targets concurrent; independent live grading'}
    write_json(output / 'experiment.json', record)
    try:
        for repetition in record['repetitions'][1:]:
            repetition.update(status='running', start=datetime.now(timezone.utc).isoformat())
            write_json(output / 'experiment.json', record)
            command = [sys.executable, str(ROOT / 'tools/run_generated_comparison.py'),
                       '--output-dir', repetition['root'], '--expected-snapshot-sha256', record['snapshot_sha256'],
                       '--snapshot-file', str(output / 'snapshot.json.gz'), '--repetition-index', str(repetition['index'])]
            print('Repetition', repetition['index'], 'starting', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            from benchmark.repeated_comparison import aggregate
            ids=set()
            for name in ('scenarios.json','negative_scenarios.json'):
                ids.update(str(row['id']) for row in json.loads((suite/name).read_text()))
            for spec in config['targets']:
                records=[json.loads(p.read_text()) for p in (Path(repetition['root'])/spec['name']/'measurements').glob('*.json')]
                aggregate([(repetition['index'],records)],ids,require_complete=True)
            repetition.update(status='completed', end=datetime.now(timezone.utc).isoformat())
            write_json(output / 'experiment.json', record)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        record['end'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'experiment.json', record)


if __name__ == '__main__':
    main()
