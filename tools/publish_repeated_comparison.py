# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.6"]
# ///
"""Publish portable evidence, repetition means/variation, graphs and offline HTML."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark.measurement import suite_hash, write_json
from benchmark.repeated_comparison import aggregate, load_experiment, mean_sd, snapshot
from benchmark.criterion_report import criterion_rows, publish_criterion_averages

COLORS = ['#42684f', '#60816a', '#90a98c', '#b2bc9c', '#d4b477']
RUBRICS = {'task_completion': 'Completion', 'data_retrieval_accuracy': 'Accuracy',
           'generalized_result_verification': 'Verification', 'agent_sequence_correct': 'Sequence',
           'clarity_and_justification': 'Clarity', 'hallucinations': 'No hallucinations'}


def cleaner(env_file):
    secrets = []
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*)", line)
            if match and re.search(r'KEY|TOKEN|SECRET|PASSWORD', match[1], re.I) and not match[1].endswith(('URL', 'PATH', 'FILE')):
                value = match[2].strip().strip("\"'")
                if len(value) >= 8 and value not in {'password', 'PASSWORD', match[1]} and not value.startswith('${'):
                    secrets.append(value)

    def clean(text):
        for value in secrets: text = text.replace(value, '[REDACTED]')
        text = re.sub(r'(https?://)[^/\s:@]+:[^/\s@]+@', r'\1[REDACTED]@', text)
        return text.replace(str(ROOT), '<workspace>').replace(str(Path.home()), '<home>')
    return clean


def export_repetition(source, dest, suite, clean):
    if (dest / 'models').exists():
        return
    config = json.loads((ROOT / 'benchmarks/generated-comparison.json').read_text())
    paths = {}
    for spec in config['targets']:
        target = source / spec['name']
        for path in (target / 'measurements').glob('*.json'):
            r = json.loads(path.read_text())
            grade = r.get('grading') or {}
            refs = [r['trace_file'], *[a['trace_file'] for a in grade.get('attempts', [])]]
            if grade.get('trace_file'): refs.append(grade['trace_file'])
            for ref in refs:
                if not Path(ref).is_file(): raise ValueError(f'Missing trace {Path(ref).name}')
                paths[ref] = f"models/{spec['name']}/events/{Path(ref).name}.gz"
    def portable(text):
        for old, new in sorted(paths.items(), key=lambda pair: -len(pair[0])):
            text = text.replace(old, new)
        text = text.replace(str(source), 'models').replace(str(suite), '../suite')
        return clean(text)
    for ref, relative in paths.items():
        path = dest / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(portable(Path(ref).read_text()).encode(), mtime=0))
    for spec in config['targets']:
        target = source / spec['name']; output = dest / 'models' / spec['name']
        for name in ('target.json', 'settings.json', 'environment.json'):
            write_json(output / name, json.loads(portable((target / name).read_text())))
        for path in (target / 'measurements').glob('*.json'):
            record=json.loads(portable(path.read_text()))
            if 'agent_error' not in record:
                raw=json.loads(path.read_text())
                events=[json.loads(line) for line in Path(raw['trace_file']).read_text().splitlines()]
                record['agent_error']=next((e.get('error') for e in reversed(events) if e['kind']=='run_error'),None)
                record=json.loads(clean(json.dumps(record)))
            write_json(output / 'measurements' / path.name, record)
        for path in (target / 'trajectories').glob('*.json'):
            write_json(output / 'trajectories' / path.name, json.loads(portable(path.read_text())))
    (dest / 'comparison.html').write_text(portable((source / 'comparison.html').read_text()))


def rows_for(experiment, config):
    groups = {}
    ids = set()
    suite=Path(experiment['suite'])
    digest=suite_hash([suite/name for name in ('scenarios.json','negative_scenarios.json')])
    for name in ('scenarios.json', 'negative_scenarios.json'):
        ids.update(str(s['id']) for s in json.loads((Path(experiment['suite']) / name).read_text()))
    for spec in config['targets']:
        repeats = []
        for rep in experiment['repetitions']:
            records = [json.loads(p.read_text()) for p in (Path(rep['root']) / spec['name'] / 'measurements').glob('*.json')]
            if any(r['settings']['suite_sha256'] != digest for r in records):
                raise ValueError('Published suite bytes differ from the executed suite hash')
            repeats.append((rep['index'], records))
        groups[spec['name']] = aggregate(repeats, ids, require_complete=True)
    return groups


def publish(experiment_path, dest, env_file):
    config = json.loads((ROOT / 'benchmarks/generated-comparison.json').read_text())
    experiment = load_experiment(experiment_path)
    groups = rows_for(experiment, config)  # Validate all results before writing a final average.
    clean = cleaner(env_file)
    dest.mkdir(parents=True, exist_ok=True)
    baseline = Path(experiment['repetitions'][0]['root'])
    portable = {**experiment, 'suite': 'suite'}
    portable['baseline'] = os.path.relpath(baseline, dest)
    portable['repetitions'] = []
    for rep in experiment['repetitions']:
        if rep['index'] == 1:
            relative = os.path.relpath(baseline, dest)
        else:
            export_repetition(Path(rep['root']), dest / f"repetition-{rep['index']}", Path(experiment['suite']), clean)
            relative = f"repetition-{rep['index']}/models"
        portable['repetitions'].append({**rep, 'root': relative})
    # One exact suite copy; reuse original repetition 1 evidence without duplicating it.
    for name in ('run.json', 'scenarios.json', 'negative_scenarios.json'):
        path = dest / 'suite' / name; path.parent.mkdir(exist_ok=True)
        path.write_bytes((Path(experiment['suite']) / name).read_bytes())
    write_json(dest / 'experiment.json', json.loads(clean(json.dumps(portable))))
    published = load_experiment(dest / 'experiment.json')
    groups = rows_for(published, config)
    payload = snapshot(published, config)
    sessions=[]
    for model in payload['models']:
        for row in model['rows']:
            if (row['record'].get('grading') or {}).get('status')!='completed':continue
            rep=next(rep for rep in published['repetitions'] if rep['index']==row['repetition'])
            path=Path(rep['root']).parent/row['record']['grading']['trace_file']
            raw=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
            events=[json.loads(line) for line in raw.splitlines()]
            sessions.extend(e['payload']['session_id'] for e in events if e['kind']=='judge_result')
    judged=sum(m['summary']['graded'] for m in payload['models'])
    if len(sessions)!=judged or len(set(sessions))!=judged or not all(sessions):
        raise ValueError('Successful judge sessions are missing or not distinct')
    for model in payload['models']:
        for row in model['rows']:
            r = row.get('record')
            if not r: continue
            rep = next(rep for rep in published['repetitions'] if rep['index'] == row['repetition'])
            def reference(ref):
                return os.path.relpath(Path(rep['root']).parent / ref, dest)
            r['trace_file'] = reference(r['trace_file'])
            grade = r.get('grading') or {}
            if grade.get('trace_file'): grade['trace_file'] = reference(grade['trace_file'])
            for attempt in grade.get('attempts', []): attempt['trace_file'] = reference(attempt['trace_file'])
            # Avoid duplicating large payloads already present in the row and linked full trace.
            r.pop('metrics', None)
            if grade.get('result'): grade['result'] = {'score': grade['result']['score']}
            row['grade'] = {'score': row['grade']['score']} if row.get('grade') else None
    template = (ROOT / 'tools/live_evaluation/index.html').read_text()
    data = clean(json.dumps(payload, ensure_ascii=False)).replace('<', r'\u003c')
    document = template.replace('<body>', '<body><script type="application/json" id="snapshot-data">' + data + '</script>')
    document = document.replace('let data=null;', "let data=JSON.parse(document.getElementById('snapshot-data').textContent);")
    document = document.replace('));refresh();', '));render();')
    (dest / 'comparison.html').write_text(document)
    write_json(dest / 'summary.json', groups)
    existing = dest / 'manifest.json'
    write_json(existing, {'published_at': json.loads(existing.read_text())['published_at'] if existing.exists() else datetime.now(timezone.utc).isoformat(),
                             'implementation_commit_at_publication': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                             'k': 3, 'scenario_count': 52, 'model_count': 5,'assigned_trials':780,
                             'graded_results':judged,
                             'execution_failed_cases':sum(m['summary']['execution_failed_cases'] for m in payload['models']),
                             'evidence_policy': 'Original repetition 1 is referenced; repetitions 2 and 3 include measurement-linked compressed full traces.',
                         'redaction_policy': 'Credentials and local path prefixes removed; metrics and timestamps unchanged.'})
    make_report(dest, published, config, groups, payload)


def make_report(dest, experiment, config, groups, payload):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#d8dfd9', 'text.color': '#24382a',
                         'axes.labelcolor': '#526356', 'xtick.color': '#526356', 'ytick.color': '#526356',
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'svg.hashsalt': 'transformer-k3'})
    keys = [s['name'] for s in config['targets']]
    labels = [m['name'] + (' (low)' if m['key'] == 'glm-5-3-low' else '') for m in payload['models']]
    graphs = dest / 'graphs'; graphs.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(graphs / f'{name}.png', dpi=180, bbox_inches='tight')
        svg = graphs / f'{name}.svg'
        fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
        plt.close(fig)
    criterion_table = publish_criterion_averages(
        dest, criterion_rows(groups, dict(zip(keys, labels)), repeated=True), COLORS, repeated=True)
    fig, ax = plt.subplots(figsize=(10, 3.6), layout='constrained')
    for i, (key, color) in enumerate(zip(keys, COLORS)):
        avg = groups[key]['means']['pass_rate']
        ax.barh(i, avg['mean'] * 100, height=.5, color=color, alpha=.85)
        ax.errorbar(avg['mean'] * 100, i, xerr=avg['sd'] * 100, fmt='none', ecolor='#24382a', capsize=4)
        for j, rep in enumerate(groups[key]['per_repetition']):
            ax.scatter(rep['cases']['pass_rate'] * 100, i + (j - 1) * .1, s=22, color='#24382a', zorder=5)
    ax.set_yticks(range(5), labels); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xlabel('Pass rate (%) · bar = mean, whisker = sample SD, dots = repetitions')
    ax.set_title('Transformer · three independent repetitions', loc='left', pad=14, fontweight='bold')
    save(fig, 'pass-rate')

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), layout='constrained')
    for ax, metric, title in zip(axes, ['median_execution_ms', 'p95_execution_ms'], ['Median execution · mean ± SD', 'p95 execution · mean ± SD']):
        for i, (key, color) in enumerate(zip(keys, COLORS)):
            avg = groups[key]['means'][metric]
            if avg['mean'] is None:continue
            ax.errorbar(avg['mean'] / 1000, i, xerr=avg['sd'] / 1000 if avg['sd'] is not None else None, fmt='o', color=color, capsize=4)
            for j, rep in enumerate(groups[key]['per_repetition']):
                if rep['cases'][metric] is not None:ax.scatter(rep['cases'][metric] / 1000, i + (j - 1) * .1, color=color, s=14, alpha=.35)
        ax.set_yticks(range(5), labels); ax.invert_yaxis(); ax.set_xlim(left=0)
        ax.set_xlabel('Entire agent invocation (seconds)'); ax.set_title(title, loc='left', pad=14, fontweight='bold')
    save(fig, 'execution-time')

    fig, ax = plt.subplots(figsize=(10, 3.3), layout='constrained')
    matrix = np.array([[groups[k]['means']['rubric_success_rates'][r]['mean'] * 100 for r in RUBRICS] for k in keys])
    ax.imshow(matrix, vmin=0, vmax=100, cmap=LinearSegmentedColormap.from_list('success', ['#faf9f2', '#cfdbc9', '#42684f']), aspect='auto')
    ax.set_xticks(range(6), list(RUBRICS.values()), fontsize=9); ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(6):
            sd = groups[keys[i]]['means']['rubric_success_rates'][list(RUBRICS)[j]]['sd'] * 100
            ax.text(j, i, f'{matrix[i,j]:.0f}% ± {sd:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if matrix[i,j] > 75 else '#24382a')
    ax.set_title('Rubric success · mean ± sample SD across three repetitions', loc='left', pad=14, fontweight='bold')
    save(fig, 'rubric-success')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout='constrained')
    for ax, metric, divisor, title in zip(axes, ['input_tokens', 'output_tokens', 'tool_call_count'], [1000, 1000, 1],
                                         ['Mean input · k tokens', 'Mean output · k tokens', 'Mean tool calls']):
        for i, (key, color) in enumerate(zip(keys, COLORS)):
            avg = groups[key]['means'][metric]['mean']
            if avg['mean'] is not None:
                ax.barh(i, avg['mean'] / divisor, height=.5, color=color)
                if avg['sd'] is not None: ax.errorbar(avg['mean'] / divisor, i, xerr=avg['sd'] / divisor, fmt='none', ecolor='#24382a', capsize=3)
        ax.set_yticks(range(5), labels); ax.invert_yaxis(); ax.set_title(title, loc='left', pad=14, fontweight='bold')
    save(fig, 'resources')

    fig, ax = plt.subplots(figsize=(10, 12), layout='constrained')
    scenarios = payload['models'][0]['repeated']['per_scenario']
    ids = list(scenarios)
    data = [[groups[k]['per_scenario'][sid]['pass_fraction'] for k in keys] for sid in ids]
    ax.imshow(data, vmin=0, vmax=1, cmap=LinearSegmentedColormap.from_list('frequency', ['#faf9f2', '#42684f']), aspect='auto')
    ax.set_xticks(range(5), labels, fontsize=9); ax.set_yticks(range(len(ids)), [sid.replace('transformer_', '') for sid in ids], fontsize=8)
    for i, sid in enumerate(ids):
        for j, key in enumerate(keys):
            n = sum(o['passed'] for o in groups[key]['per_scenario'][sid]['outcomes'])
            ax.text(j, i, f'{n}/3', ha='center', va='center', fontsize=8, color='white' if n >= 2 else '#24382a')
    ax.set_title('Scenario repeatability · successful repetitions / 3', loc='left', pad=14, fontweight='bold')
    save(fig, 'scenario-repeatability')

    columns = ['model', 'repetition', 'scenario_id', 'attempt','status','grading_status', 'passed', 'score', 'execution_duration_ms', 'grading_duration_ms',
               'input_tokens', 'output_tokens', 'reasoning_tokens', 'cache_read_tokens', 'cache_write_tokens',
               'tool_call_count', 'tool_errors', 'database_writes_attempted', 'database_writes_succeeded', *RUBRICS, 'judge_rationale']
    with (dest / 'cases.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator='\n'); writer.writeheader()
        for model in payload['models']:
            for row in model['rows']:
                r = row['record']; score = (row.get('grade') or {}).get('score')
                values = {k: r.get(k, (row['metrics'] or {}).get(k)) for k in columns}
                values.update(model=model['name'], repetition=row['repetition'], scenario_id=row['scenario_id'],
                              grading_status=(r.get('grading') or {}).get('status'),
                              passed=score['passed'] if score else False, score=score['score'] if score else None,
                              grading_duration_ms=row['grading_ms'],
                              judge_rationale=score['details'].get('suggestions', score['rationale']) if score else None)
                values.update({k: score['details'].get(k) if score else None for k in RUBRICS}); writer.writerow(values)
    mean_columns=['model','scenario_id','pass_fraction','observed_judgments','observed_outcomes',
                  *[f'{key}_{stat}' for key in ('score','execution_duration_ms','grading_duration_ms',
                                               'tool_call_count','input_tokens','output_tokens') for stat in ('mean','sd','observed_repetitions')]]
    with (dest/'scenario-averages.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=mean_columns,lineterminator='\n');writer.writeheader()
        for key,label in zip(keys,labels):
            for sid,result in groups[key]['per_scenario'].items():
                values={'model':label,'scenario_id':sid,'pass_fraction':result['pass_fraction'],'observed_judgments':result['observed_judgments'],
                        'observed_outcomes':result['observed_outcomes']}
                for metric in ('score','execution_duration_ms','grading_duration_ms','tool_call_count','input_tokens','output_tokens'):
                    values.update({f'{metric}_{stat}':value for stat,value in result[metric].items()})
                writer.writerow(values)

    fmt = lambda v, scale=1, precision=1: '—' if v['mean'] is None else f"{v['mean'] * scale:,.{precision}f}" + (f" ± {v['sd'] * scale:,.{precision}f}" if v['sd'] is not None else '')
    table, resources = [], []
    for key, label in zip(keys, labels):
        group = groups[key]; avg = group['means']
        rates = ' | '.join(f"{rep['cases']['pass_rate']:.1%}" for rep in group['per_repetition'])
        availability='/'.join(str(r['cases']['graded']) for r in group['per_repetition'])
        table.append(f"| {label} | {rates} | {group['median_pass_rate']:.1%} | {fmt(avg['pass_rate'],100)} | {fmt(avg['mean_score'],precision=3)} | {fmt(avg['median_execution_ms'],.001)} | {fmt(avg['p95_execution_ms'],.001)} | {fmt(avg['tool_call_count']['mean'])} | {availability} |")
        resources.append(f"| {label} | {fmt(avg['input_tokens']['total'])} | {fmt(avg['output_tokens']['total'])} | {fmt(avg['reasoning_tokens']['total'])} | {fmt(avg['run_error_rate'],100)} | {fmt(avg['tool_error_rate'],100)} | {fmt(avg['total_execution_ms'],1/60000)} |")
    attempts = sum(g['pooled_attempts']['attempted'] for g in groups.values())
    judged=sum(g['pooled_cases']['graded'] for g in groups.values())
    failed=sum(g['pooled_cases']['execution_failed_cases'] for g in groups.values())
    (dest / 'README.md').write_text(f"""# Transformer · k = 3

[Offline HTML](comparison.html) · [Individual results](cases.csv) · [Criterion averages](criterion-averages.csv) · [Scenario averages](scenario-averages.csv) · [Summary JSON](summary.json) · [Experiment](experiment.json) · [Original repetition](../2026-09-30-transformer/README.md)

The original k = 1 comparison was repeated twice on the **same 52 open-form scenarios** and the **same initial database snapshot**. Each repetition executes all five models; completed answers are graded in a fresh independent Fable 5.1 session, including Fable's own execution. This produces **156 assigned trials per model, 780 overall**, with **{judged} independent judgments**, **{failed} terminal execution failure{'s' if failed!=1 else ''}**, and {attempts} retained invocation attempts.

## Average criterion scores

The [latest AssetOpsBench paper, Sections 5.1–5.3](https://arxiv.org/html/2506.03828v4#S5) reports task completion, data retrieval accuracy and result verification separately. These averages expose the same three criterion names from our existing six-criterion Fable 5.1 judgments. Each criterion is averaged over observed True/False judgments (True = 1, False = 0) within a repetition, then the three repetition averages receive equal weight. Values show **mean ± sample SD in percentage points**; the strict overall pass gate does not affect these averages.

{criterion_table}

![Average criterion scores](graphs/criterion-averages.png)

These runs use **one successful Fable judgment per execution and three execution repetitions**. The paper uses Llama-4-Maverick and averages five judgments of each trajectory, so this is a reporting comparison rather than a reproduction of its judge protocol. GLM has 52/51/52 observed judgments; its terminal execution failure has no criterion judgments and is excluded from these criterion averages. Metric-specific counts and unrounded means/SDs on the 0–1 scale are in [criterion-averages.csv](criterion-averages.csv); per-repetition rates and pooled rates remain in `summary.json`. The overall pass rates below retain the existing six-criterion gate and all assigned trials.

## Overall pass rates

![Average pass rates](graphs/pass-rate.png)

| Model | R1 | R2 | R3 | Median pass | Mean pass ± SD (%) | Mean score ± SD | Median exec ± SD (s) | p95 exec ± SD (s) | Mean tools ± SD | Judged R1/R2/R3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Execution time

![Execution times across repetitions](graphs/execution-time.png)

Timing surrounds the entire agent invocation; grading is measured separately. Each repetition's median and nearest-rank p95 use completed scenario invocations, with observed counts retained in JSON. The table averages those **three repetition statistics**; it does not relabel the pooled median as an average median. Retry-inclusive total execution time includes every measured invocation, including failures. The interactive HTML also shows the pooled execution distribution and a separate repetition table.

## Rubric and resource use

![Average rubric success](graphs/rubric-success.png)

![Average resources](graphs/resources.png)

| Model | Input tokens / repeat ± SD | Output tokens / repeat ± SD | Reported reasoning / repeat ± SD | Run error rate ± SD (%) | Tool error rate ± SD (%) | Retry-inclusive exec / repeat ± SD (min) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(resources)}

Token totals and mean tools use the latest attempt per scenario, including reported metrics from terminal failures. Reliability and total execution time retain every failed/retried attempt; a retry is not a new independent repetition. Token availability and observed denominators are stored in JSON. Missing values remain `null` or empty CSV cells. Costs remain separated into actual billed values and provider estimates in the raw measurements.

## Scenario repeatability

![Success frequency per scenario](graphs/scenario-repeatability.png)

Each cell reports successful repetitions out of three for one scenario/model pair. These are repeated-trial pass frequencies and ordinary mean pass rates; **no “any one of three passed” metric is substituted for the average**.

## Method and evidence

Means give each finished repetition equal weight. The HTML's main cards and pass-rate plot show the **median of the three repetition pass rates**; counts and the outcome bar retain all assigned trials. The repetition table and exported graphs retain means and sample SD. Pass rates use all 52 assigned cases per repetition; an execution that exhausts its three-attempt budget is a known nonpassing outcome. Its judge score, rationale and rubric results remain unavailable, not invented as zeros or false rubric findings. Scores and rubric rates use actual judgments, with availability shown above and in JSON. Whiskers and ± values show **sample standard deviation across repetitions**, not a confidence interval or a significance claim. Missing metrics are excluded with observed counts retained. The no-hallucinations column inverts the adverse raw `hallucinations` finding.

Repetitions 2 and 3 run serially, with five execution targets in parallel within each repetition and independent grading workers alongside them. Fresh per-model namespaces start from the preserved repetition-1 snapshot. State persists across scenario order within each model/repetition, with no per-scenario reset. Scenario and snapshot hashes, exact model IDs, runtime versions, reasoning, limits and judge settings are checked before averaging; per-repetition settings and timestamps are retained.

The original environment limitations still apply: FMSR's default Watsonx backend was unavailable, and the judge uses the existing 8,000-character trajectory cap. Different native harnesses are used for Claude, Codex and GLM; GLM uses low reasoning. Fable judging itself uses an independent session, which does not remove potential same-model preference bias. See the [original method](../2026-09-30-transformer/README.md) for the full rubric and environment policy.

Original repetition-1 evidence is referenced without duplicating it. `repetition-2/` and `repetition-3/` include every retained measurement, successful evaluator trajectory, model/environment settings and compressed full observed execution/judge trace. Trace paths in the HTML are relative to this folder. Credentials and local path prefixes are removed for publication. The suite JSON bytes are unchanged. `summary.json` provides repetition summaries, equal-weight means/SDs, pooled summaries and per-scenario outcomes. `checksums.sha256` covers this folder.

Rebuild this report without model calls:

```bash
uv run tools/publish_repeated_comparison.py \\
  --experiment benchmarks/runs/2026-09-30-transformer-k3/experiment.json \\
  --output-dir benchmarks/runs/2026-09-30-transformer-k3
```

Start two additional repetitions from the original baseline:

```bash
PYTHONPATH=src .venv/bin/python tools/run_repeated_comparison.py \\
  --output-dir generated/comparisons/transformer-k3-new
```
""")
    files = sorted(p for p in dest.rglob('*') if p.is_file() and p.name != 'checksums.sha256')
    (dest / 'checksums.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n' for p in files))
    print(f'Published k=3: {judged} judgments, {failed} terminal execution failures, {attempts} retained attempts.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmarks/runs/2026-09-30-transformer-k3')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    publish(args.experiment.resolve(), args.output_dir.resolve(), args.env_file)


if __name__ == '__main__':
    main()
