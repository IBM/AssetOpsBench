"""Run tool-enabled agents on a completed generated scenario suite."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from evaluation.loader import load_scenarios
from evaluation.models import PersistedTrajectory
from .measurement import invoke, observed_metrics, suite_hash, versions, write_json, read_events
import hashlib


AGENTS = {"claude": "agent.claude_agent.cli", "openai": "agent.openai_agent.cli", "codex": "agent.codex_agent.cli"}


def completed_scenarios(run_dir: Path):
    manifest = json.loads((run_dir / "run.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Generation must be complete before benchmarking")
    files = [run_dir / "scenarios.json"]
    if manifest.get("negative_count", 0):
        files.append(run_dir / "negative_scenarios.json")
    rows = load_scenarios(files)
    expected = manifest["config"]["num_scenarios"] + manifest["config"]["num_negative_scenarios"]
    if len(rows) != expected or len({r.id for r in rows}) != len(rows):
        raise ValueError("Scenario counts or unique IDs do not match the completed run")
    return rows, files


def run_target(run_dir: Path, output_dir: Path, name: str, agent: str, model: str,
               *, dry_run=False, limit=None, timeout=900, concurrency_level=1, judge_model=None, allow_self_judge=False, max_invocation_attempts=3):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Target name must contain only letters, numbers, underscores or hyphens")
    rows, files = completed_scenarios(run_dir)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    target_dir = output_dir / name
    trajectories = target_dir / "trajectories"
    metadata = {"generation_run": str(run_dir.resolve()), "agent": agent, "model": model}
    if (target_dir / "target.json").exists():
        if json.loads((target_dir / "target.json").read_text()) != metadata:
            raise ValueError("Existing target belongs to a different model, agent or scenario run")
    if not dry_run:
        trajectories.mkdir(parents=True, exist_ok=True)
        (target_dir / "target.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = {**os.environ, "AGENT_TRAJECTORY_DIR": str(trajectories.resolve())}
    settings = {**metadata, **versions(agent), 'suite_sha256':suite_hash(files),
                'repetition_index':int(os.environ['BENCHMARK_REPETITION_INDEX']) if os.environ.get('BENCHMARK_REPETITION_INDEX') else None,
                'provider':{'claude':'anthropic','codex':'openai','openai':'z.ai' if model.startswith('zai/') else 'openai-compatible'}[agent],
                'harness':AGENTS[agent], 'reasoning_effort':os.environ.get('GLM_REASONING_EFFORT') if model.startswith('zai/') else None,
                'token_limit':None, 'temperature':None, 'timeout_seconds':timeout,
                'retry_policy':{'scope':'provider SDK','max_retries':2} if agent=='openai' else None,
                'max_turns':30 if agent in {'claude','openai'} else None,
                'invocation_retry_policy':{'max_attempts':max_invocation_attempts,'scope':'whole scenario invocation'},
                'concurrency_level':concurrency_level, 'execution_order':'suite order, serial within target',
                'judge_model':judge_model, 'allow_same_model_judge':allow_self_judge,
                'grading_settings':{'temperature':None if judge_model and judge_model.startswith('claude-code/') else 0,'requested_temperature':0,'trajectory_character_limit':8000},
                'rubric_sha256':hashlib.sha256((Path(__file__).parents[1]/'evaluation/scorers/llm_judge.py').read_bytes()).hexdigest(),
                'database_policy':json.loads((target_dir/'environment.json').read_text()) if (target_dir/'environment.json').exists() else None,
                'token_semantics':'input includes reported cache tokens; output follows provider totals; raw usage and cache/reasoning subdivisions retained separately'}
    if not dry_run: write_json(target_dir/'settings.json',settings)
    for index, row in enumerate(rows, 1):
        run_id = f"{name}_{index:04d}"
        saved = trajectories / f"{run_id}.json"
        prior = [json.loads(p.read_text()) for p in (target_dir/'measurements').glob(f'{run_id}*.json')]
        latest = max(prior,key=lambda r:r['attempt']) if prior else None
        if latest and latest['status'] in {'failed','timed_out'} and latest['attempt'] >= max_invocation_attempts:
            print(f'[{name}] {index}/{len(rows)} exhausted {max_invocation_attempts} invocation attempts: {row.id}',flush=True)
            continue
        if saved.exists():
            record = PersistedTrajectory.from_raw(json.loads(saved.read_text()))
            if record.scenario_id != row.id or record.model != model or record.question != row.text:
                raise ValueError(f"Saved trajectory does not match scenario {row.id}")
            prior = [json.loads(p.read_text()) for p in (target_dir/'measurements').glob(f'{run_id}*.json')]
            latest = max(prior,key=lambda r:r['attempt']) if prior else None
            if latest is None or latest['status']=='completed':
                print(f"[{name}] {index}/{len(rows)} already saved: {row.id}", flush=True)
                continue
            archive=target_dir/'failed_trajectories'
            archive.mkdir(exist_ok=True)
            saved.replace(archive/f'{run_id}.attempt-{latest["attempt"]}.json')
        command = [sys.executable, "-m", AGENTS[agent], "--model-id", model,
                   "--scenario-id", row.id, "--run-id", run_id, "--json", row.text]
        print(f"[{name}] {index}/{len(rows)} {row.id}", flush=True)
        if dry_run:
            continue
        trace_file = target_dir/'traces'/f'{run_id}.jsonl'
        # A new invocation gets its own trace; prior failed attempts are retained.
        measurement_file = target_dir/'measurements'/f'{run_id}.json'
        attempt = 1
        while trace_file.exists() or measurement_file.exists():
            attempt += 1
            trace_file = target_dir/'traces'/f'{run_id}.attempt-{attempt}.jsonl'
            measurement_file = target_dir/'measurements'/f'{run_id}.attempt-{attempt}.json'
        env['AGENT_TRACE_FILE'] = str(trace_file.resolve())
        if os.environ.get('BENCHMARK_DB_PROXY_URL'):
            audit_id = run_id if attempt == 1 else f'{run_id}_attempt{attempt}'
            active_run_file = os.environ.get('BENCHMARK_DB_RUN_ID_FILE')
            if active_run_file:
                Path(active_run_file).write_text(audit_id)
                env['COUCHDB_URL']=os.environ['BENCHMARK_DB_PROXY_URL'].rstrip('/')
            else:
                env['COUCHDB_URL']=os.environ['BENCHMARK_DB_PROXY_URL'].rstrip('/')+'/'+run_id
        if os.environ.get('BENCHMARK_DB_PROXY_URL'):
            import httpx
            httpx.get(env['COUCHDB_URL'].rstrip('/')+'/',timeout=5).raise_for_status()
        record = {'schema_version':1, 'run_id':run_id, 'scenario_id':row.id,
                  'attempt':attempt, 'settings':settings, 'execution_index':index,
                  'trace_file':str(trace_file.resolve()), 'grading':None, 'metrics':None}
        try:
            with (target_dir / f"{measurement_file.stem}.log").open("w") as log:
                invoke(command, env=env, log=log, timeout=timeout,
                       record_path=measurement_file, record=record)
            if not saved.exists():
                record['status']='failed'
                record['error']={'type':'MissingTrajectory','message':'Agent exited without saving a trajectory'}
                raise RuntimeError(f"Agent did not save a trajectory for {row.id}; see {run_id}.log")
        finally:
            events=read_events(trace_file)
            record['metrics']=observed_metrics(events)
            record['agent_error']=next((e.get('error') for e in reversed(events) if e['kind']=='run_error'),None)
            audit_root=os.environ.get('BENCHMARK_DB_AUDIT_DIR')
            if audit_root:
                audit_file=Path(audit_root)/f'{audit_id if os.environ.get("BENCHMARK_DB_PROXY_URL") else run_id}.jsonl'
                actions=[json.loads(line) for line in audit_file.read_text().splitlines()] if audit_file.exists() else []
                record['database_audit']=actions
                record['metrics'].update(database_writes_attempted=sum(a['writes_attempted'] for a in actions),
                    database_writes_succeeded=sum(a['writes_succeeded'] for a in actions),
                    database_records_created=sum(r['action']=='created' for a in actions for r in a['records']),
                    database_records_changed=sum(r['action']=='changed' for a in actions for r in a['records']))
            first=record['metrics'].pop('time_to_first_response_timestamp')
            if first and record.get('execution_start'):
                from datetime import datetime
                record['metrics']['time_to_first_response_ms']=(datetime.fromisoformat(first)-datetime.fromisoformat(record['execution_start'])).total_seconds()*1000
            else: record['metrics']['time_to_first_response_ms']=None
            # The answer is known by child exit; exact answer-arrival time uses trace evidence.
            answers=[e['timestamp'] for e in events if e['kind']=='final_answer']
            record['metrics']['time_to_final_answer_ms']=None
            if answers:
                from datetime import datetime
                record['metrics']['time_to_final_answer_ms']=(datetime.fromisoformat(answers[-1])-datetime.fromisoformat(record['execution_start'])).total_seconds()*1000
            write_json(measurement_file,record)
    return trajectories, files


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--agent", choices=tuple(AGENTS), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--limit", type=int, help="Use a small subset for a smoke run")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judge-model", help="Optionally score saved trajectories with one common judge")
    parser.add_argument('--reasoning-effort', choices=['low','high','max'])
    parser.add_argument('--concurrency-level', type=int, default=1, help='Total concurrent execution targets (recorded metadata)')
    parser.add_argument('--max-invocation-attempts',type=int,default=3)
    parser.add_argument('--allow-self-judge', action='store_true', help='Allow same model in a separate judge session')
    args = parser.parse_args()
    if args.reasoning_effort:
        os.environ['GLM_REASONING_EFFORT']=args.reasoning_effort
    try:
        trajectories, files = run_target(args.run_dir, args.output_dir, args.name,
            args.agent, args.model_id, dry_run=args.dry_run, limit=args.limit, timeout=args.timeout,
            concurrency_level=args.concurrency_level, judge_model=args.judge_model, allow_self_judge=args.allow_self_judge,max_invocation_attempts=args.max_invocation_attempts)
        if args.judge_model and not args.dry_run:
            from .measured_grading import grade_target
            grade_target(args.output_dir/args.name, files, args.judge_model,
                         allow_self_judge=args.allow_self_judge)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
