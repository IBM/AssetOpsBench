"""Comparable whole-invocation timing and explicit missing-data metrics."""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
from collections import Counter


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def suite_hash(files):
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode()); digest.update(b'\0'); digest.update(path.read_bytes())
    return digest.hexdigest()


def versions(agent):
    package = {'claude':'claude-agent-sdk', 'openai':'openai-agents'}.get(agent)
    try: sdk = importlib.metadata.version(package) if package else None
    except importlib.metadata.PackageNotFoundError: sdk = None
    command = {'claude':'claude', 'codex':os.environ.get('BENCHMARK_CODEX_EXECUTABLE','codex')}.get(agent)
    cli = None
    if command and shutil.which(command):
        try:
            process = subprocess.Popen([command, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            cli = process.communicate(timeout=5)[0].strip() or None
        except subprocess.TimeoutExpired:
            process.kill(); process.communicate()
    return {'sdk_version': sdk, 'cli_version': cli}


def invoke(command, *, env, log, timeout, record_path, record):
    """Measure CLI startup, SDK setup, tools, final answer, persistence and exit."""
    start = time.perf_counter()
    record.update(status='running', execution_start=dt.datetime.now(dt.UTC).isoformat(),
                  execution_end=None, execution_duration_ms=None, error=None)
    write_json(record_path, record)
    process = None
    try:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = ('timed_out' if isinstance(error, subprocess.TimeoutExpired) else
                            'cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed')
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        raise
    finally:
        record['execution_end'] = dt.datetime.now(dt.UTC).isoformat()
        record['execution_duration_ms'] = (time.perf_counter()-start)*1000
        write_json(record_path, record)


def read_events(path):
    if not path.exists(): return []
    events=[]
    for line in path.read_text().splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError as error:
            events.append({'kind':'trace_decode_error','error':str(error)})
    return events


def observed_metrics(events):
    """No provider/network/database facts are inferred from a tool name."""
    from observability.benchmark_trace import tool_result_error
    tools = [{**e, "error":e.get("error") or tool_result_error(e.get("output"))} for e in events if e['kind']=='tool_end']
    completed_ids={e.get('id') for e in tools}
    tools += [{**e, 'status':'incomplete', 'duration_ms':None, 'output':None, 'error':None}
              for e in events if e['kind']=='tool_start' and e.get('id') not in completed_ids]
    requests = [e for e in events if e['kind']=='model_request_end']
    summaries = [e.get('payload', {}) for e in events if e['kind']=='provider_summary']
    usage = summaries[-1].get('usage') if summaries else None
    if not isinstance(usage, dict): usage = None
    usages = [e.get('usage') for e in requests if isinstance(e.get('usage'), dict)]
    def normalize(u):
        u=dict(u)
        out=u.get('output_tokens_details') or {}; inp=u.get('input_tokens_details') or {}
        if isinstance(out,dict) and out.get('reasoning_tokens') is not None: u['reasoning_tokens']=out['reasoning_tokens'] or None
        if isinstance(inp,dict) and inp.get('cached_tokens') is not None: u['cached_tokens']=inp['cached_tokens'] or None
        if 'cached_input_tokens' in u: u['cached_tokens']=u['cached_input_tokens']
        if 'cache_read_input_tokens' in u or 'cache_creation_input_tokens' in u:
            u['uncached_input_tokens']=u.get('input_tokens')
            parts=[u.get(k) for k in ('input_tokens','cache_read_input_tokens','cache_creation_input_tokens')]
            u['input_tokens']=sum(parts) if all(isinstance(v,(int,float)) for v in parts) else None
        elif isinstance(u.get('input_tokens'),(int,float)) and isinstance(u.get('cached_tokens'),(int,float)):
            u['uncached_input_tokens']=u['input_tokens']-u['cached_tokens']
        return u
    usage=normalize(usage) if usage is not None else None
    usages=[normalize(u) for u in usages]
    tool_capture=any(e['kind']=='capabilities' and e.get('tool_events') for e in events)
    compaction_capture=any(e['kind']=='capabilities' and e.get('compaction_events') for e in events)
    def count(key, alternative=None):
        if usage is not None:
            value = usage.get(key, usage.get(alternative) if alternative else None)
            return value if isinstance(value, (int,float)) else None
        vals = [u.get(key, u.get(alternative) if alternative else None) for u in usages]
        return sum(vals) if vals and all(isinstance(v,(int,float)) for v in vals) else None
    repeated = Counter((e.get('server'),e.get('name'),json.dumps(e.get('arguments'),sort_keys=True)) for e in tools if isinstance(e.get('name'),str) and e.get('arguments') is not None)
    first = next((e['timestamp'] for e in events if e['kind'] in {'message','model_response_observed','model_request_end'}
                  and not e.get('error')), None)
    return {
        'input_tokens':count('input_tokens'), 'uncached_input_tokens':count('uncached_input_tokens'), 'output_tokens':count('output_tokens'),
        'reasoning_tokens':count('reasoning_tokens'),
        'cache_read_tokens':count('cache_read_input_tokens','cached_tokens'),
        'cache_write_tokens':count('cache_creation_input_tokens'),
        'time_to_first_response_timestamp':first,
        'model_requests':requests or None,
        'tool_call_count':len(tools) if tools or tool_capture else None,
        'unique_tools':sorted({e['name'] for e in tools if isinstance(e.get('name'),str)}) if tools or tool_capture else None,
        'calls_by_tool':dict(Counter(e.get('name') for e in tools)) if tools or tool_capture else None,
        'calls_by_server':dict(Counter(e.get('server') for e in tools)) if tools or tool_capture else None,
        'tool_calls':tools or ([] if tool_capture else None),
        'tool_incomplete':sum(e.get('status')=='incomplete' for e in tools) if tools or tool_capture else None,
        'tool_errors':sum(bool(e.get('error')) for e in tools) if tools or tool_capture else None,
        'tool_timeouts':None if any(e.get('status')=='incomplete' or (e.get('error') and (not isinstance(e['error'],dict) or not e['error'].get('type') or e['error'].get('type') in {'ToolError','ToolResultError'})) for e in tools) else sum(e.get('error',{}).get('type') in {'TimeoutError','TimeoutExpired','ReadTimeout','ConnectTimeout','APITimeoutError'} for e in tools if isinstance(e.get('error'),dict)) if tools or tool_capture else None,
        'repeated_identical_calls':sum(n-1 for n in repeated.values()) if tools or tool_capture else None,
        'retries':None, 'context_compactions':sum(e['kind']=='compaction' for e in events) if compaction_capture else None,
        'termination_reason':summaries[-1].get('stop_reason',summaries[-1].get('subtype')) if summaries else None,
        'turn_limit_hit': True if any(e['kind']=='run_error' and e.get('error',{}).get('type')=='MaxTurnsExceeded' for e in events) else ('max_turns' in str(summaries[-1].get('subtype',''))) if summaries and summaries[-1].get('subtype') is not None else None, 'database_writes_attempted':None, 'database_writes_succeeded':None,
        'database_records_created':None, 'database_records_changed':None,
        'actual_api_cost_usd':None, 'estimated_cost_usd':summaries[-1].get('total_cost_usd') if summaries else None,
        'provider_reported_cost_usd':summaries[-1].get('total_cost_usd') if summaries else None,
        'model_turns':len(requests) if requests else (summaries[-1].get('num_turns') if summaries else None),
    }


def summarize(records):
    """Use observed denominators; preserve availability counts for every metric."""
    durations=[r['execution_duration_ms'] for r in records if r.get('status')=='completed'
               and r.get('execution_duration_ms') is not None]
    grades=[r['grading']['result']['score'] for r in records if r.get('grading') and r['grading'].get('status')=='completed']
    grading_times=[r['grading']['duration_ms'] for r in records if r.get('grading') and r['grading'].get('duration_ms') is not None]
    rubric={}
    for key in {k for g in grades for k,v in g.get('details',{}).items() if isinstance(v,bool)}:
        values=[g['details'][key] for g in grades if isinstance(g.get('details',{}).get(key),bool)]
        # Hallucinations is an adverse rubric result: success means false.
        rubric[key]={'success_rate':sum(not x if key=='hallucinations' else x for x in values)/len(values), 'observed':len(values)}
    terminal=[r for r in records if r['status'] in {'completed','failed','timed_out'}]
    result={'attempted':len(records),'completed':sum(r['status']=='completed' for r in records),
            'graded':len(grades), 'mean_score':statistics.mean(g['score'] for g in grades if isinstance(g.get('score'),(int,float))) if any(isinstance(g.get('score'),(int,float)) for g in grades) else None,
            'median_grading_ms':statistics.median(grading_times) if grading_times else None,
            'p95_grading_ms':sorted(grading_times)[max(0,math.ceil(.95*len(grading_times))-1)] if grading_times else None, 'pass_rate':sum(g['passed'] for g in grades)/len(grades) if grades else None,
            'rubric_success_rates':rubric,
            'median_execution_ms':statistics.median(durations) if durations else None,
            'p95_execution_ms':sorted(durations)[max(0,math.ceil(.95*len(durations))-1)] if durations else None,
            'execution_time_observed':len(durations),
            'run_error_rate':sum(r['status'] in {'failed','timed_out'} for r in terminal)/len(terminal) if terminal else None,
            'cancelled':sum(r['status']=='cancelled' for r in records)}
    for key in ('tool_call_count','input_tokens','output_tokens','reasoning_tokens','cache_read_tokens','cache_write_tokens','tool_errors'):
        values=[(r.get('metrics') or {}).get(key) for r in records if (r.get('metrics') or {}).get(key) is not None]
        result[key]={'mean':statistics.mean(values) if values else None,'total':sum(values) if values else None,'observed':len(values)}
    return result


def exhausted_execution(record):
    budget=((record.get('settings') or {}).get('invocation_retry_policy') or {}).get('max_attempts',3)
    return record['status'] in {'failed','timed_out'} and record['attempt']>=budget


def summarize_cases(records, *, assigned_cases=None):
    """Known final execution failures are nonpassing; their judge scores stay missing."""
    result=summarize(records)
    errors=sum(exhausted_execution(r) for r in records)
    passed=sum(r['grading']['result']['score']['passed'] for r in records
               if (r.get('grading') or {}).get('status')=='completed')
    observed=result['graded']+errors
    result.update(pass_rate=passed/observed if observed else None,passed=passed,
                  execution_failed_cases=errors,outcome_observed=observed,
                  assigned_cases=len(records) if assigned_cases is None else assigned_cases)
    return result


def refresh_capture(record):
    """Re-read this run's observed trace, preserving its measured invocation time."""
    events=read_events(Path(record['trace_file']))
    metrics=observed_metrics(events)
    for key,value in (record.get('metrics') or {}).items():
        if key.startswith('database_'):metrics[key]=value
    first=metrics.pop('time_to_first_response_timestamp')
    start=record.get('execution_start')
    metrics['time_to_first_response_ms']=(dt.datetime.fromisoformat(first)-dt.datetime.fromisoformat(start)).total_seconds()*1000 if first and start else None
    answers=[e['timestamp'] for e in events if e['kind']=='final_answer']
    metrics['time_to_final_answer_ms']=(dt.datetime.fromisoformat(answers[-1])-dt.datetime.fromisoformat(start)).total_seconds()*1000 if answers and start else None
    record['metrics']=metrics
    record['agent_error']=next((e.get('error') for e in reversed(events) if e['kind']=='run_error'),None)
    grade=record.get('grading')
    if grade:grade['estimated_cost_usd']=grade.get('provider_reported_cost_usd')
    return record
