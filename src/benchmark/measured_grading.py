"""Grade measured invocations using the existing scorer and separate timing."""
import datetime as dt
import json
import os
from pathlib import Path
import time
from evaluation.evaluator import Evaluator
from evaluation.models import ScenarioResult
from evaluation.report import build_report, write_reports_dir
from evaluation.scorers.llm_judge import install
from llm import make_backend
from .measurement import write_json, versions, read_events


def grade_target(target, files, judge_model, *, allow_self_judge=False, max_attempts=3):
    # Live grading and the suite's final grading pass share one owner per target.
    import fcntl
    target.mkdir(parents=True,exist_ok=True)
    with (target/'.grading.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _grade_target(target,files,judge_model,allow_self_judge=allow_self_judge,max_attempts=max_attempts)


def _grade_target(target, files, judge_model, *, allow_self_judge=False, max_attempts=3):
    install(make_backend(judge_model))
    evaluator=Evaluator(judge_model=judge_model, allow_self_judge=allow_self_judge)
    results=[]
    for path in sorted((target/'measurements').glob('*.json')):
        record=json.loads(path.read_text())
        if record['status']!='completed' or ('metrics' in record and record['metrics'] is None): continue
        if record.get('grading') and record['grading'].get('status')=='completed':
            results.append(ScenarioResult.model_validate(record['grading']['result']))
            continue
        previous = list((record.get('grading') or {}).get('attempts', []))
        grading={'runtime_versions':versions('claude') if judge_model.startswith('claude-code/') else versions('openai'), 'model':judge_model, 'separate_session':True, 'same_model_allowed':allow_self_judge,
                 'start':dt.datetime.now(dt.UTC).isoformat(), 'end':None, 'duration_ms':None,
                 'status':'running', 'result':None, 'error':None, 'attempts':previous,
                 'retry_policy':{'max_attempts':max_attempts,'scope':'judge backend/parse failures'}}
        record['grading']=grading; write_json(path,record)
        started=time.perf_counter()
        old=os.environ.get('AGENT_TRACE_FILE')
        os.environ['AGENT_TRACE_FILE']=str((target/'traces'/f"{record['run_id']}.judge.jsonl").resolve())
        try:
            for attempt in range(1, max_attempts + 1):
                attempt_started = time.perf_counter()
                attempt_record = {'start':dt.datetime.now(dt.UTC).isoformat(), 'status':'running'}
                grading['attempts'].append(attempt_record)
                os.environ['AGENT_TRACE_FILE'] = str((target/'traces'/f"{record['run_id']}.judge-{len(grading['attempts'])}.jsonl").resolve())
                try:
                    report=evaluator.evaluate(trajectories_path=target/'trajectories'/f"{record['run_id']}.json",
                        scenarios_paths=files, scenario_ids={record['scenario_id']})
                    result,=report.results
                    if result.score.rationale.startswith('judge backend error') or result.score.rationale=='judge returned unparseable JSON':
                        raise RuntimeError(result.score.rationale)
                    attempt_record['status']='completed'
                except Exception as error:
                    attempt_record.update(status='failed', error={'type':type(error).__name__,'message':str(error)})
                    if attempt == max_attempts: raise
                finally:
                    attempt_record.update(end=dt.datetime.now(dt.UTC).isoformat(),
                        duration_ms=(time.perf_counter()-attempt_started)*1000,
                        trace_file=os.environ['AGENT_TRACE_FILE'])
                    write_json(path,record)
                if attempt_record['status']=='completed': break
            grading.update(status='completed',result=json.loads(result.model_dump_json()))
            results.append(result)
        except BaseException as error:
            grading.update(status='cancelled' if isinstance(error,KeyboardInterrupt) else 'failed',
                           error={'type':type(error).__name__,'message':str(error)})
            raise
        finally:
            judge_trace=Path(os.environ['AGENT_TRACE_FILE'])
            events=read_events(judge_trace)
            payloads=[e['payload'] for e in events if e['kind']=='judge_result']
            grading['usage']=payloads[-1].get('usage') if payloads else None
            grading['provider_reported_cost_usd']=payloads[-1].get('total_cost_usd') if payloads else None
            grading['actual_api_cost_usd']=None
            grading['estimated_cost_usd']=grading['provider_reported_cost_usd']
            grading['trace_file']=str(judge_trace)
            grading['end']=dt.datetime.now(dt.UTC).isoformat()
            grading['duration_ms']=(time.perf_counter()-started)*1000
            write_json(path,record)
            if old is None: os.environ.pop('AGENT_TRACE_FILE',None)
            else: os.environ['AGENT_TRACE_FILE']=old
    write_reports_dir(build_report(results), target/'reports')
