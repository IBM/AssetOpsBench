"""Real local processes verify invocation timing, cancellation and evidence."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from benchmark.measurement import invoke, observed_metrics, summarize


@pytest.mark.parametrize('mode,status', [('ok','completed'),('fail','failed'),('hang','timed_out')])
def test_entire_process_duration_and_failure_trace_survive(tmp_path,mode,status):
    script=tmp_path/'child.py'
    script.write_text("""import json,os,time,sys
from pathlib import Path
Path(os.environ['TRACE']).write_text(json.dumps({'kind':'tool_start','arguments':{'asset':'transformer'}})+'\\n')
time.sleep(.04)
if sys.argv[1]=='hang': time.sleep(10)
if sys.argv[1]=='fail': sys.exit(7)
""")
    record={'run_id':'fixture'}; saved=tmp_path/'measurement.json'; trace=tmp_path/'trace.jsonl'
    with (tmp_path/'output.log').open('w') as log:
        try:
            invoke([sys.executable,str(script),mode], env={**os.environ,'TRACE':str(trace)},
                   log=log, timeout=.3, record_path=saved, record=record)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
            assert mode!='ok'
    observed=json.loads(saved.read_text())
    assert observed['status']==status
    assert observed['execution_duration_ms']>=40
    assert observed['execution_end']>observed['execution_start']
    assert json.loads(trace.read_text())['arguments']=={'asset':'transformer'}
    assert bool(observed['error'])==(mode!='ok')


def test_interrupt_marks_cancelled_and_kills_child_group(tmp_path):
    parent=tmp_path/'parent.py'; saved=tmp_path/'measurement.json'; pidfile=tmp_path/'child.pid'
    parent.write_text(f"""import sys,os
from pathlib import Path
from benchmark.measurement import invoke
record={{}}
with Path({str(tmp_path/'output.log')!r}).open('w') as log:
 invoke([sys.executable,'-c',"import os,time;open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(30)"],env=os.environ.copy(),log=log,timeout=30,record_path=Path({str(saved)!r}),record=record)
""")
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[3])}
    process=subprocess.Popen([sys.executable,str(parent)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+5
        while not pidfile.exists() and time.monotonic()<deadline: time.sleep(.01)
        assert pidfile.exists()
        process.send_signal(signal.SIGINT); process.wait(timeout=5)
        assert json.loads(saved.read_text())['status']=='cancelled'
        with pytest.raises(ProcessLookupError): os.kill(int(pidfile.read_text()),0)
    finally:
        if process.poll() is None: process.kill();process.wait()


def test_comparison_denominators_exclude_missing_and_cancelled_timing():
    records=[{'status':'completed','execution_duration_ms':100,'metrics':{'input_tokens':10,'tool_call_count':0},
              'grading':{'status':'completed','result':{'score':{'passed':True,'details':{'hallucinations':False}}}}},
             {'status':'completed','execution_duration_ms':300,'metrics':{'input_tokens':None,'tool_call_count':4}},
             {'status':'cancelled','execution_duration_ms':5000,'metrics':{}}]
    result=summarize(records)
    assert result['median_execution_ms']==200
    assert result['p95_execution_ms']==300
    assert result['input_tokens']=={'mean':10,'total':10,'observed':1}
    assert result['tool_call_count']['mean']==2
    assert result['pass_rate']==1 and result['graded']==1
    assert result['rubric_success_rates']['hallucinations']['success_rate']==1
    missing=observed_metrics([])
    assert missing['tool_call_count'] is None and missing['reasoning_tokens'] is None


def test_partial_trace_keeps_incomplete_call_and_missing_provider_details(tmp_path):
    from benchmark.measurement import read_events
    path=tmp_path/'events.jsonl'
    path.write_text(json.dumps({'kind':'capabilities','tool_events':True})+'\n'+
                    json.dumps({'kind':'tool_start','id':'pending','name':'history','server':'iot','arguments':{'asset':'transformer'}})+'\n{"kind":')
    events=read_events(path)
    metrics=observed_metrics(events)
    assert metrics['tool_call_count']==1 and metrics['tool_incomplete']==1
    assert metrics['tool_calls'][0]['duration_ms'] is None
    assert metrics['reasoning_tokens'] is None
    assert metrics['retries'] is None
    assert events[-1]['kind']=='trace_decode_error'


def test_report_renders_pending_values_as_missing(tmp_path):
    from benchmark.comparison_report import render
    root=tmp_path/'results';target=root/'astra'/'measurements';target.mkdir(parents=True)
    record={'run_id':'r','scenario_id':'<asset>','attempt':1,'execution_index':1,
            'status':'running','execution_duration_ms':None,'grading':None,'metrics':None}
    (target/'r.json').write_text(json.dumps(record))
    output=tmp_path/'comparison.html';render(root,output)
    stats=json.loads(output.with_suffix('.json').read_text())['astra']
    assert stats['median_execution_ms'] is None
    assert stats['input_tokens']['total'] is None
    assert stats['run_error_rate'] is None
    assert '—' in output.read_text() and '&lt;asset&gt;' in output.read_text()


def test_cached_claude_input_is_comparable_without_double_counting_sdk_cache():
    claude=observed_metrics([{'kind':'provider_summary','payload':{'usage':{
        'input_tokens':10,'cache_read_input_tokens':80,'cache_creation_input_tokens':10,'output_tokens':12}}}])
    sdk=observed_metrics([{'kind':'provider_summary','payload':{'usage':{
        'input_tokens':100,'input_tokens_details':{'cached_tokens':80},'output_tokens':12}}}])
    assert claude['input_tokens']==sdk['input_tokens']==100
    assert claude['uncached_input_tokens']==10
    assert sdk['uncached_input_tokens']==20
    assert claude['cache_write_tokens']==10
    assert sdk['cache_write_tokens'] is None


def test_business_tool_failures_are_counted_across_harness_envelopes():
    envelopes=[{'content':[{'type':'text','text':'{"error":"data unavailable"}'}]},
               {'structured_content':{'result':{'error':'data unavailable'}}},
               '{"result":{"error":"data unavailable"}}',
               {'content':[{'type':'text','text':'{"records":[{"error":"stored document field"}]}'}]}]
    events=[{'kind':'tool_end','id':str(i),'name':'history','server':'iot','arguments':{'asset':'transformer'},'output':v,'error':None} for i,v in enumerate(envelopes)]
    metrics=observed_metrics(events)
    assert metrics['tool_call_count']==4 and metrics['tool_errors']==3
    assert metrics['tool_timeouts'] is None
    assert metrics['tool_calls'][3]['error'] is None


def test_refresh_uses_saved_evidence_without_replacing_wall_clock_or_billing(tmp_path):
    from benchmark.measurement import refresh_capture
    path=tmp_path/'trace.jsonl'
    path.write_text(json.dumps({'kind':'tool_end','timestamp':'2026-09-30T12:00:01+00:00','id':'tool1','name':'interpret_dga','server':'fmsr','output':'{"result":{"error":"LLM unavailable"}}'})+'\n'+json.dumps({'kind':'provider_summary','timestamp':'2026-09-30T12:00:02+00:00','payload':{'total_cost_usd':.05}})+'\n')
    record={'trace_file':str(path),'execution_start':'2026-09-30T12:00:00+00:00','execution_duration_ms':2345,'metrics':{'database_writes_attempted':1},'grading':{'provider_reported_cost_usd':.01}}
    result=refresh_capture(record)
    assert result['execution_duration_ms']==2345
    assert result['metrics']['tool_errors']==1
    assert result['metrics']['database_writes_attempted']==1
    assert result['metrics']['actual_api_cost_usd'] is None
    assert result['metrics']['estimated_cost_usd']==.05
    assert result['grading']['estimated_cost_usd']==.01
    assert result['metrics']['time_to_first_response_ms'] is None


def test_sdk_default_subdivisions_do_not_fabricate_provider_reporting():
    sdk=observed_metrics([{'kind':'provider_summary','payload':{'usage':{'input_tokens':20,'output_tokens':10,'input_tokens_details':{'cached_tokens':0},'output_tokens_details':{'reasoning_tokens':0}}}}])
    assert sdk['input_tokens']==20 and sdk['output_tokens']==10
    assert sdk['cache_read_tokens'] is None and sdk['reasoning_tokens'] is None
    cli=observed_metrics([{'kind':'provider_summary','payload':{'usage':{'input_tokens':20,'output_tokens':10,'cached_input_tokens':0}}}])
    assert cli['cache_read_tokens']==0
