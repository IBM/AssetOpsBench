"""Grading time and rubric outcomes stay separate from execution measurements."""
import json
import time
from pathlib import Path
from benchmark import measured_grading
from observability.benchmark_trace import emit


def test_same_model_separate_judge_capture_and_resume(tmp_path,monkeypatch):
    root=tmp_path/'target'; (root/'measurements').mkdir(parents=True);(root/'trajectories').mkdir()
    scenario=tmp_path/'scenarios.json'
    scenario.write_text(json.dumps([{'id':'s1','text':'Read temperature','characteristic_form':'Use the tool.'}]))
    trace={'run_id':'run1','scenario_id':'s1','runner':'claude-agent','model':'claude-fable-5-1',
           'question':'Read temperature','answer':'82 C','trajectory':{'turns':[]}}
    (root/'trajectories/run1.json').write_text(json.dumps(trace))
    path=root/'measurements/run1.json'
    path.write_text(json.dumps({'run_id':'run1','scenario_id':'s1','status':'completed',
                               'execution_duration_ms':1000,'grading':None}))
    calls=[]
    class Judge:
        def generate(self,prompt):
            calls.append(prompt);time.sleep(.02)
            emit('judge_result',payload={'usage':{'input_tokens':20,'output_tokens':30},'total_cost_usd':.01})
            return json.dumps({'task_completion':True,'data_retrieval_accuracy':True,
                'generalized_result_verification':True,'agent_sequence_correct':True,
                'clarity_and_justification':True,'hallucinations':False,'suggestions':'Evidence matches.'})
    monkeypatch.setattr(measured_grading,'make_backend',lambda _:Judge())
    monkeypatch.setattr(measured_grading,'versions',lambda _:{'cli_version':'fixture'})
    measured_grading.grade_target(root,[scenario],'claude-code/claude-fable-5-1',allow_self_judge=True)
    measured_grading.grade_target(root,[scenario],'claude-code/claude-fable-5-1',allow_self_judge=True)
    record=json.loads(path.read_text());grade=record['grading']
    assert len(calls)==1
    assert record['execution_duration_ms']==1000
    assert grade['duration_ms']>=20 and grade['end']>grade['start']
    assert grade['result']['score']['passed'] is True
    assert grade['result']['score']['details']['hallucinations'] is False
    assert grade['usage']=={'input_tokens':20,'output_tokens':30}
    assert grade['actual_api_cost_usd'] is None
    assert grade['provider_reported_cost_usd']==.01
    assert Path(grade['trace_file']).is_file()


def test_malformed_judge_response_is_retried_without_failing_agent(tmp_path,monkeypatch):
    root=tmp_path/'target';(root/'measurements').mkdir(parents=True);(root/'trajectories').mkdir()
    scenario=tmp_path/'scenarios.json';scenario.write_text(json.dumps([{'id':'s1','text':'Read temperature','characteristic_form':'Use the tool.'}]))
    (root/'trajectories/r.json').write_text(json.dumps({'run_id':'r','scenario_id':'s1','runner':'claude-agent','model':'claude-opus-5-5','question':'Read temperature','answer':'82 C','trajectory':{'turns':[]}}))
    path=root/'measurements/r.json';path.write_text(json.dumps({'run_id':'r','scenario_id':'s1','status':'completed','grading':None}))
    replies=iter(['not JSON',json.dumps({'task_completion':True,'data_retrieval_accuracy':True,'generalized_result_verification':True,'agent_sequence_correct':True,'clarity_and_justification':True,'hallucinations':False,'suggestions':'Valid evidence.'})])
    class Judge:
        def generate(self,prompt):return next(replies)
    monkeypatch.setattr(measured_grading,'make_backend',lambda _:Judge())
    monkeypatch.setattr(measured_grading,'versions',lambda _: {})
    measured_grading.grade_target(root,[scenario],'claude-code/claude-fable-5-1')
    record=json.loads(path.read_text());grade=record['grading']
    assert record['status']=='completed' and grade['result']['score']['passed']
    assert [a['status'] for a in grade['attempts']]==['failed','completed']
    assert grade['attempts'][0]['error']['message']=='judge returned unparseable JSON'
    assert len({a['trace_file'] for a in grade['attempts']})==2
