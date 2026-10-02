"""The dashboard uses measured wall time and the latest scenario attempt."""
import json
from benchmark.live_results import snapshot


def test_latest_attempt_and_measured_duration_override_legacy_evidence(tmp_path):
    suite=tmp_path/'suite';suite.mkdir()
    (suite/'scenarios.json').write_text(json.dumps([{'id':'s1','text':'Read gas','type':'IoT'},{'id':'s2','text':'Missing data','type':'IoT'}]))
    (suite/'negative_scenarios.json').write_text(json.dumps([{'id':'s3','text':'Unavailable furan','type':'fmsr'}]))
    target=tmp_path/'target';(target/'measurements').mkdir(parents=True);(target/'trajectories').mkdir()
    (target/'trajectories/r.json').write_text(json.dumps({'scenario_id':'s1','answer':'118 ppm','trajectory':{'turns':[{'duration_ms':999999}]}}))
    old={'run_id':'r','scenario_id':'s1','attempt':1,'status':'failed','execution_duration_ms':90,'metrics':None,'grading':None}
    latest={**old,'attempt':2,'status':'completed','execution_duration_ms':2345,'metrics':{'input_tokens':None},'grading':{'status':'completed','duration_ms':20,'result':{'score':{'passed':True,'score':1,'details':{'hallucinations':False},'rationale':'Verified gas.'}}}}
    (target/'measurements/r.json').write_text(json.dumps(old))
    (target/'measurements/r.attempt-2.json').write_text(json.dumps(latest))
    result=snapshot(suite,target)
    first,pending,negative=result['rows']
    assert negative['type']=='negative'
    assert first['status']=='pass' and first['duration_ms']==2345 and first['grading_ms']==20
    assert first['grade']['score']['rationale']=='Verified gas.'
    assert pending['status']=='pending' and pending['metrics'] is None and pending['duration_ms'] is None
    assert result['summary']['completed']==1 and result['summary']['pass_rate']==1
    assert result['attempt_summary']['run_error_rate']==.5
