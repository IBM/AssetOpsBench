"""Grade completed scenario invocations while execution continues."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from benchmark.generated_suite_runner import completed_scenarios
from benchmark.measured_grading import grade_target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--judge',default='claude-code/claude-fable-5-1')
    args=parser.parse_args()
    rows,files=completed_scenarios(args.suite)
    failures=0
    while True:
        try:
            grade_target(args.target,files,args.judge,allow_self_judge=True)
            failures=0
        except Exception as error:
            failures+=1
            print(type(error).__name__,str(error),flush=True)
            if failures>=3:raise
        records=[json.loads(p.read_text()) for p in (args.target/'measurements').glob('*.json')]
        graded={r['scenario_id'] for r in records if (r.get('grading') or {}).get('status')=='completed'}
        print(args.target.name,'graded',len(graded),'/',len(rows),flush=True)
        if len(graded)==len(rows):return
        latest={}
        for record in records:
            sid=record['scenario_id']
            if sid not in latest or record['attempt']>latest[sid]['attempt']:latest[sid]=record
        def finished(row):
            record=latest.get(row.id)
            if not record:return False
            if record['status']=='completed':return row.id in graded
            budget=record.get('settings',{}).get('invocation_retry_policy',{}).get('max_attempts',3)
            return record['status'] in {'failed','timed_out'} and record['attempt']>=budget
        if all(finished(row) for row in rows):return
        time.sleep(5)


if __name__=='__main__':main()
