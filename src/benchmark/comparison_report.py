"""Render fresh measured runs; never reconstruct missing historical timing."""
import argparse
import html
import json
from pathlib import Path
from .measurement import summarize, write_json


def render(root, output):
    groups={}
    for target in sorted(root.iterdir()):
        if not target.is_dir(): continue
        records=[json.loads(p.read_text()) for p in (target/'measurements').glob('*.json')]
        if records: groups[target.name]=(records,summarize(records))
    # Use the live visualizer's design for configured generated-suite comparisons.
    project = Path(__file__).resolve().parents[2]
    template = project/'tools/live_evaluation/index.html'
    suite = next((r.get('settings',{}).get('generation_run') for records,_ in groups.values() for r in records if r.get('settings',{}).get('generation_run')), None)
    if suite and template.exists():
        from .live_results import NAMES, snapshot
        config = json.loads((project/'benchmarks/generated-comparison.json').read_text())
        models = [{'key':spec['name'],'name':NAMES[spec['name']],'model_id':spec['model_id'],
                   **snapshot(Path(suite),root/spec['name'])} for spec in config['targets']]
        payload = json.dumps({'models':models,'judge':config['judge']}).replace('<', r'\u003c')
        doc = template.read_text().replace('<body>', '<body><script type="application/json" id="snapshot-data">'+payload+'</script>')
        doc = doc.replace('let data=null;', "let data=JSON.parse(document.getElementById('snapshot-data').textContent);")
        doc = doc.replace('));refresh();', '));render();')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(doc)
        write_json(output.with_suffix('.json'),{name:stats for name,(_,stats) in groups.items()})
        return
    esc=lambda value:html.escape(str(value))
    value=lambda x: '—' if x is None else f'{x:,.1f}' if isinstance(x,(int,float)) else esc(x)
    rows=[]; rubric_rows=[]
    for name,(records,stats) in groups.items():
        percentage=lambda x:'—' if x is None else f'{x:.1%}'
        rows.append(f'<tr><th>{esc(name)}</th><td>{stats["graded"]}/{stats["attempted"]}</td><td>{percentage(stats["pass_rate"])}</td><td>{value(stats["median_execution_ms"])}</td><td>{value(stats["p95_execution_ms"])}</td><td>{value(stats["tool_call_count"]["mean"])}</td><td>{value(stats["input_tokens"]["total"])}</td><td>{value(stats["output_tokens"]["total"])}</td><td>{percentage(stats["run_error_rate"])}</td></tr>')
        for key,rate in stats['rubric_success_rates'].items():
            rubric_rows.append(f'<tr><th>{esc(name)}</th><td>{esc(key.replace("_"," "))}</td><td>{rate["success_rate"]:.1%}</td><td>{rate["observed"]}</td></tr>')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Measured model comparison</title><style>body{margin:0;background:#f5f6f8;color:#172033;font:15px/1.55 system-ui}main{max-width:1400px;margin:48px auto;padding:0 24px}h1{font-size:32px}table{border-collapse:collapse;width:100%;background:white;border-radius:10px}td,th{padding:12px;text-align:left;border-bottom:1px solid #e6e8ed}th{font-weight:600}.scroll{overflow:auto}.muted{color:#667085}details{background:white;padding:18px;margin-top:12px;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}summary{cursor:pointer}</style><main><h1>Measured model comparison</h1><div class="scroll"><table><thead><tr><th>Model target</th><th>Graded / attempts</th><th>Pass rate</th><th>Median ms</th><th>p95 ms</th><th>Mean tools</th><th>Input tokens</th><th>Output tokens</th><th>Run error rate</th></tr></thead><tbody>'''
    doc+=''.join(rows)+'</tbody></table></div><h2>Rubric success rates</h2><table><tr><th>Model target</th><th>Rubric</th><th>Success</th><th>Observed</th></tr>'+''.join(rubric_rows)+'</table>'
    for name,(records,stats) in groups.items():
        doc+=f'<h2>{esc(name)}</h2><details><summary>Aggregate metrics and availability</summary><pre>{esc(json.dumps(stats,indent=2))}</pre></details>'
        for record in sorted(records,key=lambda r:(r['execution_index'],r['attempt'])):
            doc+=f'<details><summary>{esc(record["scenario_id"])} · attempt {record["attempt"]} · {esc(record["status"])}</summary><pre>{esc(json.dumps(record,indent=2))}</pre></details>'
    if not groups: doc+='<p>No measured runs.</p>'
    doc+='</main></html>'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(doc)
    write_json(output.with_suffix('.json'),{name:stats for name,(_,stats) in groups.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.root,args.output)


if __name__=='__main__':main()
