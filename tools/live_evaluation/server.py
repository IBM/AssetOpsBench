"""Read-only live dashboard for measured generated-suite runs."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from benchmark.measurement import observed_metrics, read_events, summarize

UI = Path(__file__).with_name('index.html')
from benchmark.live_results import NAMES, read_json, snapshot


def main():
    config = read_json(ROOT / 'benchmarks/generated-comparison.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--suite', type=Path, default=ROOT / config['suite'])
    parser.add_argument('--target', type=Path, default=ROOT / 'generated/comparisons/transformer/opus-5-5')
    parser.add_argument('--experiment',type=Path,help='Repeated comparison experiment.json')
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/results':
                if args.experiment:
                    from benchmark.repeated_comparison import load_experiment, snapshot as repeated_snapshot
                    payload=repeated_snapshot(load_experiment(args.experiment),config)
                else:
                    models = []
                    for spec in config['targets']:
                        key = spec['name']
                        models.append({'key': key, 'name': NAMES[key], 'model_id': spec['model_id'],
                                       **snapshot(args.suite, args.target.parent / key)})
                    payload={'models':models,'judge':config['judge']}
                payload['updated']=datetime.now(timezone.utc).isoformat()
                body = json.dumps(payload).encode()
                mime = 'application/json'
            elif self.path == '/comparison.html':
                body = (args.target.parent/'comparison.html').read_bytes()
                mime = 'text/html; charset=utf-8'
            elif self.path in ('/', '/index.html'):
                body = UI.read_bytes()
                mime = 'text/html; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f'http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
