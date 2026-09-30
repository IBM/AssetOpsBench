"""Actual HTTP writes, including failed writes, are measured rather than guessed."""
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import httpx
from benchmark.database_audit import serve


def test_namespace_and_write_outcomes_are_authoritative(tmp_path):
    documents={}
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_HEAD(self):self.send_response(200 if self.path in documents else 404);self.end_headers()
        def do_PUT(self):
            data=self.rfile.read(int(self.headers['Content-Length']))
            if self.path.endswith('/fail'):self.send_response(500);self.end_headers();self.wfile.write(b'{"error":"failed"}');return
            documents[self.path]=json.loads(data)
            self.send_response(201);self.end_headers()
            self.wfile.write(json.dumps({'ok':True,'id':self.path.rsplit('/',1)[-1],'rev':'1-test'}).encode())
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200);self.end_headers();self.wfile.write(b'{"docs":[]}')
    source=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    proxy=serve(f'http://127.0.0.1:{source.server_port}',('fixture','secret'), 'eval_fixture_',tmp_path,0)
    for server in (source,proxy):threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        url=f'http://127.0.0.1:{proxy.server_port}/run-1/db'
        httpx.put(url+'/doc',json={'value':1}).raise_for_status()
        httpx.put(url+'/doc',json={'value':2}).raise_for_status()
        assert httpx.put(url+'/fail',json={'value':3}).status_code==500
        httpx.post(url+'/_find',json={'selector':{}}).raise_for_status()
        events=[json.loads(line) for line in (tmp_path/'run-1.jsonl').read_text().splitlines()]
        assert len(events)==3
        assert sum(e['writes_attempted'] for e in events)==3
        assert sum(e['writes_succeeded'] for e in events)==2
        assert [e['records'][0]['action'] for e in events[:2]]==['created','changed']
        assert documents=={'/eval_fixture_db/doc':{'value':2}}
        assert 'secret' not in (tmp_path/'run-1.jsonl').read_text()
    finally:
        proxy.shutdown();proxy.server_close();source.shutdown();source.server_close()


def test_real_couchdb_client_reads_through_proxy_and_run_switch(tmp_path):
    """couchdb3 discards URL path prefixes; the proxy must support root URLs."""
    import couchdb3
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_HEAD(self):
            self.send_response(200);self.end_headers()
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length',0)))
            self.send_response(200);self.send_header('Set-Cookie','AuthSession=fixture; Path=/; Max-Age=600');self.end_headers()
            self.wfile.write(b'{"ok":true}')
        def do_GET(self):
            self.send_response(200);self.end_headers()
            if self.path=='/eval_fixture_iot': self.wfile.write(b'{"db_name":"eval_fixture_iot","doc_count":2}')
            else: self.wfile.write(b'{"_id":"reading","hydrogen":118}')
        def do_PUT(self):
            self.rfile.read(int(self.headers.get('Content-Length',0)))
            self.send_response(201);self.end_headers();self.wfile.write(b'{"ok":true,"id":"record","rev":"2-test"}')
    active=tmp_path/'active.txt';active.write_text('first')
    source=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
    proxy=serve(f'http://127.0.0.1:{source.server_port}',('fixture','secret'),'eval_fixture_',tmp_path/'audit',0,run_id_file=active)
    for server in (source,proxy): threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        url=f'http://127.0.0.1:{proxy.server_port}'
        db=couchdb3.Database('iot',url=url,user='fixture',password='secret')
        assert db.info()['doc_count']==2
        assert db.get('reading')['hydrogen']==118
        assert not (tmp_path/'audit/first.jsonl').exists()  # Auth is not a database write.
        httpx.put(url+'/iot/record',json={'value':1}).raise_for_status()
        active.write_text('second')
        httpx.put(url+'/iot/record',json={'value':2}).raise_for_status()
        for run in ('first','second'):
            events=(tmp_path/f'audit/{run}.jsonl').read_text().splitlines()
            assert len(events)==1
            assert json.loads(events[0])['records']==[{'id':'record','action':'changed'}]
    finally:
        proxy.shutdown();proxy.server_close();source.shutdown();source.server_close()
