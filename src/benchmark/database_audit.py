"""Audit actual HTTP writes to an existing isolated CouchDB namespace.

Agents use PROXY_URL/RUN_ID as COUCHDB_URL. Authentication stays upstream;
credentials are read from environment, never written into the audit trail.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,quote
import httpx
from .measurement import write_json

_READ_POSTS={'_find','_all_docs','_changes','_revs_diff','_bulk_get','_explain'}


def serve(upstream,auth,prefix,audit_dir,port, *, run_id_file=None):
    if not re.fullmatch(r'eval_[a-z0-9_]+',prefix):
        raise ValueError('An isolated eval_ database prefix is required')
    client=httpx.Client(auth=auth,timeout=120)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def forward(self):
            parsed=urlsplit(self.path);parts=parsed.path.strip('/').split('/')
            parts = [part for part in parts if part]
            run_id=run_id_file.read_text().strip() if run_id_file is not None else parts.pop(0)
            if not re.fullmatch('[a-zA-Z0-9_-]+',run_id):self.send_error(400);return
            if parts and parts[0]=='_all_dbs':
                response=client.get(upstream+'/_all_dbs');response.raise_for_status()
                data=json.dumps([d[len(prefix):] for d in response.json() if d.startswith(prefix)]).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data);return
            if parts and parts[0] and not parts[0].startswith('_'):parts[0]=prefix+parts[0]
            remote=upstream+'/'+ '/'.join(parts)
            body=self.rfile.read(int(self.headers.get('Content-Length',0)))
            try:payload=json.loads(body) if body else None
            except (ValueError,UnicodeDecodeError):payload=None
            endpoint=parts[1] if len(parts)>1 else None
            is_write=bool(parts and parts[0].startswith(prefix)) and self.command in {'PUT','DELETE','POST'} and not (self.command=='POST' and endpoint in _READ_POSTS)
            before={}
            docs=payload.get('docs',[]) if isinstance(payload,dict) and endpoint=='_bulk_docs' else ([payload] if isinstance(payload,dict) else [])
            if is_write and parts:
                ids=[d.get('_id') for d in docs if isinstance(d,dict)]
                if endpoint and not endpoint.startswith('_'):ids.append('/'.join(parts[1:]))
                for identifier in filter(None,ids):
                    r=client.head(upstream+'/'+parts[0]+'/'+quote(identifier,safe='/'))
                    before[identifier]=r.status_code!=404
            stamp=dt.datetime.now(dt.UTC).isoformat()
            response=client.request(self.command,remote+('?' + parsed.query if parsed.query else ''),
                content=body,headers={'Content-Type':self.headers.get('Content-Type','application/json')})
            if is_write:
                try:result=response.json()
                except ValueError:result=None
                entries=result if isinstance(result,list) else [result] if isinstance(result,dict) else []
                succeeded=[r for r in entries if r.get('ok') or ('rev' in r and 'error' not in r)] if response.is_success else []
                records=[{'id':r.get('id'),'action':'deleted' if self.command=='DELETE' else
                          'changed' if before.get(r.get('id')) else 'created'} for r in succeeded]
                audit={'timestamp':stamp,'method':self.command,'path':'/'.join(parts),'request':payload,
                       'status_code':response.status_code,'response':result,
                       'writes_attempted':len(docs) if endpoint=='_bulk_docs' else 1,
                       'writes_succeeded':len(succeeded),'records':records}
                audit_dir.mkdir(parents=True,exist_ok=True)
                with (audit_dir/f'{run_id}.jsonl').open('a') as file:file.write(json.dumps(audit)+'\n')
            self.send_response(response.status_code)
            self.send_header('Content-Type',response.headers.get('Content-Type','application/json'))
            for header in ('ETag','Set-Cookie','Location'):
                if header in response.headers:self.send_header(header,response.headers[header])
            self.send_header('Content-Length',str(len(response.content)));self.end_headers()
            if self.command!='HEAD':self.wfile.write(response.content)
        do_GET=do_POST=do_PUT=do_DELETE=do_HEAD=forward
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix',required=True);parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--run-id-file',type=Path)
    args=parser.parse_args()
    server=serve(os.environ['COUCHDB_URL'].rstrip('/'),
        (os.environ.get('COUCHDB_USERNAME','admin'),os.environ.get('COUCHDB_PASSWORD','password')),
        args.prefix,args.audit_dir,args.port,run_id_file=args.run_id_file)
    server.serve_forever()


if __name__=='__main__':main()
