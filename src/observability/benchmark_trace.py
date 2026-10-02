"""Append observed benchmark events immediately, including partial failed runs."""
from __future__ import annotations
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import threading
import time
import uuid

_lock = threading.RLock()


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if hasattr(value, 'model_dump'):
        return plain(value.model_dump(mode='json'))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None  # Unsupported SDK fields are unavailable, not fabricated.


def emit(kind, **data):
    filename = os.environ.get('AGENT_TRACE_FILE')
    if not filename:
        return
    event = {'timestamp': dt.datetime.now(dt.UTC).isoformat(), 'kind': kind, **plain(data)}
    with _lock:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as file:
            file.write(json.dumps(event, ensure_ascii=False) + '\n')


def instrument_model(model):
    original = model.get_response

    async def measured(*args, **kwargs):
        identifier = str(uuid.uuid4())
        started = time.perf_counter()
        emit('model_request_start', id=identifier, input=args[:2] if args else
             {k: kwargs.get(k) for k in ('system_instructions', 'input')})
        try:
            response = await original(*args, **kwargs)
        except BaseException as error:
            emit('model_request_end', id=identifier, duration_ms=(time.perf_counter()-started)*1000,
                 error={'type': type(error).__name__, 'message': str(error)})
            raise
        emit('model_request_end', id=identifier, duration_ms=(time.perf_counter()-started)*1000,
             response=response, usage=getattr(response, 'usage', None))
        return response
    model.get_response = measured
    return model


def stream_process(command, *, prompt, cwd, timeout, on_line):
    """Timestamp JSONL arrivals live, retaining partial output on timeout."""
    import queue
    import subprocess
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, cwd=cwd)
    lines, errors, pending = [], [], queue.Queue()
    def read_output():
        for line in process.stdout:
            pending.put(line)
        pending.put(None)
    def read_errors():
        errors.extend(process.stderr.readlines())
    threading.Thread(target=read_output, daemon=True).start()
    error_thread = threading.Thread(target=read_errors, daemon=True); error_thread.start()
    deadline = time.monotonic()+timeout
    try:
        process.stdin.write(prompt); process.stdin.close()
        while True:
            remaining = deadline-time.monotonic()
            if remaining<=0: raise subprocess.TimeoutExpired(command, timeout)
            try: line = pending.get(timeout=min(remaining,.5))
            except queue.Empty: continue
            if line is None: break
            lines.append(line); on_line(line)
        process.wait(timeout=max(.01,deadline-time.monotonic()))
        error_thread.join(timeout=1)
        return subprocess.CompletedProcess(command, process.returncode, ''.join(lines), ''.join(errors))
    except BaseException:
        process.kill(); process.wait()
        raise


def tool_result_error(output):
    """Recognize explicit MCP errors and benchmark ErrorResult payloads."""
    value = plain(output)
    if isinstance(value,str):
        try:value=json.loads(value)
        except (ValueError,TypeError):return None
    if not isinstance(value,dict):return None
    if value.get('isError') or value.get('is_error'):
        return {'type':'ToolError','message':'MCP reported an error'}
    if value.get('error'):
        return {'type':'ToolResultError','message':str(value['error'])}
    for key in ('result','structured_content','structuredContent'):
        if isinstance(value.get(key),(dict,str)):
            error=tool_result_error(value[key])
            if error:return error
    for block in value.get('content',[]) if isinstance(value.get('content'),list) else []:
        if isinstance(block,dict) and block.get('type')=='text':
            error=tool_result_error(block.get('text'))
            if error:return error
    return None
