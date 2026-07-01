# Three-step contract (same as ToolUniverse)

```python
from mcphub import ToolUniverse

tu = ToolUniverse()                          # 1. init
tu.load_tools()                              # 2. load (connect + discover)
tu.run({                                     # 3. run
    "name": "iot.sensors",
    "arguments": {"site_name": "MAIN", "asset_id": "Chiller 6"},
})
tu.close()
```

`load_tools(servers=[...])` limits to specific servers. Tools are namespaced
`<server>.<tool>`; a bare name (e.g. `sensors`) also works when unambiguous.
A shorthand `tu.run("iot.sensors", {...})` is accepted too.

## Discovery

```python
tu.find_tools("failure mode")     # keyword search over loaded tools
tu.list_tools()                   # all loaded tool + workflow names
tu.list_tools("fmsr")             # tool names for one server
tu.tool_specification("iot.sensors")
```

## Workflows

Composed workflows run through the **same `run` entrypoint**:

```python
tu.run({"name": "chiller_triage", "arguments": {"asset_id": "Chiller 6"}})
```

Add one by writing `fn(tu, **arguments)` in `workflows.py` and listing its name
in `REGISTERED` (or `tu.register_workflow("name", fn)` at runtime). Built in:
`chiller_triage` (sensors → failure modes → mapping → work order) and
`sensor_inventory_gap` (installed vs measured sensors).

## Run

```bash
uv sync
docker compose -f src/couchdb/docker-compose.yaml up -d   # iot / wo data
cp .env.public .env                                       # WATSONX_* for fmsr
uv run python examples/quickstart_tooluniverse.py
```

Each server launches via `uv run <server>-mcp-server` and inherits your
environment. Override with `ToolUniverse(servers={...})` if you run outside `uv`.