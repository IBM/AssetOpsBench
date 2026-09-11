    schema: disco.repo-provenance.v1

- graph_kind: tool-surface
- lane_a_source: IBM/AssetOpsBench, this repository, read at the commit this
  file ships with. The tool surface was extracted from source by AST rather
  than from documentation, so a tool named here is a tool that is registered.
- lane_b_libraries: none. This graph documents an MCP surface and needs no
  third-party distribution to do it.
- lane_c_standards: none reproduced. This graph makes no reference to any
  standards text, table or threshold.
- inspection_method: AST extraction of the six FastMCP server modules plus
  execution of the stdio handshake against each server
- license: Apache 2.0

## Evidence

Six stdio FastMCP servers, launched by the contract in `src/mcphub/__init__.py`:

```python
DEFAULT_SERVERS = {n: ["uv", "run", f"{n}-mcp-server"]
                   for n in ["iot", "utilities", "fmsr", "wo", "tsfm", "vibration"]}
```

Registered tool counts, extracted from the server modules: `iot` 12, `fmsr` 3,
`tsfm` 41, `vibration` 8, `utilities` 6, `wo` 15. Total 85.

`AOB_READONLY=1` removes six work-order mutation tools from the `wo` surface.
Verified by launching `wo` with and without the variable and comparing
`tools/list`.

`scripts/check_servers.py` performs the same handshake at runtime and asserts
the documented names against the live names, so this file cannot drift silently
past the code it describes.

## Excluded

- No benchmark scenario payload, scorer, reference answer or expected output was
  consulted. This graph describes the tool surface only.
- `materialize_iot` is a test helper rather than a registered tool and is
  therefore absent from the inventory, although a naive grep would find it.
