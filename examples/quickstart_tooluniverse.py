"""
Quick start — ToolUniverse-style interface. Prereqs (repo root):
    uv sync
    docker compose -f src/couchdb/docker-compose.yaml up -d
    cp .env.public .env                       # WATSONX_* for fmsr
Run:
    uv run python examples/quickstart_tooluniverse.py
"""
import json

from mcphub import ToolUniverse


def show(t, o):
    print(f"\n=== {t} ===")
    print(json.dumps(o, indent=2, default=str))


def main():
    tu = ToolUniverse()                          # 1. init
    try:
        tu.load_tools(servers=["iot", "fmsr", "wo"])   # 2. load (connect + discover)

        show("find 'failure mode'",
             [s["name"] for s in tu.find_tools("failure mode")])

        # 3. run a single tool (ToolUniverse dict form)
        show("iot.sensors", tu.run({
            "name": "iot.sensors",
            "arguments": {"site_name": "MAIN", "asset_id": "Chiller 6"},
        }))

        # A workflow runs through the same entrypoint
        show("chiller_triage", tu.run({
            "name": "chiller_triage",
            "arguments": {"asset_id": "Chiller 6", "raise_work_order": False},
        }))
    finally:
        tu.close()


if __name__ == "__main__":
    main()