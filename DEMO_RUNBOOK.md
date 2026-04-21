# AssetOpsBench UI Demo Runbook

Use this checklist to launch the local browser UI for the demo from a fresh machine startup.

The short version: the browser host starts the UI bridge and calls the existing `plan-execute` backend workflow, but it does **not** start Docker CouchDB for you. Start CouchDB first, then start the browser host.

## 1. Open a Terminal

Navigate to the repository:

```bash
cd ~/AssetOpsBench
```

If your repo is somewhere else, use the actual path:

```bash
cd /home/jack/AssetOpsBench
```

## 2. Confirm Dependencies

Make sure `uv` is available:

```bash
uv --version
```

If this is a fresh clone or dependencies may be stale, install/update the project environment:

```bash
uv sync
```

## 3. Confirm Environment

Make sure the repo has a `.env` file:

```bash
ls .env
```

If it does not exist yet, create it from the public template:

```bash
cp .env.public .env
```

Then confirm the needed model/provider values are filled in.

For WatsonX models, the important variables are:

```text
WATSONX_APIKEY
WATSONX_PROJECT_ID
WATSONX_URL
```

For the local CouchDB demo data, the defaults are expected to work with Docker:

```text
COUCHDB_URL=http://localhost:5984
COUCHDB_USERNAME=admin
COUCHDB_PASSWORD=password
IOT_DBNAME=chiller
WO_DBNAME=workorder
VIBRATION_DBNAME=vibration
```

## 4. Start CouchDB

Start the local CouchDB container:

```bash
docker compose -f src/couchdb/docker-compose.yaml up -d
```

This also runs the repository's CouchDB setup scripts for the demo databases.

Verify CouchDB is reachable:

```bash
curl -X GET http://localhost:5984/
```

Optional: check that the expected databases exist:

```bash
curl -u admin:password http://localhost:5984/_all_dbs
```

## 5. Start the Browser UI Host

Start the local browser host:

```bash
uv run ui-browser-host --port 8766
```

Leave this terminal open while presenting.

Open this URL in your browser:

```text
http://127.0.0.1:8766
```

The browser host serves the UI and forwards UI actions into the existing Python backend tools. The MCP servers themselves are spawned on demand by `plan-execute`; you do not need to start each MCP server manually for the demo.

## 6. Demo Flow

Use the `Ask` tab for a single model/single question run.

Suggested question:

```text
What assets are at site MAIN?
```

Use the `Evaluation` tab for leaderboard evaluation.

Recommended demo settings:

- Keep the default evaluation questions loaded from `src/evaluation/questions.json`.
- Select two or three models first if time is limited.
- Select all models only if you have enough time for the full run.

The evaluation leaderboard shows:

- per-question scores
- average score
- average latency
- average tokens
- score per 1k tokens

Expandable response sections show per-model answers and evaluation details.

## 7. If the Port Is Busy

If you see:

```text
OSError: [Errno 98] Address already in use
```

Either stop the old browser host with `Ctrl+C` in the terminal where it is running, or choose another port:

```bash
uv run ui-browser-host --port 9000
```

Then open:

```text
http://127.0.0.1:9000
```

## 8. Stop After the Demo

Stop the browser host by pressing `Ctrl+C` in the terminal running:

```text
uv run ui-browser-host --port 8766
```

If you are done with the local database too, stop CouchDB:

```bash
docker compose -f src/couchdb/docker-compose.yaml down
```

## 9. Quick Pre-Demo Smoke Test

Before the presentation, from a clean terminal:

```bash
cd ~/AssetOpsBench
docker compose -f src/couchdb/docker-compose.yaml up -d
curl -X GET http://localhost:5984/
uv run ui-browser-host --port 8766
```

Open:

```text
http://127.0.0.1:8766
```

Check:

- the model dropdown loads
- the `Ask` tab can submit one question
- the `Evaluation` tab shows the default questions
- the evaluation model buttons are visible

If CouchDB responds and the first question works, the UI and backend flow are ready for the demo.
