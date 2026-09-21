#!/usr/bin/env bash
# Find the real MCP endpoint. The python client turns any 404 on POST into
# "Session terminated", which hides the status code; this shows it.
#
#   TSFM_API_KEY=... bash probe_tsfm_endpoint.sh
set -u
: "${TSFM_API_KEY:?export TSFM_API_KEY first}"

BODY='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}'

for URL in \
  "https://api.tsfm.ai/mcp" \
  "https://api.tsfm.ai/mcp/" \
  "https://api.tsfm.ai/v1/mcp" \
  "https://api.tsfm.ai/mcp/v1" \
  "https://tsfm.ai/api/mcp" \
  "https://mcp.tsfm.ai" \
  "https://mcp.tsfm.ai/mcp"
do
  CODE=$(curl -s -o /tmp/probe_body.txt -w '%{http_code}' -X POST "$URL" \
    -H "Authorization: Bearer $TSFM_API_KEY" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    --max-time 20 --data "$BODY")
  printf '%-34s %s  %s\n' "$URL" "$CODE" "$(head -c 160 /tmp/probe_body.txt | tr '\n' ' ')"
done
