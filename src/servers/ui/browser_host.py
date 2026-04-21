import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from .main import ASK_UI_PATH, ask_aob, list_evaluation_questions, list_models, run_evaluation

logger = logging.getLogger("ui-browser-host")

TOOL_HANDLERS = {
    "ask_aob": ask_aob,
    "list_evaluation_questions": list_evaluation_questions,
    "list_models": list_models,
    "run_evaluation": run_evaluation,
}

HOST_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AssetOpsBench Ask Host</title>
    <style>
      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        background: #f7f8fa;
      }

      iframe {
        display: block;
        width: 100%;
        height: 100vh;
        border: 0;
      }
    </style>
  </head>
  <body>
    <iframe id="ask-frame" src="/ask.html" title="AssetOpsBench Ask"></iframe>

    <script>
      const frame = document.getElementById("ask-frame");

      function sendResponse(messageId, payload) {
        frame.contentWindow.postMessage(
          {
            type: "ui-message-response",
            messageId,
            payload,
          },
          "*",
        );
      }

      window.addEventListener("message", async (event) => {
        const data = event.data || {};

        if (data.type === "ui-size-change") {
          return;
        }

        if (data.type !== "tool" || !data.messageId) {
          return;
        }

        try {
          const response = await fetch("/tool", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(data.payload || {}),
          });
          const body = await response.json();

          if (!response.ok) {
            throw new Error(body.error || "Tool call failed.");
          }

          sendResponse(data.messageId, { response: body });
        } catch (error) {
          sendResponse(data.messageId, { error: error.message });
        }
      });
    </script>
  </body>
</html>
"""


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value


class BrowserHostHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_text(HOST_HTML, content_type="text/html; charset=utf-8")
            return

        if path == "/ask.html":
            self._send_text(
                ASK_UI_PATH.read_text(encoding="utf-8"),
                content_type="text/html; charset=utf-8",
            )
            return

        self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/tool":
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            request = self._read_json()
            tool_name = request.get("toolName")
            params = request.get("params") or {}

            handler = TOOL_HANDLERS.get(tool_name)
            if not handler:
                self._send_json(
                    {"error": f"Unsupported tool: {tool_name}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            result = handler(**params)
            self._send_json(_to_jsonable(result))
        except TypeError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:
            logger.exception("Browser UI tool call failed")
            self._send_json(
                {"error": str(error)},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        return json.loads(body or "{}")

    def _send_text(
        self,
        body: str,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self,
        body: dict[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), BrowserHostHandler)
    url_host = "localhost" if host in {"", "0.0.0.0"} else host
    print(f"AssetOpsBench Ask UI: http://{url_host}:{server.server_port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the AssetOpsBench Ask UI in a browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
