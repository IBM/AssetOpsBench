import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Union, Optional

import pendulum
import couchdb3
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import os

# Setup logging — default WARNING so stderr stays quiet when used as MCP server;
# set LOG_LEVEL=INFO (or DEBUG) in the environment to see verbose output.
_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("utilities-mcp-server")

mcp = FastMCP(
    "utilities",
    instructions="General utilities: read JSON files, get current date/time, and write the "
    "scenario's final result to CouchDB for the grader.",
)

# --- CouchDB (final-result store) ---
# init_data seeds an EMPTY `final_result` collection (one placeholder doc, id "result") at the start
# of each scenario run and rebuilds DBs from scratch, so the run is already isolated — no run id.
COUCHDB_URL = os.environ.get("COUCHDB_URL")
COUCHDB_USERNAME = os.environ.get("COUCHDB_USERNAME")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_PASSWORD")
FINAL_RESULT_DBNAME = os.environ.get("FINAL_RESULT_DBNAME", "final_result")
_RESULT_DOC_ID = "result"  # fixed: one result document per scenario run

try:
    _result_db = couchdb3.Database(
        FINAL_RESULT_DBNAME,
        url=COUCHDB_URL,
        user=COUCHDB_USERNAME,
        password=COUCHDB_PASSWORD,
    )
    logger.info("Connected to CouchDB: %s", FINAL_RESULT_DBNAME)
except Exception as e:  # noqa: BLE001
    logger.error("Failed to connect to final_result DB: %s", e)
    _result_db = None


class DateTimeResult(BaseModel):
    currentDateTime: str
    currentDateTimeDescription: str


class TimeEnglishResult(BaseModel):
    english: str
    iso: str


class ErrorResult(BaseModel):
    error: str


class WriteResultResponse(BaseModel):
    ok: bool
    doc_id: str
    message: str


class ReadResultResponse(BaseModel):
    found: bool
    result: Optional[Union[Dict[str, Any], List[Any]]] = None
    written_at: Optional[str] = None
    message: str


# --- Helper Functions ---


def get_temp_filename() -> str:
    tmpdir = tempfile.gettempdir()
    tmppath = Path(tmpdir)
    basepath = Path("cbmdir")
    filename = str(uuid4())

    tmpdir_path = tmppath / basepath
    tmpdir_path.mkdir(parents=True, exist_ok=True)

    filepath = tmpdir_path / (filename + ".json")
    return str(filepath)


# --- JSON Tools ---


@mcp.tool(title="Read JSON File")
def json_reader(file_name: str) -> str:
    """Reads a JSON file, parses its content, and returns the parsed data."""
    try:
        with open(file_name, "r") as fp:
            contents = json.load(fp)
        return json.dumps(contents)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_name}: {e}")
        return json.dumps({"error": str(e)})


# --- Final Result Tool ---


@mcp.tool(title="Write Final Result")
def write_final_result(
    result: Union[Dict[str, Any], List[Any]],
) -> Union[WriteResultResponse, ErrorResult]:
    """Persist this scenario's FINAL answer as a JSON payload to CouchDB so the grader can read it.

    Call this exactly once, at the end, with your final answer. `result` is the JSON the task
    description asks for — a JSON object, or a list of JSON objects. There is no task id: the run is
    scoped to a single scenario (init_data seeds an empty `final_result` collection per run), so it is
    one document. Calling again overwrites it.
    """
    if _result_db is None:
        return ErrorResult(error="CouchDB not connected")
    doc = {
        "_id": _RESULT_DOC_ID,
        "doctype": "result",
        "result": result,
        "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        existing = _result_db.get(_RESULT_DOC_ID)  # seeded placeholder after init_data
        doc["_rev"] = existing["_rev"]
    except Exception:  # noqa: BLE001
        pass  # not seeded yet -> create fresh
    try:
        _result_db.save(doc)
        return WriteResultResponse(
            ok=True, doc_id=_RESULT_DOC_ID, message="final result written"
        )
    except Exception as e:  # noqa: BLE001
        logger.error("write_final_result failed: %s", e)
        return ErrorResult(error=str(e))


def _get_final_result_doc():
    """Fetch the raw result document (or None). Shared by the read tool and the grader helper below."""
    if _result_db is None:
        return None
    try:
        return _result_db.get(_RESULT_DOC_ID)
    except Exception:  # noqa: BLE001
        return None


def _get_final_result_payload():
    """Grader-side helper (importable by the offline grader): return just the persisted payload, or
    None if nothing was written."""
    doc = _get_final_result_doc()
    return doc.get("result") if doc else None


@mcp.tool(title="Read Final Result")
def read_final_result() -> Union[ReadResultResponse, ErrorResult]:
    """Read back the scenario's final result payload (what write_final_result stored). Returns
    found=false if nothing has been written yet (the seeded placeholder has result=null).
    """
    if _result_db is None:
        return ErrorResult(error="CouchDB not connected")
    doc = _get_final_result_doc()
    if doc is None:
        return ReadResultResponse(
            found=False, result=None, message="no result document yet"
        )
    res = doc.get("result")
    return ReadResultResponse(
        found=res is not None,
        result=res,
        written_at=doc.get("written_at"),
        message=(
            "final result read"
            if res is not None
            else "placeholder only (no result written yet)"
        ),
    )


# --- Time Tools ---


@mcp.tool(title="Get Current Date and Time")
def current_date_time() -> DateTimeResult:
    """Provides the current date time as a JSON object."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat().replace("+00:00", "Z")

    date_part = now_iso.split("T")[0]
    time_part = now_iso.split("T")[1].split(".")[0]

    description = f"Today's date is {date_part} and time is {time_part}."

    return DateTimeResult(
        currentDateTime=now_iso, currentDateTimeDescription=description
    )


@mcp.tool(title="Get Current Time in English")
def current_time_english() -> TimeEnglishResult:
    """Returns the current time in English text."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat().replace("+00:00", "Z")

    dt = pendulum.parse(now_iso)
    eng = dt.to_datetime_string()

    return TimeEnglishResult(english=eng, iso=now_iso)


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
