import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pendulum
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import os

# Setup logging — default WARNING so stderr stays quiet when used as MCP server;
# set LOG_LEVEL=INFO (or DEBUG) in the environment to see verbose output.
_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("utilities-mcp-server")

mcp = FastMCP("Utilities")


class DateTimeResult(BaseModel):
    currentDateTime: str
    currentDateTimeDescription: str


class TimeEnglishResult(BaseModel):
    english: str
    iso: str


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


@mcp.tool()
def json_reader(file_name: str) -> str:
    """Reads a JSON file, parses its content, and returns the parsed data."""
    try:
        with open(file_name, "r") as fp:
            contents = json.load(fp)
        return json.dumps(contents)
    except Exception as e:
        logger.error(f"Error reading JSON file {file_name}: {e}")
        return json.dumps({"error": str(e)})


# --- Time Tools ---


@mcp.tool()
def current_date_time() -> DateTimeResult:
    """Provides the current date time as a JSON object."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat().replace("+00:00", "Z")

    date_part = now_iso.split("T")[0]
    time_part = now_iso.split("T")[1].split(".")[0]

    description = f"Today's date is {date_part} and time is {time_part}."

    return DateTimeResult(currentDateTime=now_iso, currentDateTimeDescription=description)


@mcp.tool()
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
