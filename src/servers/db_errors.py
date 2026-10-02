"""Agent-facing errors for CouchDB failures.

A missing or unreachable database gets one message, so an agent can tell it apart
from a wrong key, and no message carries a database name, host or URL.
"""

DATA_UNAVAILABLE = (
    "the data source does not exist or is unreachable in this "
    "environment; the data is unavailable, do not retry with other arguments"
)

_HTTP_CLIENT_MODULES = ("requests", "urllib3", "httpx", "httpcore")


def db_failure_text(exc: Exception) -> str:
    """Describe a failed database call without the URL, host or database name
    that HTTP client exceptions embed in their text."""
    if (type(exc).__module__ or "").startswith(_HTTP_CLIENT_MODULES):
        return f"database request failed ({type(exc).__name__})"
    return str(exc)
