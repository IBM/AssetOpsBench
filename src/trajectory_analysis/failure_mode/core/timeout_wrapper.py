"""Timeout wrapper for LLM calls in trajectory analysis.

This module provides timeout protection for LLM backend calls without
modifying the base LLM backend code.
"""

import threading
from typing import Any, Callable, Optional


class TimeoutError(Exception):
    """Raised when an LLM call times out."""

    pass


def call_with_timeout(
    func: Callable[..., Any], timeout_seconds: int = 30, *args, **kwargs
) -> Any:
    """Call a function with a timeout.

    Args:
        func: Function to call
        timeout_seconds: Timeout in seconds (default: 30)
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        TimeoutError: If function call exceeds timeout
        Exception: Any exception raised by func
    """
    result: list = [None]
    exception: list = [None]

    def wrapper():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(
            f"LLM call timed out after {timeout_seconds} seconds. "
            "The model may be unavailable or experiencing connectivity issues."
        )

    if exception[0] is not None:
        raise exception[0]

    return result[0]


# Made with Bob
