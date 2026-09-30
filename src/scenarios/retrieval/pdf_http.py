"""Plain HTTP GET for PDF bytes (browser-like headers)."""

from __future__ import annotations

import requests

_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
_PDF_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "application/pdf,*/*",
}


def probe_pdf_url(url: str, *, timeout: float = 15.0) -> bool:
    cleaned = (url or "").strip()
    if not cleaned:
        return False

    session = requests.Session()
    try:
        response = session.head(
            cleaned,
            headers=_PDF_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        if response.status_code == 200:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "pdf" in content_type:
                return True
            if "octet-stream" in content_type:
                return True
    except (requests.RequestException, OSError):
        pass

    try:
        response = session.get(
            cleaned,
            headers=_PDF_HEADERS,
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
        chunk = next(response.iter_content(4096), b"")
        response.close()
        return len(chunk) >= 4 and chunk[:4] == b"%PDF"
    except (requests.RequestException, OSError, StopIteration):
        return False


def fetch_pdf_bytes(url: str, *, timeout: float = 20) -> bytes:
    session = requests.Session()
    headers = dict(_PDF_HEADERS)
    headers["Accept"] = "application/pdf"
    response = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.content
