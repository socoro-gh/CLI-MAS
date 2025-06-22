"""Utility tools available to MAS agents.
Each tool returns plain text and is referenced via DEFAULT_TOOLS mapping.
"""
from __future__ import annotations

import math
import os
import re
from html import unescape
from pathlib import Path
from typing import Dict

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def _safe_eval(expr: str) -> str:
    """Evaluate a simple math expression safely."""
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    allowed_names.update({"pi": math.pi, "e": math.e, "tau": math.tau, "round": round})
    try:
        value = eval(expr, {"__builtins__": {}}, allowed_names)
        return str(value)
    except Exception as exc:
        return f"[calc error: {exc}]"


def _strip_html(text: str) -> str:
    text = unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tool_calc(expression: str) -> str:
    return _safe_eval(expression)


def _tool_search(query: str) -> str:
    """Very lightweight DuckDuckGo scrape – returns title & URL of first result."""
    if requests is None:
        return "[search error: requests lib missing]"
    try:
        resp = requests.get(
            "https://duckduckgo.com/html/", params={"q": query}, timeout=3, headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        # crude parsing
        m = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text, re.S)
        if not m:
            return "[search: no results]"
        url, title_raw = m.group(1), m.group(2)
        title = _strip_html(title_raw)
        return f"{title} – {url}"
    except Exception as exc:  # pragma: no cover
        return f"[search error: {exc}]"

# --- File helpers ---------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent
_NOTES_DIR = _BASE_DIR / "notes"
_NOTES_DIR.mkdir(exist_ok=True)

_SAFE_PATH_RE = re.compile(r"^[\w\-./]+$")


def _sanitize_path(path_str: str) -> Path | None:
    if not _SAFE_PATH_RE.fullmatch(path_str) or ".." in path_str or os.path.isabs(path_str):
        return None
    return _BASE_DIR / path_str


def _tool_read(path_str: str) -> str:
    path = _sanitize_path(path_str)
    if path is None or not path.exists() or not path.is_file():
        return "[read error: invalid path]"
    try:
        data = path.read_text(encoding="utf-8")
        return data[:2048]  # cap output
    except Exception as exc:
        return f"[read error: {exc}]"


def _tool_write(arg: str) -> str:
    try:
        dest_raw, _, content = arg.partition(":" )
        if not dest_raw or not content:
            return "[write usage: filename.txt:content]"
        dest_path = _sanitize_path(f"notes/{dest_raw.strip()}")
        if dest_path is None:
            return "[write error: bad filename]"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("a", encoding="utf-8") as fp:
            fp.write(content + "\n")
        return f"[wrote {len(content)} chars to {dest_path.relative_to(_BASE_DIR)}]"
    except Exception as exc:
        return f"[write error: {exc}]"

# --------------------------------------------------------------------------------------
DEFAULT_TOOLS: Dict[str, Dict[str, object]] = {
    "calc": {"description": "Evaluate a math expression", "func": _tool_calc},
    "search": {"description": "DuckDuckGo search – returns first result", "func": _tool_search},
    "read": {"description": "Read a text file within project (max 2KB)", "func": _tool_read},
    "write": {"description": "Append a line to notes/<file>: usage filename.txt:content", "func": _tool_write},
} 