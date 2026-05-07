from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_JSON_STRING_LINEBREAKS = {
    "\n",
    "\r",
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
}


def make_json_serializable(payload: Any) -> Any:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): make_json_serializable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return [make_json_serializable(item) for item in payload]
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except Exception:
            return payload.hex()

    module_name = type(payload).__module__.lower()
    type_name = type(payload).__name__
    if module_name.startswith("pil.") or "image" in type_name.lower():
        summary: dict[str, Any] = {"__type__": type_name}
        for attr in ("mode", "size", "format", "filename"):
            value = getattr(payload, attr, None)
            if value not in (None, ""):
                summary[attr] = make_json_serializable(value)
        return summary

    if hasattr(payload, "tolist"):
        try:
            return make_json_serializable(payload.tolist())
        except Exception:
            pass
    if hasattr(payload, "item"):
        try:
            return make_json_serializable(payload.item())
        except Exception:
            pass
    if hasattr(payload, "isoformat"):
        try:
            return payload.isoformat()
        except Exception:
            pass
    return repr(payload)


def json_dumps_safe(payload: Any, *, ensure_ascii: bool = False, indent: int | None = None) -> str:
    return json.dumps(make_json_serializable(payload), ensure_ascii=ensure_ascii, indent=indent)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json_dumps_safe(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json_dumps_safe(row, ensure_ascii=False) for row in rows)
    file_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
    return file_path


def _read_jsonl_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _escape_raw_newlines_in_json_strings(text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                repaired.append(ch)
                escaped = False
                continue
            if ch == "\\":
                repaired.append(ch)
                escaped = True
                continue
            if ch == '"':
                repaired.append(ch)
                in_string = False
                continue
            if ch in _JSON_STRING_LINEBREAKS:
                repaired.append("\\n")
                continue
            repaired.append(ch)
            continue

        repaired.append(ch)
        if ch == '"':
            in_string = True

    return "".join(repaired)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    text = file_path.read_text(encoding="utf-8")
    try:
        return _read_jsonl_lines(text)
    except json.JSONDecodeError:
        repaired = _escape_raw_newlines_in_json_strings(text)
        return _read_jsonl_lines(repaired)
