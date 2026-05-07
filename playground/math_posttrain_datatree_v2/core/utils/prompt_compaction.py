from __future__ import annotations

from typing import Any


def clip_text(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def summarize_dataset_manifest(manifest: dict[str, Any], *, max_sources: int = 8) -> str:
    datasets = manifest.get("datasets") if isinstance(manifest, dict) else None
    if not isinstance(datasets, list) or not datasets:
        return "No dataset sources are available."

    lines: list[str] = []
    for index, entry in enumerate(datasets[:max_sources], start=1):
        if not isinstance(entry, dict):
            continue
        source_id = entry.get("source_id") or entry.get("name") or f"source_{index}"
        task_type = entry.get("task_type") or "unknown"
        split = entry.get("split") or "default"
        local_path = entry.get("local_path") or entry.get("full_local_path") or ""
        tags = entry.get("coverage_tags")
        tag_text = ", ".join(str(tag) for tag in tags[:5]) if isinstance(tags, list) else ""
        quality = entry.get("quality_signals")
        quality = quality if isinstance(quality, dict) else {}
        note = quality.get("relevance") or quality.get("sample_evidence") or ""
        lines.append(
            f"{index}. {source_id} | task={task_type} | split={split} | tags={tag_text}\n"
            f"   local_path={clip_text(local_path, 220)}\n"
            f"   note={clip_text(note, 220)}"
        )

    if len(datasets) > max_sources:
        lines.append(f"... {len(datasets) - max_sources} more sources omitted; inspect the manifest file if needed.")
    return "\n".join(lines)


def summarize_probe_payload(probe_payload: dict[str, Any], *, max_sources: int = 8) -> str:
    datasets = probe_payload.get("datasets") if isinstance(probe_payload, dict) else None
    if not isinstance(datasets, list) or not datasets:
        return "No probe summary is available. Inspect the local dataset files directly."

    lines: list[str] = []
    for index, entry in enumerate(datasets[:max_sources], start=1):
        if not isinstance(entry, dict):
            continue
        source_id = entry.get("source_id") or f"source_{index}"
        schema = entry.get("schema_keys")
        schema_text = ", ".join(str(key) for key in schema[:8]) if isinstance(schema, list) else ""
        local_path = entry.get("local_path") or entry.get("full_local_path") or ""
        sample_rows_path = entry.get("sample_rows_path") or entry.get("probe_sample_rows_path") or ""
        preview = _preview_probe_row(entry.get("preview_rows"))
        lines.append(
            f"{index}. {source_id} | sampled_rows={entry.get('sample_row_count')} | schema={schema_text}\n"
            f"   data={clip_text(local_path, 220)}\n"
            f"   probe_rows={clip_text(sample_rows_path, 220)}\n"
            f"   preview={preview}"
        )

    if len(datasets) > max_sources:
        lines.append(f"... {len(datasets) - max_sources} more probe entries omitted; inspect probe_summary.json if needed.")
    return "\n".join(lines)


def _preview_probe_row(preview_rows: Any) -> str:
    if not isinstance(preview_rows, list) or not preview_rows:
        return ""
    first = preview_rows[0]
    if not isinstance(first, dict):
        return clip_text(first, 220)

    parts: list[str] = []
    for key in ("instruction", "prompt", "problem", "input", "output", "solution"):
        value = first.get(key)
        if value:
            parts.append(f"{key}={clip_text(value, 90)}")
        if len(parts) >= 3:
            break
    return "; ".join(parts)
