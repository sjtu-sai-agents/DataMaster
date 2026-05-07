from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class ProbeDatasetRowsParams(BaseToolParams):
    """Fetch a few raw dataset rows for red-node inspection.

    Prefer the Hugging Face datasets-server rows API so this stays fast. The
    tool falls back to the heavier local materialization path only when the
    rows API cannot return samples.
    """

    name: ClassVar[str] = "probe_dataset_rows"
    source_id: str = Field(description="Canonical Hugging Face dataset id, for example org/dataset")
    output_dir: str = Field(description="Absolute directory where sample artifacts should be written")
    split: str = Field(default="train", description="Dataset split to sample, usually train")
    url: str = Field(default="", description="Dataset URL, optional")
    sample_count: int = Field(default=2, description="Number of raw sample rows to return")


class ProbeDatasetRowsTool(BaseTool):
    name: ClassVar[str] = "probe_dataset_rows"
    params_class: ClassVar[type[BaseToolParams]] = ProbeDatasetRowsParams

    def execute(self, session: Any, args_json: str) -> tuple[str, dict[str, Any]]:
        params = self.parse_params(args_json)
        try:
            from playground.math_posttrain_datatree_v3.core.utils.data import (
                DEFAULT_DATASETS_SERVER_BASE,
                _datasets_server_get,
                _normalize_remote_record,
                prepare_dataset_probe,
            )

            data_access_cfg = _config_section_to_dict(getattr(getattr(session, "config", None), "data_access", None))
            datasets_server_cfg = _config_section_to_dict(data_access_cfg.get("datasets_server"))
            base_url = str(datasets_server_cfg.get("base_url") or DEFAULT_DATASETS_SERVER_BASE).rstrip("/")
            sample_count = min(max(int(params.sample_count), 1), 20)
            output_dir = Path(params.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            rows, backend, resolved = _fetch_fast_rows(
                source_id=params.source_id,
                split=params.split or "train",
                sample_count=sample_count,
                base_url=base_url,
                datasets_server_get=_datasets_server_get,
                normalize_remote_record=_normalize_remote_record,
            )

            if not rows:
                rows, backend, resolved = _fetch_fallback_rows(
                    source_id=params.source_id,
                    url=params.url,
                    split=params.split or "train",
                    sample_count=sample_count,
                    output_dir=output_dir,
                    data_access_cfg=data_access_cfg,
                    prepare_dataset_probe=prepare_dataset_probe,
                )

            sample_rows_path = output_dir / f"{_safe_source_filename(params.source_id)}_sample_rows.jsonl"
            _write_jsonl(sample_rows_path, rows)
            schema_keys = sorted({str(key) for row in rows for key in row.keys()})
            result = {
                "status": "passed" if rows else "failed",
                "source_id": params.source_id,
                "sample_count": len(rows),
                "schema_keys": schema_keys,
                "sample_rows_path": str(sample_rows_path),
                "samples": rows,
                "backend": backend,
                "resolved": resolved,
            }
            if not rows:
                result["reason"] = "no sample rows returned"
        except Exception as exc:
            result = {
                "status": "failed",
                "source_id": getattr(params, "source_id", ""),
                "reason": f"probe error: {exc}",
            }

        return _format_probe_observation(result), {"probe_result": result}


def _fetch_fast_rows(
    *,
    source_id: str,
    split: str,
    sample_count: int,
    base_url: str,
    datasets_server_get,
    normalize_remote_record,
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    direct_candidates = [("default", split or "train"), ("default", "train")]
    tried: set[tuple[str, str]] = set()
    for config_name, split_name in direct_candidates:
        if (config_name, split_name) in tried:
            continue
        tried.add((config_name, split_name))
        rows = _fetch_rows_page(
            source_id=source_id,
            config_name=config_name,
            split_name=split_name,
            sample_count=sample_count,
            base_url=base_url,
            datasets_server_get=datasets_server_get,
            normalize_remote_record=normalize_remote_record,
        )
        if rows:
            return rows, "datasets_server_rows", {"config": config_name, "split": split_name}

    try:
        resp = datasets_server_get(
            "splits",
            params={"dataset": source_id},
            timeout=10,
            dataset_id=source_id,
            base_url=base_url,
        )
        split_entries = resp.json().get("splits")
    except Exception:
        split_entries = None
    if not isinstance(split_entries, list):
        return [], "datasets_server_rows", {}

    for item in split_entries:
        if not isinstance(item, dict):
            continue
        config_name = str(item.get("config") or "default").strip() or "default"
        split_name = str(item.get("split") or "").strip()
        if not split_name or any(token in split_name.lower() for token in ("test", "eval", "dev")):
            continue
        if (config_name, split_name) in tried:
            continue
        rows = _fetch_rows_page(
            source_id=source_id,
            config_name=config_name,
            split_name=split_name,
            sample_count=sample_count,
            base_url=base_url,
            datasets_server_get=datasets_server_get,
            normalize_remote_record=normalize_remote_record,
        )
        if rows:
            return rows, "datasets_server_rows", {"config": config_name, "split": split_name}
    return [], "datasets_server_rows", {}


def _fetch_rows_page(
    *,
    source_id: str,
    config_name: str,
    split_name: str,
    sample_count: int,
    base_url: str,
    datasets_server_get,
    normalize_remote_record,
) -> list[dict[str, Any]]:
    try:
        resp = datasets_server_get(
            "rows",
            params={
                "dataset": source_id,
                "config": config_name or "default",
                "split": split_name,
                "offset": 0,
                "length": sample_count,
            },
            timeout=15,
            dataset_id=source_id,
            base_url=base_url,
        )
        server_rows = resp.json().get("rows")
    except Exception:
        return []
    if not isinstance(server_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in server_rows[:sample_count]:
        if not isinstance(item, dict):
            continue
        row = item.get("row")
        if isinstance(row, dict):
            rows.append(normalize_remote_record(row))
    return rows


def _fetch_fallback_rows(
    *,
    source_id: str,
    url: str,
    split: str,
    sample_count: int,
    output_dir: Path,
    data_access_cfg: dict[str, Any],
    prepare_dataset_probe,
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    entry = {
        "source_id": source_id,
        "name": source_id,
        "url": url or f"https://huggingface.co/datasets/{source_id}",
        "split": split,
        "task_type": "math_reasoning",
    }
    probe_payload = prepare_dataset_probe(
        [entry],
        output_dir,
        max_rows_per_source=sample_count,
        materialize_max_rows=sample_count,
        data_access_config=data_access_cfg,
    )
    source_probe = (probe_payload.get("sources") or [{}])[0]
    rows = source_probe.get("preview_rows") or []
    rows = [row for row in rows if isinstance(row, dict)]
    return rows, "materialize_fallback", {"split": split}


def _config_section_to_dict(section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if isinstance(section, dict):
        return section
    if hasattr(section, "model_dump"):
        value = section.model_dump()
        return value if isinstance(value, dict) else {}
    if hasattr(section, "dict"):
        value = section.dict()
        return value if isinstance(value, dict) else {}
    if hasattr(section, "__dict__"):
        return {key: val for key, val in vars(section).items() if not key.startswith("_")}
    return {}


def _safe_source_filename(source_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in source_id).strip("_") or "dataset"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_probe_observation(result: dict[str, Any]) -> str:
    lines = [
        f"Status: {result.get('status', 'unknown')}",
        f"Source: {result.get('source_id', '')}",
    ]
    if result.get("reason"):
        lines.append(f"Reason: {result.get('reason')}")
    lines.append(f"Backend: {result.get('backend', '')}")
    resolved = result.get("resolved") if isinstance(result.get("resolved"), dict) else {}
    if resolved:
        parts = [f"{key}={value}" for key, value in resolved.items() if value]
        if parts:
            lines.append("Resolved: " + ", ".join(parts))
    lines.append(f"Sample rows: {result.get('sample_count', 0)}")
    schema_keys = result.get("schema_keys") or []
    if schema_keys:
        lines.append(f"Schema keys: {', '.join(str(k) for k in schema_keys)}")
    sample_path = result.get("sample_rows_path")
    if sample_path:
        lines.append(f"Sample rows path: {sample_path}")
    samples = result.get("samples") or []
    if samples:
        lines.append("Samples:")
        for idx, sample in enumerate(samples, start=1):
            lines.append(f"  {idx}. {json.dumps(sample, ensure_ascii=False)[:1000]}")
    return "\n".join(lines)
