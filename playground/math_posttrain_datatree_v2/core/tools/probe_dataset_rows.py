from __future__ import annotations

import os
import json
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar

import requests
from pydantic import Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


DEFAULT_DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"


class ProbeDatasetRowsParams(BaseToolParams):
    """Fetch a few raw dataset rows for red-node inspection.

    Prefer the Hugging Face datasets-server rows API so this stays fast. If
    that API is unavailable, sample rows with local streaming. This tool only
    writes the requested sample rows, not shared dataset cache files.
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
            )

            if not rows:
                rows, backend, resolved = _fetch_fallback_rows(
                    source_id=params.source_id,
                    split=params.split or "train",
                    sample_count=sample_count,
                    data_access_cfg=data_access_cfg,
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
                result["reason"] = resolved.get("reason") or "no sample rows returned"
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
        )
        if rows:
            return rows, "datasets_server_rows", {"config": config_name, "split": split_name}

    try:
        split_payload = _quick_datasets_server_json(
            base_url,
            "splits",
            params={"dataset": source_id},
            timeout=4,
        )
        split_entries = split_payload.get("splits")
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
) -> list[dict[str, Any]]:
    try:
        payload = _quick_datasets_server_json(
            base_url,
            "rows",
            params={
                "dataset": source_id,
                "config": config_name or "default",
                "split": split_name,
                "offset": 0,
                "length": sample_count,
            },
            timeout=4,
        )
        server_rows = payload.get("rows")
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
            rows.append(_normalize_record(row))
    return rows


def _quick_datasets_server_json(
    base_url: str,
    endpoint: str,
    *,
    params: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    response = requests.get(
        f"{base_url.rstrip('/')}/{endpoint}",
        params=params,
        timeout=(3, timeout),
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _fetch_fallback_rows(
    *,
    source_id: str,
    split: str,
    sample_count: int,
    data_access_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    rows, resolved = _fetch_local_streaming_rows(
        source_id=source_id,
        split=split,
        sample_count=sample_count,
        data_access_cfg=data_access_cfg,
    )
    if rows:
        return rows, "local_streaming_rows", resolved
    return [], "local_streaming_rows", resolved


def _fetch_local_streaming_rows(
    *,
    source_id: str,
    split: str,
    sample_count: int,
    data_access_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    try:
        from datasets import load_dataset
        import signal

        hf_endpoint = str(data_access_cfg.get("hf_endpoint") or os.getenv("HF_ENDPOINT") or "").strip()
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
            os.environ["HF_HUB_URL"] = hf_endpoint
        hf_cache = str(data_access_cfg.get("hf_cache") or os.getenv("HF_DATASETS_CACHE") or "").strip()
        if hf_cache:
            os.environ["HF_DATASETS_CACHE"] = hf_cache

        split_name = split or "train"
        errors: list[str] = []

        # Set timeout for load_dataset operations (30 seconds per attempt)
        def timeout_handler(signum, frame):
            raise TimeoutError("Dataset loading timed out after 30 seconds")

        for load_args in ((source_id,), (source_id, "default")):
            try:
                # Set alarm for 30 seconds
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)

                dataset = load_dataset(*load_args, split=split_name, streaming=True)
                rows = [
                    _normalize_record(row)
                    for row in islice(dataset, sample_count)
                    if isinstance(row, dict)
                ]

                # Cancel alarm if successful
                signal.alarm(0)

                if rows:
                    return rows, {
                        "split": split_name,
                        "config": "default" if len(load_args) > 1 else "",
                    }
            except TimeoutError as exc:
                signal.alarm(0)
                errors.append(f"timeout after 30s: {str(exc)}")
            except Exception as exc:
                signal.alarm(0)
                message = str(exc)
                errors.append(message)
                if "Available splits" in message or "Unknown split" in message or "Bad split" in message:
                    return [], {
                        "split": split_name,
                        "reason": _clip_text(message, 240),
                        "terminal": "true",
                    }
        return [], {"split": split_name, "reason": _clip_text("; ".join(errors), 240)}
    except Exception as exc:
        return [], {"split": split or "train", "reason": _clip_text(str(exc), 240)}


def _pick_first(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    problem = _pick_first(
        payload,
        ("problem", "question", "Question", "prompt", "Prompt", "input", "Input", "query", "instruction", "Instruction"),
    )
    solution = _pick_first(
        payload,
        (
            "solution",
            "Solution",
            "response",
            "Response",
            "output",
            "Output",
            "rationale",
            "cot",
            "reasoning",
            "analysis",
            "explanation",
            "generated_solution",
            "Generated Solution",
        ),
    )
    final_answer = _pick_first(
        payload,
        ("final_answer", "Final Answer", "answer", "Answer", "target", "Target", "label", "Label", "expected_answer", "Expected Answer"),
    )
    if problem:
        payload["problem"] = problem
    if solution:
        payload["solution"] = solution
    if final_answer:
        payload["final_answer"] = final_answer
    return payload


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


def _clip_text(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


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
