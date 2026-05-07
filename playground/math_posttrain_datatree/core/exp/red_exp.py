from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance

from ..utils.data import infer_coverage_tags, prepare_dataset_probe
from ..utils.memory import build_prompt_memory
from ..utils.io import read_json, write_json
from ..utils.handoff import (
    get_global_pool_manifest_path,
    load_json_payload,
    merge_global_pool_manifest,
    summarize_black_handoff,
    summarize_global_pool_manifest,
)
from ..utils.types import DatasetEntry, DatasetManifest
from . import NodeExp

logger = logging.getLogger(__name__)


def _config_section_to_dict(section: Any) -> dict[str, Any]:
    def _convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if hasattr(value, "model_dump"):
            return {key: _convert(val) for key, val in value.model_dump().items()}
        if hasattr(value, "dict"):
            return {key: _convert(val) for key, val in value.dict().items()}
        if hasattr(value, "__dict__"):
            return {
                key: _convert(val)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    if section is None:
        return {}
    if isinstance(section, dict):
        return _convert(section)
    converted = _convert(section)
    return converted if isinstance(converted, dict) else {}


DISALLOWED_DATASET_IDS = {
    "open-web-math/open-web-math",
}

DISALLOWED_CORPUS_HINTS = (
    "common crawl",
    "web text",
    "web-scale",
    "6.3m documents",
    "14.7b tokens",
    "6.3 million documents",
    "14.7 billion tokens",
    "pretraining",
    "pre-training",
)

KNOWN_DATASET_ALIASES: dict[str, dict[str, Any]] = {
    "math dataset (hendrycks)": {
        "source_id": "nlile/hendrycks-MATH-benchmark",
        "name": "nlile/hendrycks-MATH-benchmark",
        "url": "https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark",
        "coverage_tags": ["competition_math"],
    },
    "hendrycks math": {
        "source_id": "nlile/hendrycks-MATH-benchmark",
        "name": "nlile/hendrycks-MATH-benchmark",
        "url": "https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark",
        "coverage_tags": ["competition_math"],
    },
    "aime 1983-2024": {
        "source_id": "gneubig/aime-1983-2024",
        "name": "gneubig/aime-1983-2024",
        "url": "https://huggingface.co/datasets/gneubig/aime-1983-2024",
        "coverage_tags": ["aime", "competition_math"],
    },
    "gsm8k": {
        "source_id": "openai/gsm8k",
        "name": "openai/gsm8k",
        "url": "https://huggingface.co/datasets/openai/gsm8k",
        "coverage_tags": ["gsm8k"],
    },
    "openmathreasoning": {
        "source_id": "nvidia/OpenMathReasoning",
        "name": "nvidia/OpenMathReasoning",
        "url": "https://huggingface.co/datasets/nvidia/OpenMathReasoning",
        "coverage_tags": ["competition_math"],
    },
    "numinamath-1.5": {
        "source_id": "AI-MO/NuminaMath-1.5",
        "name": "AI-MO/NuminaMath-1.5",
        "url": "https://huggingface.co/datasets/AI-MO/NuminaMath-1.5",
        "coverage_tags": ["competition_math"],
    },
}
HF_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def get_dataset_manifest_path(task_workspace: Path, node_id: str | None = None) -> Path:
    manifests_dir = task_workspace / "artifacts" / "manifests"
    if node_id is None:
        return manifests_dir / "dataset_manifest.json"
    return manifests_dir / f"dataset_manifest_{node_id}.json"


class RedExp(NodeExp):
    def __init__(
        self,
        agent,
        session,
        workspace: Path,
        task_workspace: Path,
        config,
        node,
        manifest_path: Path,
        search_goal: str,
        global_pool_manifest_path: Path | None = None,
        input_black_handoff_path: Path | None = None,
        exp_index: int = 0,
    ):
        super().__init__(agent, session, workspace, task_workspace, config, node, exp_index)
        self.manifest_path = manifest_path
        self.search_goal = search_goal
        self.global_pool_manifest_path = global_pool_manifest_path or get_global_pool_manifest_path(task_workspace)
        self.input_black_handoff_path = input_black_handoff_path

    def _lookup_alias(self, value: str) -> dict[str, Any]:
        key = re.sub(r"\s+", " ", (value or "").strip().lower())
        return KNOWN_DATASET_ALIASES.get(key, {})

    def _canonicalize_dataset_ref(
        self,
        source_id: str,
        name: str,
        url: str,
    ) -> tuple[str, str, str, list[str]]:
        alias_info: dict[str, Any] = {}
        for candidate in (source_id, name):
            if not candidate:
                continue
            alias_info = self._lookup_alias(candidate)
            if alias_info:
                break

        canonical_source_id = source_id.strip()
        canonical_name = name.strip()
        canonical_url = url.strip()
        inferred_tags = list(alias_info.get("coverage_tags", []))

        if not HF_ID_RE.match(canonical_source_id):
            if alias_info.get("source_id"):
                canonical_source_id = str(alias_info["source_id"])
            elif HF_ID_RE.match(canonical_name):
                canonical_source_id = canonical_name

        if not canonical_name:
            canonical_name = canonical_source_id
        elif alias_info.get("name"):
            canonical_name = str(alias_info["name"])
        elif HF_ID_RE.match(canonical_name):
            canonical_source_id = canonical_name

        if canonical_url in {"", "agent_search"}:
            if alias_info.get("url"):
                canonical_url = str(alias_info["url"])
            elif HF_ID_RE.match(canonical_source_id):
                canonical_url = f"https://huggingface.co/datasets/{canonical_source_id}"

        if not canonical_name:
            canonical_name = canonical_source_id
        return canonical_source_id, canonical_name, canonical_url or "agent_search", inferred_tags

    def _coerce_dataset_entry(self, payload: dict[str, Any]) -> DatasetEntry | None:
        source_id = str(
            payload.get("source_id")
            or payload.get("dataset_id")
            or payload.get("id")
            or payload.get("name")
            or ""
        ).strip()
        if not source_id:
            return None
        full_local_path = str(
            payload.get("full_local_path")
            or payload.get("local_path")
            or payload.get("path")
            or ""
        ).strip()
        probe_sample_rows_path = str(payload.get("probe_sample_rows_path") or "").strip()
        url = str(
            payload.get("url")
            or payload.get("huggingface_url")
            or payload.get("source_url")
            or ""
        ).strip()
        source_id, canonical_name, url, inferred_tags = self._canonicalize_dataset_ref(
            source_id=source_id,
            name=str(payload.get("name") or source_id),
            url=url,
        )
        if not source_id:
            return None
        description = str(payload.get("description") or payload.get("summary") or "").strip()
        size_text = str(payload.get("size") or payload.get("size_text") or "").strip()
        combined_text = " ".join(
            part
            for part in (
                source_id,
                canonical_name,
                url,
                description,
                size_text,
            )
            if part
        ).lower()
        if source_id in DISALLOWED_DATASET_IDS:
            logger.info("Filtered dataset %s from red manifest: disallowed dataset id", source_id)
            return None
        if any(hint in combined_text for hint in DISALLOWED_CORPUS_HINTS):
            logger.info("Filtered dataset %s from red manifest: corpus-style dataset is not suitable for post-train", source_id)
            return None
        coverage_tags = payload.get("coverage_tags")
        if not isinstance(coverage_tags, list):
            categories = payload.get("categories")
            if isinstance(categories, list):
                coverage_tags = [str(item) for item in categories]
            else:
                coverage_tags = []
        for tag in inferred_tags:
            if tag not in coverage_tags:
                coverage_tags.append(tag)
        quality = payload.get("quality_signals")
        if not isinstance(quality, dict):
            quality = {}
            if payload.get("quality"):
                quality["quality"] = str(payload["quality"])
            if payload.get("relevance"):
                quality["relevance"] = str(payload["relevance"])
            if payload.get("priority") is not None:
                quality["priority"] = payload["priority"]

        return DatasetEntry(
            source_id=source_id,
            name=canonical_name,
            license=str(payload.get("license") or "unknown"),
            url=url,
            local_path=full_local_path,
            full_local_path=full_local_path,
            probe_sample_rows_path=probe_sample_rows_path,
            split=str(payload.get("split") or ""),
            config=str(
                payload.get("config")
                or payload.get("config_name")
                or payload.get("subset")
                or payload.get("dataset_config")
                or ""
            ),
            task_type=str(payload.get("task_type") or "math_reasoning"),
            answer_style=str(payload.get("answer_style") or "mixed"),
            difficulty=str(payload.get("difficulty") or "unknown"),
            language=str(payload.get("language") or "en"),
            quality_signals=quality,
            coverage_tags=[str(item) for item in coverage_tags],
        )

    def _load_agent_manifest_entries(self) -> list[DatasetEntry]:
        payload = read_json(self.manifest_path, default={}) or {}
        raw_datasets = payload.get("datasets")
        if not isinstance(raw_datasets, list):
            return []
        entries: list[DatasetEntry] = []
        for item in raw_datasets:
            if not isinstance(item, dict):
                continue
            entry = self._coerce_dataset_entry(item)
            if entry is not None:
                entries.append(entry)
        return entries

    def _scan_local_sources(self) -> list[DatasetEntry]:
        roots = [
            self.task_workspace / "data_sources",
            self.workspace / "data_sources",
        ]
        entries: list[DatasetEntry] = []
        seen: set[str] = set()
        for root in roots:
            if not root.exists():
                continue
            for file_path in sorted(root.rglob("*")):
                if not file_path.is_file() or file_path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
                    continue
                source_id = file_path.stem
                if source_id in seen:
                    continue
                seen.add(source_id)
                try:
                    rows = []
                    if file_path.suffix.lower() == ".jsonl":
                        with file_path.open("r", encoding="utf-8") as handle:
                            for idx, line in enumerate(handle):
                                if idx >= 20:
                                    break
                                rows.append(json.loads(line))
                    elif file_path.suffix.lower() == ".json":
                        payload = json.loads(file_path.read_text(encoding="utf-8"))
                        if isinstance(payload, list):
                            rows = payload[:20]
                        elif isinstance(payload, dict):
                            rows = payload.get("data", [])[:20]
                    coverage_tags = infer_coverage_tags(file_path, rows)
                except Exception:
                    coverage_tags = ["math_reasoning"]
                entries.append(
                    DatasetEntry(
                        source_id=source_id,
                        name=file_path.stem,
                        license="unknown",
                        url="local_scan",
                        local_path=str(file_path.resolve()),
                        full_local_path=str(file_path.resolve()),
                        split="",
                        config="",
                        answer_style="mixed",
                        difficulty="unknown",
                        language="en",
                        quality_signals={"scanner": "local_fallback"},
                        coverage_tags=coverage_tags,
                    )
                )
        return entries

    def run(self, task_description: str) -> dict:
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"math_red_{node_id[:8]}", exp_index=self.exp_index)
        agent_text = ""
        global_pool_manifest = load_json_payload(self.global_pool_manifest_path)
        global_pool_summary = summarize_global_pool_manifest(global_pool_manifest)
        upstream_black_handoff = load_json_payload(self.input_black_handoff_path)
        upstream_black_handoff_summary = summarize_black_handoff(upstream_black_handoff)
        if self.agent is not None:
            orig_fmt = self.agent._prompt_format_kwargs.copy()
            self.agent._prompt_format_kwargs.update(
                {
                    "task_description": task_description,
                    "workspace": str(self.workspace),
                    "task_workspace": str(self.task_workspace),
                    "manifest_path": str(self.manifest_path),
                    "search_goal": self.search_goal,
                    "memory_summary": build_prompt_memory(self.task_workspace, self.node),
                    "global_pool_manifest_path": str(self.global_pool_manifest_path),
                    "global_pool_manifest_summary_json": json.dumps(global_pool_summary, ensure_ascii=False, indent=2),
                    "input_black_handoff_path": str(self.input_black_handoff_path or ""),
                    "input_black_handoff_json": json.dumps(upstream_black_handoff_summary, ensure_ascii=False, indent=2),
                }
            )
            try:
                task = TaskInstance(
                    task_id=f"{node_id}_red",
                    task_type="red",
                    description=task_description,
                    input_data={},
                )
                traj = self.agent.run(task)
                from evomaster.core.exp import extract_agent_response

                agent_text = extract_agent_response(traj)
            except Exception as exc:
                logger.warning("Red agent execution failed, using fallback scan: %s", exc)
            finally:
                self.agent._prompt_format_kwargs = orig_fmt

        entries = self._load_agent_manifest_entries()
        if not entries:
            entries = self._scan_local_sources()

        global_pool_manifest = merge_global_pool_manifest(
            global_pool_manifest,
            [item.to_dict() for item in entries],
            node_id=node_id,
            search_goal=self.search_goal,
            agent_summary=agent_text,
        )
        data_access_cfg = _config_section_to_dict(getattr(self.config, "data_access", None))
        pool_probe_dir = self.task_workspace / "artifacts" / "data_pool" / "global_pool_probe"
        pool_probe_payload = prepare_dataset_probe(
            global_pool_manifest.get("datasets") or [],
            pool_probe_dir,
            materialize_max_rows=None,
            data_access_config=data_access_cfg,
        )
        prepared_by_source = {
            str(item.get("source_id") or item.get("name") or ""): item
            for item in (global_pool_manifest.get("datasets") or [])
            if isinstance(item, dict)
        }
        for entry in entries:
            prepared = prepared_by_source.get(str(entry.source_id))
            if not prepared:
                continue
            if prepared.get("full_local_path") or prepared.get("local_path"):
                entry.full_local_path = str(prepared.get("full_local_path") or prepared.get("local_path") or "")
                entry.local_path = entry.full_local_path
            if prepared.get("probe_sample_rows_path"):
                entry.probe_sample_rows_path = str(prepared.get("probe_sample_rows_path") or "")

        manifest = DatasetManifest(
            manifest_id=f"dataset_manifest_{node_id}",
            created_from_node=node_id,
            search_goal=self.search_goal,
            datasets=entries,
            coverage_tags=sorted({tag for item in entries for tag in item.coverage_tags}),
            source_summary={
                "source_count": len(entries),
                "agent_summary": agent_text[:500],
            },
        )
        write_json(self.manifest_path, manifest.to_dict())
        write_json(self.global_pool_manifest_path, global_pool_manifest)
        pool_summary = summarize_global_pool_manifest(global_pool_manifest)
        new_source_ids = (global_pool_manifest.get("source_summary") or {}).get("latest_added_source_ids") or []
        return {
            "plan": self.search_goal,
            "code": "",
            "raw_response": agent_text,
            "exec": {
                "stdout": (
                    f"dataset_manifest_written={len(entries) > 0} path={self.manifest_path} "
                    f"global_pool={self.global_pool_manifest_path} total_sources={pool_summary.get('source_count', 0)} prepared_rows={int(pool_probe_payload.get('total_sample_rows') or 0)}"
                ),
                "exit_code": 0 if entries else 1,
            },
            "metric": None,
            "metric_detail": {
                "manifest_ok": bool(entries),
                "manifest_path": str(self.manifest_path),
                "global_pool_manifest_path": str(self.global_pool_manifest_path),
                "global_pool_source_count": int(pool_summary.get("source_count") or 0),
                "new_source_count": len(new_source_ids),
                "prepared_probe_dir": str(pool_probe_dir),
                "prepared_probe_rows": int(pool_probe_payload.get("total_sample_rows") or 0),
                "has_submission": False,
            },
        }
