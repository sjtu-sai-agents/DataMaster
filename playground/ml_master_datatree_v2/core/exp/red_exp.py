"""Red Experiment v2 - Data Scout: search, download, characterize → write manifest.

Red 节点不负责训练，只负责：
1. 搜索 & 下载外部数据集
2. 用 execute_bash 探查数据格式（image size、schema、label 分布）
3. 将探查结果写入节点专属 manifest 文件
4. 验证 loading_snippet 可以实际运行

成功判定：manifest 存在 且 至少包含一个有效 dataset 条目。
"""

import json
import logging
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance

from . import NodeExp
from playground.ml_master_datatree.core.utils.runtime import (
    extract_agent_response,
    run_code_via_bash,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "data_manifest.json"
MANIFEST_DIRNAME = "manifests"


def get_manifest_path(task_workspace: Path, node_id: str | None = None) -> Path:
    """Return the manifest path for a task or a specific red node."""
    if node_id is None:
        return task_workspace / MANIFEST_FILENAME
    return task_workspace / MANIFEST_DIRNAME / f"manifest_{node_id}.json"


def load_manifest(task_workspace: Path, manifest_path: Path | None = None) -> dict | None:
    """Read and return a manifest file, or None if missing/invalid."""
    manifest_path = manifest_path or get_manifest_path(task_workspace)
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse manifest at %s: %s", manifest_path, exc)
        return None


def manifest_has_datasets(manifest: dict | None) -> bool:
    """Return True when a manifest contains at least one dataset entry."""
    if not manifest:
        return False
    return bool(manifest.get("external_datasets", []))


def _infer_text_and_label_columns(columns: list[str]) -> tuple[str | None, str | None]:
    lower_to_raw = {col.lower(): col for col in columns}

    text_candidates = [
        "comment", "text", "tweet", "content", "sentence", "post", "user_input",
    ]
    label_candidates = [
        "label", "labels", "target", "class", "is_hate", "ishate",
        "toxicity", "toxic", "insult", "offensive", "sentiment",
    ]

    text_col = next((lower_to_raw[name] for name in text_candidates if name in lower_to_raw), None)
    label_col = next((lower_to_raw[name] for name in label_candidates if name in lower_to_raw), None)

    if text_col is None and columns:
        text_col = columns[0]
    if label_col is None and len(columns) > 1:
        label_col = columns[-1]
    if label_col == text_col:
        label_col = columns[1] if len(columns) > 1 else None
    return text_col, label_col


class RedExp(NodeExp):
    """Red 实验 v2：数据侦察 agent，输出 data_manifest.json。

    不跑训练，成功判定由 playground 检查 manifest 文件。
    """

    def __init__(
        self,
        agent,
        metric_agent,
        session,
        workspace: Path,
        task_workspace: Path,
        manifest_path: Path | None,
        input_manifest_path: Path | None,
        demand_spec: dict[str, object] | None,
        exp_id: str | None,
        data_preview: str,
        node,
        exp_index: int = 0,
    ):
        super().__init__(agent, metric_agent, session, workspace, exp_id, data_preview, node, exp_index)
        # task_workspace: 任务级共享目录，data_links 等共享资源放在这里
        self.task_workspace = task_workspace
        self.manifest_path = manifest_path or get_manifest_path(task_workspace, node.id)
        self.input_manifest_path = input_manifest_path
        self.demand_spec = demand_spec or {}

    def _build_loading_snippet(
        self,
        file_path: Path,
        file_format: str,
        text_col: str | None,
        label_col: str | None,
    ) -> str:
        if file_format == "csv":
            return (
                "import pandas as pd\n"
                f"df = pd.read_csv(r'{file_path}')\n"
                f"text_col = {text_col!r}\n"
                f"label_col = {label_col!r}\n"
                "texts = df[text_col].fillna('').astype(str)\n"
                "labels = df[label_col] if label_col in df.columns else None\n"
                "print(df.shape)"
            )
        if file_format == "tsv":
            return (
                "import pandas as pd\n"
                f"df = pd.read_csv(r'{file_path}', sep='\\t')\n"
                f"text_col = {text_col!r}\n"
                f"label_col = {label_col!r}\n"
                "texts = df[text_col].fillna('').astype(str)\n"
                "labels = df[label_col] if label_col in df.columns else None\n"
                "print(df.shape)"
            )
        if file_format == "parquet":
            return (
                "import pandas as pd\n"
                f"df = pd.read_parquet(r'{file_path}')\n"
                f"text_col = {text_col!r}\n"
                f"label_col = {label_col!r}\n"
                "texts = df[text_col].fillna('').astype(str)\n"
                "labels = df[label_col] if label_col in df.columns else None\n"
                "print(df.shape)"
            )
        return (
            "import pandas as pd\n"
            f"df = pd.read_json(r'{file_path}', lines={file_format == 'jsonl'})\n"
            f"text_col = {text_col!r}\n"
            f"label_col = {label_col!r}\n"
            "texts = df[text_col].fillna('').astype(str)\n"
            "labels = df[label_col] if label_col in df.columns else None\n"
            "print(df.shape)"
        )

    def _infer_dataset_entry(self, file_path: Path) -> dict[str, Any] | None:
        suffix = file_path.suffix.lower()
        file_format = {
            ".csv": "csv",
            ".tsv": "tsv",
            ".parquet": "parquet",
            ".json": "json",
            ".jsonl": "jsonl",
        }.get(suffix)
        if file_format is None:
            return None

        try:
            import pandas as pd

            if file_format == "csv":
                try:
                    df = pd.read_csv(file_path)
                except Exception:
                    df = pd.read_csv(file_path, on_bad_lines="skip", engine="python")
            elif file_format == "tsv":
                try:
                    df = pd.read_csv(file_path, sep="\t")
                except Exception:
                    df = pd.read_csv(file_path, sep="\t", on_bad_lines="skip", engine="python")
            elif file_format == "parquet":
                df = pd.read_parquet(file_path)
            elif file_format == "jsonl":
                df = pd.read_json(file_path, lines=True)
            else:
                df = pd.read_json(file_path)
        except Exception as exc:
            logger.warning("Fallback manifest skipped unreadable file %s: %s", file_path, exc)
            return None

        columns = [str(col) for col in df.columns]
        text_col, label_col = _infer_text_and_label_columns(columns)
        label_distribution: dict[str, Any] = {}
        if label_col and label_col in df.columns:
            try:
                label_distribution = {
                    str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).head(20).items()
                }
            except Exception:
                label_distribution = {}

        return {
            "name": file_path.stem,
            "description": f"Fallback manifest entry auto-generated from {file_path.name}.",
            "local_path": str(file_path.resolve()),
            "format": file_format,
            "files": [file_path.name],
            "schema": {
                "text_column": text_col,
                "label_column": label_col,
                "label_type": str(df[label_col].dtype) if label_col and label_col in df.columns else None,
            },
            "statistics": {
                "num_samples": int(len(df)),
                "num_columns": int(len(columns)),
                "label_distribution": label_distribution,
            },
            "label_mapping": {},
            "loading_snippet": self._build_loading_snippet(file_path, file_format, text_col, label_col),
        }

    def _write_fallback_manifest(self, manifest_path: Path, node_id: str) -> dict | None:
        data_links_dir = self.task_workspace / "data_links"
        if not data_links_dir.exists():
            return None

        candidates: list[Path] = []
        for file_path in sorted(data_links_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in {".csv", ".tsv", ".parquet", ".json", ".jsonl"}:
                candidates.append(file_path)

        entries: list[dict[str, Any]] = []
        for file_path in candidates[:3]:
            entry = self._infer_dataset_entry(file_path)
            if entry:
                entries.append(entry)

        if not entries:
            return None

        manifest = {
            "version": "1.0",
            "task_id": self.exp_id or "unknown",
            "created_by_node": node_id,
            "generated_by": "red_fallback",
            "external_datasets": entries,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def run(
        self,
        task_description: str,
        prev_code: str,
        memory: str,
        term_out: str = "",
        best_code: str | None = None,
        best_metric: float | None = None,
    ) -> dict:
        """运行 Red 侦察节点。

        Returns:
            包含 plan, code, raw_response, exec, manifest, metric 字段的字典。
            metric 为 None（Red 不评估 F1），由 playground 检查 manifest。
        """
        node_id = self.node.id
        BaseAgent.set_exp_info(exp_name=f"red_{node_id[:8]}", exp_index=self.exp_index)

        manifest_path = self.manifest_path

        # 读取已有的 manifest 内容注入 prompt（避免重复搜索相同数据源）
        existing_manifest_str = ""
        existing = load_manifest(self.task_workspace, self.input_manifest_path)
        if existing:
            existing_manifest_str = json.dumps(existing, ensure_ascii=False, indent=2)
        demand_spec_section = (
            json.dumps(self.demand_spec, ensure_ascii=False, indent=2)
            if self.demand_spec
            else "（暂无结构化需求，请根据父节点方案与任务描述自行提炼）"
        )

        tools_manual_path = Path("playground/ml_master_datatree/prompts/general/tools_manual.md")
        with open(tools_manual_path, encoding="utf-8") as f:
            tools_manual = f.read()

        general_instructions_path = Path("playground/ml_master_datatree/prompts/general/general_instructions.md")
        with open(general_instructions_path, encoding="utf-8") as f:
            general_instructions = f.read()

        fmt = {
            "task_description": task_description,
            "general_instruction_content": general_instructions,
            "previous_code": prev_code,
            "execution_output": term_out,
            "memory": memory,
            "best_metric": best_metric or "N/A",
            "data_preview": self.data_preview,
            "workspace": str(self.workspace),
            "task_workspace": str(self.task_workspace),
            "manifest_path": str(manifest_path),
            "node_id": node_id,
            "operation_tools_readme": tools_manual,
            "existing_manifest": existing_manifest_str or "（暂无已有 manifest）",
            "demand_spec_section": demand_spec_section,
        }

        orig_fmt = self.agent._prompt_format_kwargs.copy()
        self.agent._prompt_format_kwargs.update(fmt)
        try:
            task = TaskInstance(
                task_id=f"{node_id}_red",
                task_type="red",
                description=task_description,
                input_data={},
            )
            traj = self.agent.run(task)
            text = extract_agent_response(traj)
        finally:
            self.agent._prompt_format_kwargs = orig_fmt

        # Red 不运行训练代码，但仍需要一个 exec_res 供 playground 检查
        # 通过检查 manifest 文件是否写入成功来判断
        manifest_data = load_manifest(self.task_workspace, manifest_path)
        fallback_used = False
        if not manifest_has_datasets(manifest_data):
            manifest_data = self._write_fallback_manifest(manifest_path, node_id)
            fallback_used = manifest_has_datasets(manifest_data)
        manifest_ok = manifest_has_datasets(manifest_data)

        exec_res = {
            "stdout": (
                f"[Red Scout] manifest_written={manifest_ok} "
                f"fallback_used={fallback_used} path={manifest_path}"
            ),
            "exit_code": 0 if manifest_ok else 1,
            "script": "",
            "code": "",
        }

        return {
            "plan": "",
            "code": "",  # Red 不产出训练代码
            "raw_response": text,
            "exec": exec_res,
            "manifest": manifest_data,
            # metric=None 表示 Red 不参与 F1 评分；
            # playground 会用 manifest_ok 决定是否 buggy
            "metric": None,
            "metric_detail": {
                "is_bug": not manifest_ok,
                "has_submission": False,
                "manifest_ok": manifest_ok,
                "fallback_used": fallback_used,
                "manifest_path": str(manifest_path),
            },
        }
