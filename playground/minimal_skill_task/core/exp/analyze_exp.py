"""AnalyzeExp：仅包含 analyze agent，查看数据库结构并输出 query 写作规范"""

import logging
from pathlib import Path
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.rag_utils import extract_agent_response, update_agent_format_kwargs


def _project_root() -> Path:
    """EvoMaster 项目根目录（含 evomaster/、playground/、configs/；exp -> 上五级）"""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


class AnalyzeExp(BaseExp):
    def __init__(self, analyze_agent, config):
        super().__init__(analyze_agent, config)
        self.analyze_agent = analyze_agent
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        task_description: str,
        db: dict,
        task_id: str = "exp_001",
    ) -> tuple[str, Any]:
        """运行 Analyze Agent，返回 (analyze_output, trajectory)。"""
        self.logger.info("Starting AnalyzeExp")
        root = _project_root()
        vec_dir = db["vec_dir"]
        # db 已在 workflow 中转为绝对路径；若仍是相对路径则相对项目根
        vec_path = Path(vec_dir) if Path(vec_dir).is_absolute() else root / vec_dir
        nodes_jsonl_path = vec_path / "nodes.jsonl"

        # 与 minimal_kaggle 一致：运行时再注入 task_description 等，模板用 {task_description}
        update_agent_format_kwargs(
            self.analyze_agent,
            task_description=task_description,
            vec_dir=vec_dir,
            nodes_data=db["nodes_data"],
            model=db["model"],
            nodes_jsonl_path=str(nodes_jsonl_path),
        )
        task = TaskInstance(
            task_id=f"{task_id}_analyze",
            task_type="analyze",
            description=task_description,
            input_data={},
        )
        trajectory = self.analyze_agent.run(task)
        output = extract_agent_response(trajectory)
        self.logger.info("AnalyzeExp completed")
        return output, trajectory
