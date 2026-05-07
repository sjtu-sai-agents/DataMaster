"""SearchExp：包含 plan agent 和 search agent，至少两轮 Plan → Search；两轮全空时放宽 threshold 再检索一轮"""

import logging
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance

from ..utils.rag_utils import (
    parse_plan_output,
    extract_agent_response,
    update_agent_format_kwargs,
)

DEFAULT_QUERY = "Summarize the following machine learning task in one complete English sentence."
RELAXED_THRESHOLD = 2.0  # 多轮结果均为空时放宽 threshold 重试


def _is_result_empty(text: str) -> bool:
    """检索结果是否视为空（无有效 content）"""
    if not text or not text.strip():
        return True
    # 无 node_id / content 等有效信息视为空
    stripped = text.strip().lower()
    if len(stripped) < 30:
        return True
    if "node_id" not in stripped and "content" not in stripped and "distance" not in stripped:
        return True
    return False


class SearchExp(BaseExp):
    def __init__(self, plan_agent, search_agent, config):
        super().__init__(plan_agent, config)
        self.plan_agent = plan_agent
        self.search_agent = search_agent
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        task_description: str,
        analyze_output: str,
        db: dict,
        task_id: str = "exp_001",
    ) -> tuple[str, list]:
        """运行两轮 Plan → Search，返回 (combined_search_results, [trajectories])。"""
        self.logger.info("Starting SearchExp (plan + search, 2 rounds)")
        trajectories = []

        # ---------- Round 1: Plan (initial) ----------
        stage_input = analyze_output or "(无 Analyze 输出)"
        update_agent_format_kwargs(
            self.plan_agent,
            task_description=task_description,
            stage_input=stage_input,
            **db,
        )
        plan_task_1 = TaskInstance(
            task_id=f"{task_id}_plan1",
            task_type="plan",
            description=task_description,
            input_data={},
        )
        plan_traj_1 = self.plan_agent.run(plan_task_1)
        trajectories.append(plan_traj_1)
        plan_output_1 = extract_agent_response(plan_traj_1)

        # ---------- Round 1: Search ----------
        params1 = parse_plan_output(plan_output_1)
        if not params1.get("query"):
            params1["query"] = DEFAULT_QUERY
        update_agent_format_kwargs(self.search_agent, **params1, **db)
        search_task_1 = TaskInstance(
            task_id=f"{task_id}_search1",
            task_type="search",
            description=task_description,
            input_data={},
        )
        search_traj_1 = self.search_agent.run(search_task_1)
        trajectories.append(search_traj_1)
        search_results_1 = extract_agent_response(search_traj_1)

        # ---------- Round 2: Plan (second params) ----------
        first_round_empty = _is_result_empty(search_results_1 or "")
        stage_input_2 = (
            "第一轮检索结果：\n"
            + (search_results_1 or "(无)")
            + "\n\n请给出第二轮 query、top_k、threshold（格式：query: ... top_k: ... threshold: ...）。"
            + ("**第一轮无有效结果，第二轮请务必放宽 threshold（建议 1.5～2.0 或更高），不要沿用过严的 threshold。** " if first_round_empty else "")
            + "query 仍须符合 Analyze 的 (2) query 写作规范，可沿用首轮 query 或在其规范内微调。若认为第一轮已足够可说明仅用第一轮。"
        )
        update_agent_format_kwargs(
            self.plan_agent,
            task_description=task_description,
            stage_input=stage_input_2,
            **db,
        )
        plan_task_2 = TaskInstance(
            task_id=f"{task_id}_plan2",
            task_type="plan",
            description=task_description,
            input_data={},
        )
        plan_traj_2 = self.plan_agent.run(plan_task_2)
        trajectories.append(plan_traj_2)
        plan_output_2 = extract_agent_response(plan_traj_2)

        # ---------- Round 2: Search ----------
        params2 = parse_plan_output(plan_output_2)
        if not params2.get("query"):
            params2 = params1
        update_agent_format_kwargs(self.search_agent, **params2, **db)
        search_task_2 = TaskInstance(
            task_id=f"{task_id}_search2",
            task_type="search",
            description=task_description,
            input_data={},
        )
        search_traj_2 = self.search_agent.run(search_task_2)
        trajectories.append(search_traj_2)
        search_results_2 = extract_agent_response(search_traj_2)

        # 强制要求：多轮结果均为空时，放宽 threshold 后重试一轮
        search_results_3 = None
        if _is_result_empty(search_results_1 or "") and _is_result_empty(search_results_2 or ""):
            self.logger.info("Both rounds empty; retrying with relaxed threshold")
            params_retry = {**params2, "threshold": max(RELAXED_THRESHOLD, params2.get("threshold", 1.5) * 1.2)}
            update_agent_format_kwargs(self.search_agent, **params_retry, **db)
            search_task_3 = TaskInstance(
                task_id=f"{task_id}_search_retry",
                task_type="search",
                description=task_description,
                input_data={},
            )
            search_traj_3 = self.search_agent.run(search_task_3)
            trajectories.append(search_traj_3)
            search_results_3 = extract_agent_response(search_traj_3)

        combined = (search_results_1 or "") + "\n\n--- 第二轮 ---\n\n" + (search_results_2 or "")
        if search_results_3:
            combined += "\n\n--- 放宽 threshold 重试 ---\n\n" + (search_results_3 or "")
        self.logger.info("SearchExp completed")
        return combined, trajectories
