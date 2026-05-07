import logging
import os
import uuid
from openai import OpenAI
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from evomaster.agent import BaseAgent


@dataclass
class DataCandidate:
    """数据集候选结果 - 直接包装 Agent 的回答"""
    agent_response: str  # Data Agent 的原始回答
    local_path: str = ""  # 从回答中解析出的本地路径


class DataExp(BaseExp):
    """数据发现实验 - 简化版，只解析 Agent 输出

    Data Exp 的职责：
    1. 使用 LLM-as-a-judge 判断方向是否为数据增强方向
    2. 调用 Data Agent 搜索和下载数据集
    3. 直接返回 Agent 的回答（让 Improve Exp 去处理）
    """

    # 数据增强方向的标识（作为 fallback）
    DATA_DIRECTION_NAMES = ["Data Enhancement", "External Data", "Data Augmentation", "Additional Data"]

    # LLM-as-a-judge 的系统提示词
    _JUDGE_SYSTEM_PROMPT = """You are a binary classification expert. Your task is to determine whether a given improvement direction for a machine learning task is a "Data Enhancement" direction.

A "Data Enhancement" direction is one that requires finding, integrating, or augmenting with external data sources. This includes:
- Searching for external datasets to merge with training data
- Finding additional features from external sources
- Data augmentation techniques that require new data
- Using pre-trained embeddings or features from external datasets

Non-data enhancement directions include:
- Model architecture changes
- Hyperparameter tuning
- Training strategy improvements

- Regularization techniques

Respond with ONLY "YES" or "NO" (without quotes or any additional text)."""

    def __init__(self, data_agent, config, exp_index):
        super().__init__(data_agent, config)
        self.data_agent = data_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.data_agent.session.config.workspace_path
        self.exp_index = exp_index

        # 创建数据存储目录
        self.external_data_dir = os.path.join(self.workspace_path, "external_data")
        os.makedirs(self.external_data_dir, exist_ok=True)

        # 缓存判断结果（避免重复调用 LLM）
        self._direction_cache: Dict[str, bool] = {}

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return f"Data_{self.exp_index}"

    def is_data_direction(self, direction_name: str) -> bool:
        """判断是否为数据增强方向 (使用 LLM-as-a-judge)

        Args:
            direction_name: 改进方向名称或描述

        Returns:
            True 如果是数据增强方向，False 否则
        """
        # 检查缓存
        if direction_name in self._direction_cache:
            return self._direction_cache[direction_name]

        # 首先尝试快速匹配常见关键词（作为优化）
        direction_lower = direction_name.lower()
        for name in self.DATA_DIRECTION_NAMES:
            if name.lower() in direction_lower:
                self._direction_cache[direction_name] = True
                return True

        # 使用 LLM-as-a-judge 进行判断
        try:
            result = self._llm_judge_direction(direction_name)
            self._direction_cache[direction_name] = result
            return result
        except Exception as e:
            self.logger.warning(f"LLM judge failed for direction '{direction_name}': {e}, falling back to keyword matching")
            # fallback: 如果不包含任何数据相关关键词，则认为不是
            return any(keyword in direction_lower for keyword in
                      ["data", "dataset", "external", "augment", "additional", "merge"])

    def _llm_judge_direction(self, direction_name: str) -> bool:
        """使用 LLM 判断方向是否为数据增强方向（使用原始 OpenAI 客户端）

        Args:
            direction_name: 改进方向名称或描述

        Returns:
            True 如果是数据增强方向，False 否则
        """
        # 从配置获取 LLM 配置
        try:
            client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8899/v1",
            )

            # 构造消息（OpenAI 格式）
            messages = [
                {"role": "system", "content": self._JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"""Analyze the following improvement direction for a machine learning task and determine if it is a "Data Enhancement" direction:

Improvement Direction: "{direction_name}"

Is this a Data Enhancement direction? Respond with ONLY "YES" or "NO"."""}
            ]

            # 调用 LLM
            response = client.chat.completions.create(
                model="gpt-oss",
                messages=messages,
                temperature=0.0,
                max_tokens=100,
            )

            content = response.choices[0].message.content or ""

            # 解析响应
            content_upper = content.strip().upper()
            is_data = "YES" in content_upper or "TRUE" in content_upper

            self.logger.info(f"LLM judge for '{direction_name}': {content.strip()} -> {'Data Enhancement' if is_data else 'Not Data Enhancement'}")
            return is_data

        except Exception as e:
            self.logger.error(f"Error: {e}")
            exit(1)

    def run(self, task_description: str, data_preview: str, best_solution: str,
            idea: Tuple[str, str], knowledge: str, task_id: str = "exp_001") -> Optional[DataCandidate]:
        """
        执行数据发现流程

        Args:
            task_description: 任务描述
            data_preview: 数据预览
            best_solution: 当前最佳方案代码
            idea: (idea_key, idea_description) 元组
            knowledge: 已尝试的方法记录

        Returns:
            DataCandidate 如果找到可行的数据集，否则返回 None
        """
        idea_key, idea_description = idea

        self.logger.info("=" * 60)
        self.logger.info("Data Discovery Stage")
        self.logger.info("=" * 60)
        self.logger.info(f"Idea: {idea_description}")

        try:
            # 调用 Data Agent 搜索数据集
            BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)

            data_original_format_kwargs = self.data_agent._prompt_format_kwargs.copy()
            self.data_agent._prompt_format_kwargs.update({
                'task_description': task_description,
                'data_preview': data_preview,
                'idea_description': idea_description,
                'workspace': self.workspace_path,  # 传递 workspace 路径给 Agent
            })

            data_task = TaskInstance(
                task_id=f"{task_id}_data_search",
                task_type="data_search",
                description=f"Search for dataset based on idea: {idea_description}",
                input_data={},
            )

            data_trajectory = self.data_agent.run(data_task)

            # 调试：打印轨迹信息
            self._debug_trajectory(data_trajectory)
            breakpoint()
            agent_response = self._extract_agent_response(data_trajectory)

            # 提取并打印工具调用信息
            self._log_tool_calls(data_trajectory)

            # self.logger.info(f"Data Agent response: {agent_response[:500]}...")
            self.data_agent._prompt_format_kwargs = data_original_format_kwargs

            # 检查是否找到数据集
            if not agent_response or "NO_DATASET_FOUND" in agent_response:
                self.logger.warning("No datasets found")
                return None

            # 直接返回 Agent 的回答，包装成 DataCandidate
            return DataCandidate(
                agent_response=agent_response,
                local_path=""  # 路径由 Data Agent 在工具调用中处理
            )

        except Exception as e:
            self.logger.error(f"Data discovery failed: {e}", exc_info=True)
            return None

    def _log_tool_calls(self, trajectory) -> None:
        """从轨迹中提取并打印工具调用信息

        Args:
            trajectory: 执行轨迹
        """
        if not trajectory or not trajectory.steps:
            return

        self.logger.info("=" * 60)
        self.logger.info("📋 Data Agent Tool Calls Summary:")
        self.logger.info("=" * 60)

        for step_idx, step in enumerate(trajectory.steps, 1):
            # 打印 assistant message 中的工具调用
            if hasattr(step, 'assistant_message') and step.assistant_message.tool_calls:
                for tool_call in step.assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    self.logger.info(f"\n[Step {step_idx}] Tool Call: {tool_name}")
                    self.logger.info(f"  Arguments: {tool_args}")

            # 打印工具响应（输出）
            if hasattr(step, 'tool_responses') and step.tool_responses:
                for tool_response in step.tool_responses:
                    tool_name = tool_response.name if hasattr(tool_response, 'name') else "unknown"
                    tool_output = tool_response.content if hasattr(tool_response, 'content') else ""

                    # 截断过长的输出
                    if len(tool_output) > 1000:
                        tool_output = tool_output[:500] + "\n... [truncated] ...\n" + tool_output[-500:]

                    self.logger.info(f"\n  Tool Output ({tool_name}):")
                    self.logger.info(f"  {tool_output}")

        self.logger.info("=" * 60)

    def _debug_trajectory(self, trajectory) -> None:
        """调试：打印轨迹的详细信息

        Args:
            trajectory: 执行轨迹
        """
        self.logger.info("=" * 80)
        self.logger.info("🔍 DEBUG: Trajectory Information")
        self.logger.info("=" * 80)

        # 基本信息
        self.logger.info(f"Task ID: {trajectory.task_id if hasattr(trajectory, 'task_id') else 'N/A'}")
        self.logger.info(f"Status: {trajectory.status if hasattr(trajectory, 'status') else 'N/A'}")
        self.logger.info(f"Total Steps: {len(trajectory.steps) if hasattr(trajectory, 'steps') else 0}")
        self.logger.info(f"Total Dialogs: {len(trajectory.dialogs) if hasattr(trajectory, 'dialogs') else 0}")

        # 打印每个 step 的详细信息
        if hasattr(trajectory, 'steps') and trajectory.steps:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("📋 Steps Details:")
            self.logger.info("=" * 80)

            for step_idx, step in enumerate(trajectory.steps, 1):
                self.logger.info(f"\n--- Step {step_idx} ---")

                # Assistant message
                if hasattr(step, 'assistant_message'):
                    msg = step.assistant_message
                    self.logger.info(f"Assistant Content Length: {len(msg.content) if hasattr(msg, 'content') and msg.content else 0}")

                    if hasattr(msg, 'content') and msg.content:
                        # 截断显示前500字符
                        content_preview = msg.content[:500] if len(msg.content) > 500 else msg.content
                        self.logger.info(f"Assistant Content Preview:\n{content_preview}")

                    # Tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        self.logger.info(f"Tool Calls Count: {len(msg.tool_calls)}")
                        for tc_idx, tool_call in enumerate(msg.tool_calls, 1):
                            self.logger.info(f"  Tool Call {tc_idx}:")
                            self.logger.info(f"    Name: {tool_call.function.name}")
                            self.logger.info(f"    Arguments: {tool_call.function.arguments[:200]}...")
                    else:
                        self.logger.info("Tool Calls: None")

                # Tool responses
                if hasattr(step, 'tool_responses') and step.tool_responses:
                    self.logger.info(f"Tool Responses Count: {len(step.tool_responses)}")
                    for tr_idx, tool_resp in enumerate(step.tool_responses, 1):
                        self.logger.info(f"  Tool Response {tr_idx}:")
                        self.logger.info(f"    Name: {tool_resp.name if hasattr(tool_resp, 'name') else 'N/A'}")
                        content = tool_resp.content if hasattr(tool_resp, 'content') else ''
                        content_preview = content[:500] if len(content) > 500 else content
                        self.logger.info(f"    Content Preview:\n{content_preview}")
                else:
                    self.logger.info("Tool Responses: None")

        self.logger.info("\n" + "=" * 80)
