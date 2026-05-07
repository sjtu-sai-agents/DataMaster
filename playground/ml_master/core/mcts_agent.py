"""ML-Master MCTS Agent implementation

This implements the core MCTS-based search agent that generates,
executes, and iteratively improves machine learning solutions.
"""

import logging
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from ..core import MCTSNode, Journal, MetricValue, get_worst_metric, SearchConfig, ExecutionResult
from ..utils import plan_and_code_query, plan_and_code_query_with_agent, query_with_feedback, wrap_code
from ..utils.mcts_utils import linear_decay, exponential_decay, piecewise_decay, dynamic_piecewise_decay
from ..prompts import get_draft_prompt, get_improve_prompt, get_debug_prompt, get_review_prompt, get_evaluation_instructions

logger = logging.getLogger(__name__)
GREEN = "\033[92m"
RESET = "\033[0m"



ExecCallbackType = Callable[[str, str, bool], ExecutionResult]


@dataclass
class DecayConfig:
    """Configuration for exploration decay."""

    decay_type: Literal["none", "linear", "exponential", "piecewise", "dynamic_piecewise"] = "none"
    exploration_constant: float = 1.414
    lower_bound: float = 0.1

    # Linear decay
    alpha: float = 0.01

    # Exponential decay
    gamma: float = 0.995

    # Piecewise decay
    phase_ratios: list = field(default_factory=lambda: [0.3, 0.3, 0.4])


@dataclass
class AgentConfig:
    """Configuration for ML-Master Agent."""

    # Search limits
    steps: int = 500
    time_limit: int = 43200  # 12 hours

    # Code generation config
    code_model: str = "gpt-4"
    code_temp: float = 0.5
    code_base_url: str | None = None
    code_api_key: str | None = None

    # Feedback config
    feedback_model: str = "gpt-4o"
    feedback_temp: float = 0.0
    feedback_base_url: str | None = None
    feedback_api_key: str | None = None

    # Search config
    num_drafts: int = 5
    num_bugs: int = 3
    num_improves: int = 3
    parallel_search_num: int = 3
    exploration_constant: float = 1.414
    metric_improvement_threshold: float = 0.001
    max_improve_failure: int = 2
    max_debug_depth: int = 3
    back_debug_depth: int = 1
    invalid_metric_upper_bound: int = 50

    # Behavior flags
    obfuscate: bool = False
    steerable_reasoning: bool = False
    check_format: bool = False
    expose_prediction: bool = False
    k_fold_validation: int = 1
    save_all_submission: bool = False
    convert_system_to_user: bool = False
    preprocess_data: bool = False

    # Agent mode config
    use_agent_mode: bool = False      # Use Agent framework with tool calling
    max_turns: int = 10               # Max turns for each Agent call

    # Decay config
    decay: DecayConfig = field(default_factory=DecayConfig)


class MLMasterAgent:
    """ML-Master MCTS Agent for automated machine learning.

    This agent uses Monte Carlo Tree Search to explore the space of
    machine learning solutions, generating code, executing it,
    evaluating results, and iteratively improving.
    """

    def __init__(
        self,
        task_desc: str,
        cfg: AgentConfig,
        journal: Journal,
        llm,
        feedback_llm,
        workspace_dir: str | Path,
        session=None,  # Session for tool execution (Agent mode)
        tools=None,    # ToolRegistry (Agent mode)
    ):
        """Initialize the ML-Master Agent.

        Args:
            task_desc: Task description
            cfg: Agent configuration
            journal: Journal for storing the solution tree
            llm: LLM for code generation
            feedback_llm: LLM for feedback/evaluation
            workspace_dir: Working directory for code execution
            session: Session for tool execution (required for Agent mode)
            tools: ToolRegistry for Agent mode
        """
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.llm = llm
        self.feedback_llm = feedback_llm
        self.workspace_dir = Path(workspace_dir)
        self.session = session
        self.tools = tools
        if not self.tools:
            logger.warning("Warning: Tools not configured for agents")
        if not self.session:
            logger.warning("Warning: Sessions empty")
        logger.info(f"Using Agent Mode: {self.cfg.use_agent_mode}")

        # Search state
        self.current_step = 0
        self.current_node: MCTSNode | None = None
        self.best_metric: float | None = None
        self.best_node: MCTSNode | None = None
        self.search_start_time: float | None = None
        self.start_time = time.time()

        # State management (matching ML-Master)
        self.all_root = []  # All draft nodes
        self._locked_drafts = set()  # Locked draft node IDs

        # Virtual root node
        self.virtual_root = MCTSNode(
            parent=None,
            plan="virtual root",
            code="# virtual root",
            metric=get_worst_metric(True),
            stage="root"
        )
        self.journal.append(self.virtual_root)

        # Data preview (cached)
        self.data_preview: str | None = None

        logger.info(f"ML-Master Agent initialized with {cfg.steps} steps")

    def update_data_preview(self, data_preview: str) -> None:
        """Update the data preview for prompts.

        Args:
            data_preview: Data preview string
        """
        self.data_preview = data_preview

    @property
    def _prompt_environment(self) -> dict:
        """Get environment prompt section."""
        pkgs = [
            "numpy", "pandas", "scikit-learn", "statsmodels",
            "xgboost", "lightgbm", "torch", "torchvision",
            "transformers", "nltk", "spacy"
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        return {
            "Installed Packages": (
                f"Your solution can use relevant ML packages such as: {pkg_str}. "
                "Feel free to use any other packages too (all packages are already installed!)."
            )
        }

    def _get_environment(self) -> dict:
        """Get environment as dict (for prompts module)."""
        return self._prompt_environment

    @property
    def _prompt_impl_guideline(self) -> dict:
        """Get implementation guidelines."""
        return {"Implementation guideline": self._get_impl_guideline_list()}

    def _get_impl_guideline_list(self) -> list:
        """Get implementation guidelines as list (matching ML-Master original)."""
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.cfg.time_limit - tot_time_elapsed

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {self._format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.cfg.steps - self.current_step}>",
            "The code must print the evaluation metric computed on a hold-out validation set.",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program that is self-contained.",
            "Your response should only contain a single code block.",
            "Be aware of the running time, it should complete within a reasonable time.",
            'All the provided input data is stored in "./input" directory.',
            '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.",
            "If you use `DataLoader`, you need to increase the parameter `num_workers` to speed up the training process."
        ]

        if self.cfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.cfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.cfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return impl_guideline

    def _format_time(self, seconds: float) -> str:
        """Format seconds to readable time string."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hrs}hrs {mins}mins {secs}secs"

    def _draft(self) -> MCTSNode:
        """Generate a new draft node.

        Returns:
            New MCTSNode with draft stage
        """
        logger.info("Starting draft: generating new node")

        # Get implementation guidelines and environment
        impl_guideline = self._get_impl_guideline_list()
        environment = self._get_environment()

        # Build prompt using prompts module
        prompt_data = get_draft_prompt(
            task_desc=self.task_desc,
            memory=self.virtual_root.fetch_child_memory(),
            data_preview=self.data_preview,
            impl_guideline=impl_guideline,
            environment=environment,
            obfuscate=self.cfg.obfuscate,
        )

        messages = [
            {"role": "system", "content": prompt_data["introduction"]},
            {"role": "user", "content": prompt_data["user_prompt"]}
        ]

        self.virtual_root.add_expected_child_count()
        

        # Choose query method based on agent mode config
        if self.cfg.use_agent_mode and self.session and self.tools:
            logger.info(f"{GREEN}Agent mode enabled with max_turns={self.cfg.max_turns} for drafting{RESET}")
            plan, code = plan_and_code_query_with_agent(
                llm=self.llm,
                session=self.session,
                tools=self.tools,
                system_prompt=prompt_data["introduction"],
                user_prompt=prompt_data["user_prompt"],
                max_turns=self.cfg.max_turns,
                temperature=self.cfg.code_temp,
            )
        else:
            logger.info(f"{GREEN}Agent mode disabled with max_turns={self.cfg.max_turns} for drafting{RESET}")
            plan, code = plan_and_code_query(
                self.llm,
                messages,
                temperature=self.cfg.code_temp,
                steerable_reasoning=self.cfg.steerable_reasoning
            )

        new_node = MCTSNode(
            plan=plan,
            code=code,
            parent=self.virtual_root,
            stage="draft",
            local_best_node=self.virtual_root
        )
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _improve(self, parent_node: MCTSNode) -> MCTSNode:
        """Generate an improvement node.

        Args:
            parent_node: Parent node to improve

        Returns:
            New MCTSNode with improve stage
        """
        logger.info(f"Starting improve: enhancing node {parent_node.id}")

        # Get implementation guidelines
        impl_guideline = self._get_impl_guideline_list()

        # Build prompt using prompts module
        prompt_data = get_improve_prompt(
            task_desc=self.task_desc,
            memory=parent_node.fetch_child_memory(),
            data_preview=self.data_preview,
            previous_code=wrap_code(parent_node.code),
            execution_output=wrap_code(parent_node.term_out, lang=""),
            impl_guideline=impl_guideline,
            obfuscate=self.cfg.obfuscate,
        )

        messages = [
            {"role": "system", "content": prompt_data["introduction"]},
            {"role": "user", "content": prompt_data["user_prompt"]}
        ]

        parent_node.add_expected_child_count()

        # Choose query method based on agent mode config
        if self.cfg.use_agent_mode and self.session and self.tools:
            logger.info(f"{GREEN}Agent mode enabled with max_turns={self.cfg.max_turns} for improvement{RESET}")
            plan, code = plan_and_code_query_with_agent(
                llm=self.llm,
                session=self.session,
                tools=self.tools,
                system_prompt=prompt_data["introduction"],
                user_prompt=prompt_data["user_prompt"],
                max_turns=self.cfg.max_turns,
                temperature=self.cfg.code_temp,
            )
        else:
            logger.info(f"{GREEN}Agent mode disabled with max_turns={self.cfg.max_turns} for improvement{RESET}")
            plan, code = plan_and_code_query(
                self.llm,
                messages,
                temperature=self.cfg.code_temp,
                steerable_reasoning=self.cfg.steerable_reasoning
            )

        new_node = MCTSNode(
            plan=plan,
            code=code,
            parent=parent_node,
            stage="improve",
            local_best_node=parent_node.local_best_node
        )
        logger.info(f"Improved node {parent_node.id} -> {new_node.id}")
        return new_node

    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
        """Generate a debug node.

        Args:
            parent_node: Parent node with bugs

        Returns:
            New MCTSNode with debug stage
        """
        logger.info(f"Starting debug: fixing node {parent_node.id}")

        # Get implementation guidelines
        impl_guideline = self._get_impl_guideline_list()

        # Build prompt using prompts module
        prompt_data = get_debug_prompt(
            task_desc=self.task_desc,
            data_preview=self.data_preview,
            buggy_code=wrap_code(parent_node.code),
            execution_output=wrap_code(parent_node.term_out, lang=""),
            impl_guideline=impl_guideline,
            obfuscate=self.cfg.obfuscate,
            check_format=self.cfg.check_format,
        )

        messages = [
            {"role": "system", "content": prompt_data["introduction"]},
            {"role": "user", "content": prompt_data["user_prompt"]}
        ]

        parent_node.add_expected_child_count()

        # Choose query method based on agent mode config
        if self.cfg.use_agent_mode and self.session and self.tools:
            logger.info(f"{GREEN}Agent mode enabled with max_turns={self.cfg.max_turns} for debugging{RESET}")
            plan, code = plan_and_code_query_with_agent(
                llm=self.llm,
                session=self.session,
                tools=self.tools,
                system_prompt=prompt_data["introduction"],
                user_prompt=prompt_data["user_prompt"],
                max_turns=self.cfg.max_turns,
                temperature=self.cfg.code_temp,
            )
        else:
            logger.info(f"{GREEN}Agent mode disabled with max_turns={self.cfg.max_turns} for debugging{RESET}")
            plan, code = plan_and_code_query(
                self.llm,
                messages,
                temperature=self.cfg.code_temp,
                steerable_reasoning=self.cfg.steerable_reasoning
            )

        new_node = MCTSNode(
            plan=plan,
            code=code,
            parent=parent_node,
            stage="debug",
            local_best_node=parent_node.local_best_node
        )
        logger.info(f"Debugged node {parent_node.id} -> {new_node.id}")
        return new_node

    def parse_exec_result(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        """Parse execution results and evaluate the node.

        This follows ML-Master's evaluation logic to ensure consistency.

        Args:
            node: The node whose execution results to parse
            exec_result: The execution result

        Returns:
            The updated node
        """
        logger.info(f"Parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        # Build review prompt using prompts module
        prompt = get_review_prompt(
            task_desc=self.task_desc,
            code=wrap_code(node.code),
            execution_output=wrap_code(node.term_out, lang=""),
        )

        # Add evaluation instructions
        evaluation_instructions = get_evaluation_instructions()

        response = query_with_feedback(
            self.feedback_llm,
            system_prompt={**prompt, "Instructions": evaluation_instructions.strip()},
            user_prompt=None,
            temperature=self.cfg.feedback_temp,
        )

        # Validate metric - must be a float
        if not isinstance(response.get("metric"), float):
            response["metric"] = None

        # Check for submission file (do an extra check to catch cases where LLM fails)
        has_csv_submission = (
            self.workspace_dir / "submission" / f"submission_{node.id}.csv"
        ).exists()

        node.analysis = response.get("summary", "")

        # Determine if buggy - matches ML-Master's logic exactly
        node.is_buggy = (
            response.get("is_bug", False)
            or node.exc_type is not None
            or response.get("metric") is None
            or not has_csv_submission
        )

        # Log why node is marked buggy (for debugging)
        if node.is_buggy:
            if response.get("is_bug"):
                logger.warning(f"Node {node.id} is marked as buggy because response['is_bug'] is True")
            elif node.exc_type is not None:
                logger.warning(f"Node {node.id} is marked as buggy because node.exc_type is not None: {node.exc_type}")
            elif response.get("metric") is None:
                logger.warning(f"Node {node.id} is marked as buggy because response['metric'] is None")
            elif not has_csv_submission:
                logger.warning(f"Node {node.id} is marked as buggy because has_csv_submission is False")

        if node.is_buggy:
            logger.info(f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv")
            node.metric = get_worst_metric(True)
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy with metric {response.get('metric')}")
            lower_is_better = response.get("lower_is_better", False)
            node.metric = MetricValue(
                response["metric"],
                maximize=not lower_is_better
            )

        return node

    def backpropagate(self, node: MCTSNode, value: float, add_to_tree: bool = True) -> None:
        """Backpropagate reward through the tree (matching ML-Master original).

        Args:
            node: Node to start backpropagation from
            value: Reward value
            add_to_tree: Whether to add to tree (for journal tracking)
        """
        logger.info(f"node {node.id} start backpropagating with reward {value}.")

        while node is not None:
            # Update debug success status (matching ML-Master logic)
            if node.parent:
                if node.is_buggy is False and node.parent.is_buggy is True:
                    node.parent.is_debug_success = True
                elif node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
                    node.parent.is_debug_success = True

                # Propagate continue_improve flag
                if node.parent.stage != "root":
                    node.parent.continue_improve = node.continue_improve

            # Unlock draft nodes
            if node.stage == "draft" and node.lock:
                node.lock = False
                logger.info(f"Draft node {node.id} is unlocked.")

            # Reset improve failure depth
            if node.improve_failure_depth > 0:
                node.improve_failure_depth = 0

            node.update(value, add_to_tree)
            node = node.parent

    def check_improvement(self, cur_node: MCTSNode, parent_node: MCTSNode) -> bool:
        """Check if improvement is sufficient or should continue.

        Args:
            cur_node: Current node
            parent_node: Parent node

        Returns:
            True if should backpropagate, False if should continue improving
        """
        improvement = 0
        should_backpropagate = False
        local_best_node = cur_node.local_best_node

        if local_best_node is None:
            local_best_node = cur_node
            cur_node.local_best_node = cur_node
            cur_node.continue_improve = True
            return False

        local_best_metric = local_best_node.metric.value if local_best_node.metric else None

        if cur_node.is_buggy is False:
            new_metric = cur_node.metric.value if cur_node.metric else None

            if new_metric and local_best_metric:
                if cur_node.metric.maximize:
                    improvement = new_metric - local_best_metric
                else:
                    improvement = local_best_metric - new_metric

                scfg = self._get_search_config()
                if improvement < scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < scfg.max_improve_failure:
                    local_best_node.improve_failure_depth += 1
                    logger.warning(f"Improvement {improvement} below threshold, trying again ({local_best_node.improve_failure_depth}/{scfg.max_improve_failure})")
                    cur_node.continue_improve = True
                elif improvement < scfg.metric_improvement_threshold:
                    logger.warning(f"Max improve attempts reached, backpropagating")
                    cur_node.continue_improve = False
                    should_backpropagate = True
                    cur_node.is_terminal = True
                else:
                    logger.info(f"Improvement {improvement} above threshold, continuing")
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
            elif new_metric:
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
        else:
            if cur_node.debug_depth >= self._get_search_config().back_debug_depth:
                should_backpropagate = True
                scfg = self._get_search_config()
                if cur_node.debug_depth >= scfg.max_debug_depth:
                    cur_node.is_terminal = True

        if should_backpropagate:
            reward = self._get_node_reward(cur_node)
            self.backpropagate(cur_node, reward)

        return should_backpropagate

    def _get_node_reward(self, node: MCTSNode) -> float:
        """Calculate reward for a node.

        Args:
            node: The node

        Returns:
            Reward value
        """
        reward = 0

        if node.is_buggy or node.metric.value is None:
            return -1

        if self.best_metric is not None and node.metric.value is not None:
            if node.metric.maximize:
                improvement = node.metric.value - self.best_metric
            else:
                improvement = self.best_metric - node.metric.value

            if improvement > 0:
                logger.info(f"Node {node.id} improves over best!")
                reward += 1

        if node.parent and node.parent.is_buggy:
            reward += 1
        else:
            reward += 1

        return reward

    def _get_search_config(self) -> SearchConfig:
        """Get search configuration from agent config.

        Returns:
            SearchConfig instance
        """
        return SearchConfig(
            max_debug_depth=self.cfg.max_debug_depth,
            debug_prob=0.0,
            num_drafts=self.cfg.num_drafts,
            invalid_metric_upper_bound=self.cfg.invalid_metric_upper_bound,
            metric_improvement_threshold=self.cfg.metric_improvement_threshold,
            back_debug_depth=self.cfg.back_debug_depth,
            num_bugs=self.cfg.num_bugs,
            num_improves=self.cfg.num_improves,
            max_improve_failure=self.cfg.max_improve_failure,
            parallel_search_num=self.cfg.parallel_search_num,
            exploration_constant=self.cfg.exploration_constant,
        )

    def get_C(self) -> float:
        """Get exploration constant with decay (matching ML-Master original).

        Returns:
            Current exploration constant (possibly decayed)
        """
        dcfg = self.cfg.decay

        if dcfg.decay_type == "linear":
            return linear_decay(
                t=self.current_step,
                initial_C=dcfg.exploration_constant,
                alpha=dcfg.alpha,
                lower_bound=dcfg.lower_bound
            )
        elif dcfg.decay_type == "exponential":
            return exponential_decay(
                t=self.current_step,
                initial_C=dcfg.exploration_constant,
                gamma=dcfg.gamma,
                lower_bound=dcfg.lower_bound
            )
        elif dcfg.decay_type == "piecewise":
            # Calculate T1 and T2 from phase_ratios
            phase_ratios = dcfg.phase_ratios
            T1 = int(self.cfg.steps * phase_ratios[0])
            T2 = int(self.cfg.steps * (phase_ratios[0] + phase_ratios[1]))
            return piecewise_decay(
                t=self.current_step,
                initial_C=dcfg.exploration_constant,
                T1=T1,
                T2=T2,
                alpha=dcfg.alpha,
                lower_bound=dcfg.lower_bound
            )
        elif dcfg.decay_type == "dynamic_piecewise":
            if self.search_start_time is None:
                self.search_start_time = time.time()
            return dynamic_piecewise_decay(
                steps_limit=self.cfg.steps,
                n_nodes=self.current_step,
                initial_C=dcfg.exploration_constant,
                start_time=self.search_start_time,
                time_limit=self.cfg.time_limit,
                alpha=dcfg.alpha,
                lower_bound=dcfg.lower_bound,
                phase_ratios=dcfg.phase_ratios
            )
        else:  # "none"
            return dcfg.exploration_constant

    def is_root(self, node: MCTSNode) -> bool:
        """Check if a node is the virtual root.

        Args:
            node: Node to check

        Returns:
            True if node is virtual root
        """
        return node == self.virtual_root

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select a node for expansion using UCT (matching ML-Master original).

        Args:
            node: Starting node (usually virtual root)

        Returns:
            Selected node for expansion
        """
        scfg = self._get_search_config()
        logger.info(f"[select] Processing node: {node.id}")

        current = node
        while current and not current.is_terminal:
            if not current.is_fully_expanded_with_expected(scfg):
                if current.is_buggy and current.is_debug_success is True:
                    current = self._uct_select(current)
                elif current.continue_improve and len(current.children) > 0:
                    current = self._uct_select(current)
                else:
                    logger.info(f"Node {current.id} is not fully expanded, expanding")
                    return current
            else:
                current = self._uct_select(current)

        logger.info(f"[select] choose a node for expanding: {current.id}")
        return current

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCT value (matching ML-Master original).

        Args:
            node: Parent node

        Returns:
            Child with highest UCT
        """
        if self.is_root(node):
            filtered_children = [child for child in node.children if not child.lock]
            logger.info(f"For node {node.id}, there are {len(node.children) - len(filtered_children)}/{len(node.children)} is locked.")
            selected_node = node
            if len(filtered_children) > 0:
                selected_node = max(filtered_children, key=lambda child: child.uct_value(exploration_constant=self.get_C()))

            if selected_node.stage == "draft":
                selected_node.lock = True
                logger.info(f"Draft node {selected_node.id} is locked.")
            return selected_node
        else:
            return max(node.children, key=lambda child: child.uct_value(exploration_constant=self.get_C()))

    def _step_search(self, parent_node: MCTSNode, exec_callback: ExecCallbackType) -> tuple[bool, MCTSNode | None]:
        """Execute one search step.

        Args:
            parent_node: Node to expand from
            exec_callback: Callback for executing code

        Returns:
            Tuple of (should_return_to_root, result_node)
        """
        logger.info(f"Search step from node {parent_node.id}")
        result_node = None

        if not parent_node.is_terminal:
            try:
                # Generate new node
                if parent_node == self.virtual_root:
                    result_node = self._draft()
                    result_node.lock = True
                elif parent_node.is_buggy:
                    result_node = self._debug(parent_node)
                elif parent_node.is_buggy is False:
                    result_node = self._improve(parent_node)
                else:
                    logger.warning(f"Unexpected node state for {parent_node.id}")

                if result_node:
                    # Execute code
                    exe_res = exec_callback(result_node.code, result_node.id, True)
                    result_node = self.parse_exec_result(result_node, exe_res)

                    # Check for submission file
                    submission_path = self.workspace_dir / "submission" / f"submission_{result_node.id}.csv"
                    if not submission_path.exists():
                        result_node.is_buggy = True
                        result_node.metric = get_worst_metric(result_node.metric.maximize if result_node.metric else True)
                        logger.info(f"Node {result_node.id} did not produce submission.csv")

                    logger.info(f"Node {result_node.id} metric: {result_node.metric.value}")

                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

                    if parent_node.is_buggy and result_node.is_buggy is False:
                        parent_node.is_debug_success = True

                    _root = self.check_improvement(result_node, parent_node)

                    if self.best_node and result_node.metric:
                        if self.best_node.metric.maximize != result_node.metric.maximize:
                            logger.warning("Metric inconsistency detected!")
                            raise ValueError("Metric direction inconsistency")

                    self.journal.append(result_node)
                    return _root, result_node

            except Exception as e:
                logger.error(f"Error in search step: {e}", exc_info=True)
                self.backpropagate(parent_node, 0, add_to_tree=False)
                if hasattr(parent_node, 'sub_expected_child_count'):
                    parent_node.sub_expected_child_count()
                raise
        else:
            logger.info(f"Node {parent_node.id} is terminal, backpropagating")
            self.backpropagate(parent_node, 0)
            return True, None

        return False, result_node

    def step(self, exec_callback: ExecCallbackType) -> MCTSNode:
        """Execute one MCTS step.

        Args:
            exec_callback: Callback for executing code

        Returns:
            Node to continue from, or virtual_root
        """
        if not self.journal.nodes or self.data_preview is None:
            if self.search_start_time is None:
                self.search_start_time = time.time()

        # Select node to expand
        if self.current_node is None or self.current_node.stage == "root":
            node = self.select(self.virtual_root)
        else:
            node = self.select(self.current_node)

        # Execute search step
        _root, result_node = self._step_search(node, exec_callback)

        # Update best node
        if result_node and result_node.metric.value is not None:
            if self.best_node is None or self.best_node.metric < result_node.metric:
                logger.info(f"Node {result_node.id} is new best! Metric: {result_node.metric.value}")
                self.best_node = result_node
                self.best_metric = result_node.metric.value

                # Save best solution
                self._save_best_solution(result_node)

        self.current_step = len(self.journal) - 1

        if _root or result_node is None:
            logger.info("Returning to virtual root")
            self.current_node = None
            return self.virtual_root
        else:
            self.current_node = result_node
            return result_node

    def _save_best_solution(self, node: MCTSNode) -> None:
        """Save the best solution to disk.

        Args:
            node: The best node
        """
        best_dir = self.workspace_dir / "best_solution"
        best_dir.mkdir(exist_ok=True, parents=True)

        submission_dir = self.workspace_dir / "best_submission"
        submission_dir.mkdir(exist_ok=True, parents=True)

        # Save code
        with open(best_dir / "solution.py", "w") as f:
            f.write(node.code)

        with open(best_dir / "node_id.txt", "w") as f:
            f.write(str(node.id))

        # Copy submission
        src = self.workspace_dir / "submission" / f"submission_{node.id}.csv"
        if src.exists():
            shutil.copy(src, submission_dir / "submission.csv")
            logger.info(f"Saved best solution from node {node.id}")
