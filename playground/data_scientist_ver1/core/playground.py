import os
import logging
import sys
from pathlib import Path
import shutil
import copy
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent
    
# 全局管理
# 数据搜集需要 feedback
# 批量模型下载 router


from .exp.draft_exp import DraftExp
from .exp.research_exp import ResearchExp
from .exp.improve_exp import ImproveExp
from .exp.data_exp import DataExp
from .utils.data_preview import generate# type: ignore
from .utils.code import save_code_to_file

@register_playground("data_scientist_ver1")
class DataScientistPlayground(BasePlayground):
    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "data_scientist_ver1"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.draft_agent = None
        self.debug_agent = None
        self.improve_agent = None
        self.reseach_agent = None
        
        # * 新增加的模块 data_agent
        self.data_agent = None
        
        self.knowledge_promotion_agent = None
        self.metric_agent = None

        self.best_score = None
        self.best_solution = None
        self.knowledge = "There is no memory now."

        self.is_lower_better = False
        self.mcp_manager = None

        self.exp_index = 0 # for trajectory visualizing

    def setup(self) -> None:
        self.logger.info("Setting up multi-agent playground...")

        llm_config_dict = self._setup_llm_config()
        self._llm_config_dict = llm_config_dict

        self._setup_session()

        self._setup_tools()

        agents_config = getattr(self.config, 'agents', {})
        if not agents_config:
            raise ValueError(
                "No agents configuration found. "
                "Please add 'agents' section to config.yaml"
            )

        agent_names = ["draft", "debug", "improve", "reseach", "knowledge_promotion", "metric", "data"]

        for name in agent_names:
            if name not in agents_config:
                raise ValueError(f"缺少 agent 配置: {name}")
            cfg = agents_config[name]
            enable_tools = cfg.get("enable_tools", False)
            agent = self._create_agent(
                name=name,
                agent_config=cfg,
                enable_tools=enable_tools,
                llm_config_dict=llm_config_dict,
            )
            setattr(self, name + "_agent", agent)
            self.logger.info("Agent created: %s", name)
            
        self.logger.info("Data Scientist Playground setup complete")

    def compare_score(self, old_score, new_score):
        if old_score is None or new_score is None:
            return True if new_score is not None else False
        if old_score < new_score and self.is_lower_better == False:
            return True
        elif old_score > new_score and self.is_lower_better == True:
            return True
        else:
            return False

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        try:
            self.setup()

            self._setup_trajectory_file(output_file)

            data_knowledge = "NO DATA KNOWLEDGE this time"
            model_knowledge = "NO MODEL KNOWLEDGE this time"
            self.logger.info(f"working_dir: {self.draft_agent.session.config.workspace_path}")
            os.makedirs(os.path.join(self.draft_agent.session.config.workspace_path, "best_submission"), exist_ok=True)
            os.makedirs(os.path.join(self.draft_agent.session.config.workspace_path, "best_solution"), exist_ok=True)
            os.makedirs(os.path.join(self.draft_agent.session.config.workspace_path, "submission"), exist_ok=True)
            os.makedirs(os.path.join(self.draft_agent.session.config.workspace_path, "working"), exist_ok=True)
            data_preview = generate(self.draft_agent.session.config.workspace_path)
            self.logger.info(f"Data preview: {data_preview}")
            self.logger.info("Running experiment...")

            # ==================== Draft Stage ====================
            draft_exp = DraftExp(self.draft_agent, self.debug_agent, self.metric_agent, self.config, self.exp_index)
            self.exp_index += 1
            is_sucess, validation_score, uid, self.best_solution = draft_exp.run(
                task_description, data_preview, data_knowledge, model_knowledge
            )
            if is_sucess:
                self.best_score = validation_score
                shutil.copy(
                    os.path.join(self.draft_agent.session.config.workspace_path, "submission", f"submission_{uid}.csv"),
                    os.path.join(self.draft_agent.session.config.workspace_path, "best_submission", "submission.csv")
                )
                save_code_to_file(
                    os.path.join(self.draft_agent.session.config.workspace_path, "best_solution"),
                    "best_solution.py",
                    self.best_solution
                )

            # ==================== Iteration Stage ====================
            for reseach_round in range(10):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Research Round {reseach_round + 1}/10")
                self.logger.info(f"{'='*60}\n")

                # Research Stage
                research_exp = ResearchExp(self.reseach_agent, self.config, self.exp_index)
                self.exp_index += 1
                research_plan = research_exp.run(task_description, data_preview, self.best_solution, self.knowledge)

                self.logger.info(f"Research plan keys: {list(research_plan.keys())}")

                # Process each direction
                for direction in research_plan:
                    self.logger.info(f"\n--- Processing direction: {direction} ---")

                    # Check if this is a Data Enhancement direction using LLM-as-a-judge
                    # Create a temporary DataExp instance to access the judge method
                    temp_data_exp = DataExp(self.data_agent, self.config, self.exp_index)
                    is_data_direction = temp_data_exp.is_data_direction(direction)

                    direction_best_solution = self.best_solution
                    direction_best_score = self.best_score
                    ideas = list(research_plan[direction].items())

                    for idea in ideas:
                        if is_data_direction:
                            # ==================== Data Discovery Flow ====================
                            self.logger.info(f"Data Enhancement direction detected, running DataExp for idea: {idea[0]}")

                            data_exp = DataExp(self.data_agent, self.config, self.exp_index)
                            self.exp_index += 1
                            data_candidate = data_exp.run(
                                task_description, data_preview, self.best_solution,
                                idea, self.knowledge
                            )

                            if data_candidate is None:
                                # No feasible data found, skip this idea
                                self.logger.warning(f"No feasible data found, skipping idea: {idea[0]}")
                                continue

                            # Data found, now run Improve with data context
                            self.logger.info(f"Data Agent response: {data_candidate.agent_response[:200]}...")
                            improve_exp = ImproveExp(
                                self.improve_agent, self.debug_agent, self.metric_agent,
                                self.config, self.exp_index
                            )
                            self.exp_index += 1
                            is_sucess, validation_score, uid, self.best_solution = improve_exp.run(
                                task_description, data_preview, direction_best_solution, idea,
                                data_context={
                                    "agent_response": data_candidate.agent_response,
                                }
                            )

                        else:
                            # ==================== Normal Improve Flow ====================
                            improve_exp = ImproveExp(
                                self.improve_agent, self.debug_agent, self.metric_agent,
                                self.config, self.exp_index
                            )
                            self.exp_index += 1
                            is_sucess, validation_score, uid, self.best_solution = improve_exp.run(
                                task_description, data_preview, direction_best_solution, idea
                            )

                        # Update best if improved
                        if is_sucess and self.compare_score(direction_best_score, validation_score):
                            direction_best_score = validation_score
                            direction_best_solution = self.best_solution
                            shutil.copy(
                                os.path.join(self.improve_agent.session.config.workspace_path, "submission", f"submission_{uid}.csv"),
                                os.path.join(self.draft_agent.session.config.workspace_path, "best_submission", "submission.csv")
                            )
                            save_code_to_file(
                                os.path.join(self.improve_agent.session.config.workspace_path, "best_solution"),
                                "best_solution.py",
                                self.best_solution
                            )
                            self.logger.info(f"New best score for this direction: {direction_best_score}")

                    # Update global best after processing all ideas in this direction
                    self.best_solution = direction_best_solution
                    self.best_score = direction_best_score
                    self.logger.info(f"Direction '{direction}' completed. Best score: {direction_best_score}")

            result = {
                "status": "completed",
                "steps": self.exp_index,
                "best_score": self.best_score,
            }
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Experiment completed. Final best score: {self.best_score}")
            self.logger.info(f"{'='*60}\n")
            return result

        except Exception as e:
            self.logger.error(f"Data Scientist task execution failed: {e}", exc_info=True)
            result = {
                "status": "failed",
                "steps": 0,
                "error": str(e),
            }
            return result

        finally:
            self.cleanup()
