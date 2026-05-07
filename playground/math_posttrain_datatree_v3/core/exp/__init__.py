from __future__ import annotations

from pathlib import Path

from evomaster.core.exp import BaseExp


class NodeExp(BaseExp):
    def __init__(self, agent, session, workspace: Path, task_workspace: Path, config, node, exp_index: int = 0):
        super().__init__(agent=agent, config=config)
        self.session = session
        self.workspace = workspace
        self.task_workspace = task_workspace
        self.node = node
        self.exp_index = exp_index
