"""Minimal Playground 实现

最简单的 playground 实现，展示如何使用 EvoMaster 基础功能。
"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("minimal")
class MinimalPlayground(BasePlayground):
    """Minimal Playground

    最简单的 playground 实现，展示如何使用 EvoMaster 基础功能。
    当前使用默认的 BasePlayground 行为，未来可以添加定制逻辑。

    使用方式：
        # 通过统一入口
        python run.py --agent minimal --task "任务描述"

        # 或使用独立入口
        python playground/minimal/main.py
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """初始化 MinimalPlayground

        Args:
            config_dir: 配置目录路径，默认为 configs/minimal/
            config_path: 配置文件完整路径（如果提供，会覆盖 config_dir）
        """
        if config_path is None and config_dir is None:
            # 默认配置目录
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "minimal"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
