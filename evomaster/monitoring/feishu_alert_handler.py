"""
飞书告警日志处理器
"""

import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import traceback


class FeishuAlertHandler(logging.Handler):
    """
    将logging.error和logging.warning转发到飞书的日志处理器

    使用方式：
        handler = FeishuAlertHandler(agent, alert_manager)
        logger.addHandler(handler)
    """

    def __init__(self, agent, alert_manager):
        """
        初始化处理器

        Args:
            agent: BaseAgent实例，用于获取上下文信息
            alert_manager: FeishuAlertManager实例，用于发送告警
        """
        super().__init__()
        self.agent = agent
        self.alert_manager = alert_manager
        self._lock = threading.Lock()

        # 只处理ERROR和WARNING级别
        self.setLevel(logging.WARNING)

    def emit(self, record):
        """
        处理日志记录

        Args:
            record: 日志记录对象
        """
        # 只处理ERROR和WARNING
        if record.levelno < logging.WARNING:
            return

        try:
            with self._lock:
                # 提取关键信息
                message = record.getMessage()
                level = logging.getLevelName(record.levelno)

                # 构建上下文信息
                context = self._build_context(record)

                # 发送飞书告警
                self.alert_manager.send_alert(
                    message=message,
                    level=level,
                    context=context
                )

        except Exception as e:
            # 避免日志处理器本身产生错误，导致循环
            # 静默处理，避免循环告警
            pass

    def _build_context(self, record) -> Dict[str, Any]:
        """
        从日志记录和agent构建上下文信息

        Args:
            record: 日志记录对象

        Returns:
            包含上下文信息的字典
        """
        context = {}

        # 从agent获取上下文
        try:
            if hasattr(self.agent, 'playground') and self.agent.playground:
                if hasattr(self.agent.playground, 'run_dir'):
                    context['run_dir'] = str(self.agent.playground.run_dir)

                if hasattr(self.agent.playground, 'config'):
                    config = self.agent.playground.config
                    # 尝试从config中获取exp_id
                    if hasattr(config, 'exp_id'):
                        context['exp_id'] = config.exp_id
                    elif isinstance(config, dict):
                        context['exp_id'] = config.get('exp_id')

            # 从agent获取任务信息
            if hasattr(self.agent, 'trajectory') and self.agent.trajectory:
                if hasattr(self.agent.trajectory, 'task_id'):
                    context['task_id'] = self.agent.trajectory.task_id

            # 获取步骤数
            if hasattr(self.agent, '_step_count'):
                context['step_count'] = self.agent._step_count

        except Exception as e:
            # 如果获取上下文失败，不影响告警发送
            pass

        # 从日志记录获取异常信息
        if hasattr(record, 'exc_info') and record.exc_info:
            context['error_type'] = record.exc_info[0].__name__
            context['stack_trace'] = ''.join(
                traceback.format_exception(*record.exc_info)
            )

        # 添加时间戳
        context['timestamp'] = datetime.now().isoformat()

        return context
