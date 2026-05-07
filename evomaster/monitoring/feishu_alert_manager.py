"""
飞书告警管理器
"""

import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import traceback
from evomaster.utils.feishu_call import send_feishu_message_with_creds
import yaml

logger = logging.getLogger(__name__)


class FeishuAlertManager:
    """飞书告警管理器单例"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.config_path = "configs/feishu_config.yaml"
        self.config = None
        self.enabled = False
        self.app_id = None
        self.app_secret = None
        self.chat_id = None
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """加载飞书配置"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Feishu config file not found: {self.config_path}")
                return

            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            feishu_config = config_data.get('feishu_message_send', {})
            if not feishu_config:
                logger.warning("No feishu_message_send configuration found")
                return

            self.app_id = feishu_config.get('APP_ID')
            self.app_secret = feishu_config.get('APP_SECRET')
            self.chat_id = feishu_config.get('chat_id')

            if self.app_id and self.app_secret and self.chat_id:
                self.enabled = True
                logger.info("Feishu alert manager initialized successfully")
            else:
                logger.warning("Feishu credentials incomplete")

        except Exception as e:
            logger.error(f"Failed to load feishu config: {e}")

    def send_alert(
        self,
        message: str,
        level: str = "INFO",
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        发送飞书告警

        Args:
            message: 告警消息内容
            level: 告警级别 (ERROR, WARNING, INFO)
            context: 上下文信息 (run_dir, exp_id, etc.)

        Returns:
            bool: 发送是否成功
        """
        if not self.enabled:
            logger.debug("Feishu alerts are disabled")
            return False

        try:
            # 构建告警消息
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            emoji_map = {"ERROR": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}
            emoji = emoji_map.get(level, "ℹ️")

            # 构建完整消息
            alert_message = f"""{emoji} [EvoMaster] {level}告警

时间: {timestamp}
级别: {level}
消息: {message}
"""

            # 添加上下文信息
            if context:
                if context.get('run_dir'):
                    alert_message += f"\n运行目录: {context['run_dir']}"
                if context.get('exp_id'):
                    alert_message += f"\n实验ID: {context['exp_id']}"
                if context.get('task_id'):
                    alert_message += f"\n任务ID: {context['task_id']}"
                if context.get('step_count'):
                    alert_message += f"\n步骤数: {context['step_count']}"
                if context.get('error_type'):
                    alert_message += f"\n错误类型: {context['error_type']}"
                if context.get('stack_trace'):
                    alert_message += f"\n堆栈跟踪:\n```\n{context['stack_trace']}\n```"

            # 发送飞书消息
            success = send_feishu_message_with_creds(
                message_content=alert_message,
                app_id=self.app_id,
                app_secret=self.app_secret,
                chat_id=self.chat_id
            )

            if success:
                logger.info(f"Feishu alert sent successfully: {level} - {message[:50]}...")
            else:
                logger.error("Failed to send feishu alert")

            return success

        except Exception as e:
            logger.error(f"Error sending feishu alert: {e}")
            return False
