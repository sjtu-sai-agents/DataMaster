"""
EvoMaster监控模块

提供飞书告警、日志监控等功能
"""

from .feishu_alert_manager import FeishuAlertManager

# 全局单例
feishu_alert_manager = FeishuAlertManager()
