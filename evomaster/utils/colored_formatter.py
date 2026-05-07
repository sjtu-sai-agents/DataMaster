"""彩色日志格式化器

为不同类型的日志消息添加终端颜色，提升日志可读性。
"""

import logging
import re


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器

    根据日志内容自动添加 ANSI 颜色代码：
    - Tool Call Start: 蓝色
    - Tool Call End: 绿色
    - Output/观察结果: 黄色
    - Arguments: 青色
    - Agent finished: 绿色加粗
    - Step [x/y]: 紫色
    - Error/Failed: 红色
    """

    # ANSI 颜色代码
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',

        # 前景色
        'BLACK': '\033[30m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'GRAY': '\033[90m',

        # 背景色
        'BG_RED': '\033[41m',
        'BG_GREEN': '\033[42m',
        'BG_YELLOW': '\033[43m',
    }

    # 日志内容匹配规则（按优先级排序）
    PATTERNS = [
        # Agent 状态
        (r'✅\s*Agent finished task', 'GREEN', True),  # 绿色加粗
        (r'⚠️.*Reached max turns limit', 'YELLOW', True),
        (r'❌.*Agent execution failed', 'RED', True),
        (r'📍\s*Step\s*\[.*?\]', 'MAGENTA', True),  # 紫色加粗

        # 工具调用
        (r'Tool Call Start:', 'BLUE', False),
        (r'Tool Call End:', 'GREEN', False),
        (r'Arguments:', 'CYAN', False),
        (r'Output:', 'YELLOW', False),

        # 特殊标记
        (r'Finish Tool Arguments:', 'GREEN', True),
        (r'\[Tool Call\]', 'BLUE', False),
        (r'\[Tool Output\]', 'GREEN', False),

        # 错误和警告（使用词边界 \b 避免误匹配）
        (r'\b(ERROR|Error|error|FAILED|Failed|failed)\b', 'RED', False),
        (r'\b(WARNING|Warning|warning)\b', 'YELLOW', False),

        # 文件路径
        (r'文件已成功保存至:', 'GREEN', False),

        # 分隔线（保持原色）
        (r'^=+$', None, False),  # 不着色
        (r'^-+$', None, False),  # 不着色
    ]

    def __init__(self, fmt=None, datefmt=None, style='%', enable_color=True):
        """初始化彩色格式化器

        Args:
            fmt: 日志格式字符串
            datefmt: 日期格式字符串
            style: 格式化风格 ('%', '{', '$')
            enable_color: 是否启用颜色（默认 True）
        """
        super().__init__(fmt, datefmt, style)
        self.enable_color = enable_color

    def format(self, record):
        """格式化日志记录，添加颜色

        Args:
            record: 日志记录对象

        Returns:
            格式化后的日志字符串（带颜色代码）
        """
        # 先调用父类格式化
        log_message = super().format(record)

        # 如果禁用颜色，直接返回
        if not self.enable_color:
            return log_message

        # 检查是否匹配任何模式
        for pattern, color, bold in self.PATTERNS:
            if re.search(pattern, log_message):
                # 如果不需要着色（None），跳过
                if color is None:
                    continue

                # 构建颜色代码
                color_code = self.COLORS.get(color, '')
                bold_code = self.COLORS['BOLD'] if bold else ''
                reset_code = self.COLORS['RESET']

                # 给整行添加颜色
                return f"{bold_code}{color_code}{log_message}{reset_code}"

        # 没有匹配任何模式，返回原始消息
        return log_message

    def formatException(self, exc_info):
        """格式化异常信息，使用红色

        Args:
            exc_info: 异常信息元组

        Returns:
            格式化后的异常字符串（红色）
        """
        result = super().formatException(exc_info)
        if self.enable_color:
            return f"{self.COLORS['RED']}{result}{self.COLORS['RESET']}"
        return result


def setup_colored_logging(logger=None, level=logging.INFO, fmt=None, enable_color=True):
    """为 logger 设置彩色输出

    Args:
        logger: Logger 对象（如果为 None，则使用 root logger）
        level: 日志级别
        fmt: 日志格式字符串（如果为 None，使用默认格式）
        enable_color: 是否启用颜色

    Returns:
        配置好的 logger
    """
    if logger is None:
        logger = logging.getLogger()

    # 默认格式
    if fmt is None:
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 创建彩色格式化器
    colored_formatter = ColoredFormatter(fmt, enable_color=enable_color)

    # 为所有现有的 handler 设置彩色格式化器
    for handler in logger.handlers:
        # 只为控制台 handler 设置彩色输出
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setFormatter(colored_formatter)
            handler.setLevel(level)

    return logger
