import logging
import sys
from pathlib import Path
from typing import Optional


class GlobalLogger:
    """
    Global logger class with singleton pattern for consistent logging across the application.
    """

    _instance: Optional['GlobalLogger'] = None
    _loggers: dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

    def setup_logger(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
        reset: bool = False
    ) -> logging.Logger:
        """
        Setup and configure a logger with the specified parameters.

        Args:
            name: Logger name, typically using __name__ from the calling module.
            level: Logging level (e.g., logging.INFO, logging.DEBUG).
            log_file: Optional path to log file. If None, logs only to console.
            format_string: Custom format string. If None, uses default format.
            reset: If True, remove existing handlers and reconfigure.

        Returns:
            Configured logger instance.
        """
        if name in self._loggers and not reset:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if reset:
            logger.handlers.clear()
        elif logger.handlers:
            return logger

        if format_string is None:
            format_string = (
                '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s'
            )

        formatter = logging.Formatter(format_string)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self._loggers[name] = logger
        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get an existing logger or create a default one.

        Args:
            name: Logger name.

        Returns:
            Logger instance.
        """
        if name in self._loggers:
            return self._loggers[name]

        return self.setup_logger(name)


# Global instance
_global_logger = GlobalLogger()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get or create a logger with the specified configuration.

    This is the main interface for obtaining loggers throughout the application.

    Args:
        name: Logger name, typically using __name__ from the calling module.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to log file.
        format_string: Custom format string.

    Returns:
        Configured logger instance.

    Example:
        >>> from codebase._utils._logging import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return _global_logger.setup_logger(name, level, log_file, format_string)
