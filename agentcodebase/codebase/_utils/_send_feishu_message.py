import requests
from typing import Optional

from codebase._utils._config import get_config
from codebase._utils._logging import get_logger


class FeishuBot:
    """
    Feishu bot client for sending messages via webhook.

    This class encapsulates all Feishu bot operations and provides
    a clean interface for sending notifications.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        keyword: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize Feishu bot with configuration.

        Args:
            webhook_url: Feishu bot webhook URL. If None, loads from config.
            keyword: Keyword required by Feishu bot. If None, loads from config.
            config_path: Path to configuration file. If None, uses default.

        Raises:
            ValueError: If webhook_url is not provided and not found in config.
        """
        self.logger = get_logger(__name__)

        if webhook_url is None or keyword is None:
            config = get_config(config_path)

            if hasattr(config, 'feishu'):
                webhook_url = webhook_url or config.feishu.url
                keyword = keyword or config.feishu.keyword
            else:
                webhook_url = webhook_url or config.get('feishu.url')
                keyword = keyword or config.get('feishu.keyword')

        if not webhook_url:
            raise ValueError(
                "Feishu webhook URL not provided. "
                "Please provide webhook_url parameter or configure it in config.yaml under 'feishu.url'"
            )

        self.webhook_url = webhook_url
        self.keyword = keyword
        self.logger.info(
            f"FeishuBot initialized with webhook URL: {webhook_url[:30]}..."
        )

    def send_message(
        self,
        message: str,
        add_keyword: bool = True,
        timeout: int = 10
    ) -> bool:
        """
        Send a text message to Feishu bot webhook.

        Args:
            message: The text message to send.
            add_keyword: Whether to automatically add keyword to message.
            timeout: Request timeout in seconds.

        Returns:
            True if message was sent successfully, False otherwise.

        Example:
            >>> bot = FeishuBot()
            >>> bot.send_message("API test failed")
            True
        """
        if add_keyword and self.keyword and self.keyword not in message:
            formatted_message = f"【{self.keyword}】{message}"
        else:
            formatted_message = message

        headers = {"Content-Type": "application/json"}
        payload = {
            "msg_type": "text",
            "content": {"text": formatted_message}
        }

        self.logger.info(f"Sending message to Feishu: '{message[:100]}...'")

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()

            response_data = response.json()
            if response_data.get("code") == 0 or response_data.get("StatusCode") == 0:
                self.logger.info("Message sent to Feishu successfully")
                return True
            else:
                self.logger.error(
                    f"Failed to send message to Feishu. Response: {response_data}"
                )
                return False

        except requests.exceptions.Timeout:
            self.logger.error(
                f"Timeout while sending message to Feishu (timeout={timeout}s)"
            )
            return False

        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Network error occurred while sending message to Feishu: {e}"
            )
            return False

    def send_alert(
        self,
        title: str,
        content: str,
        add_keyword: bool = True
    ) -> bool:
        """
        Send an alert message with title and formatted content.

        Args:
            title: Alert title.
            content: Alert content details.
            add_keyword: Whether to automatically add keyword to message.

        Returns:
            True if alert was sent successfully, False otherwise.

        Example:
            >>> bot = FeishuBot()
            >>> bot.send_alert("API Error", "Connection timeout")
            True
        """
        message = f"⚠️ {title}\n\n{content}"
        return self.send_message(message, add_keyword=add_keyword)


# Global bot instance for backward compatibility
_global_bot: Optional[FeishuBot] = None


def get_feishu_bot(config_path: Optional[str] = None) -> FeishuBot:
    """
    Get or create the global Feishu bot instance.

    Args:
        config_path: Path to configuration file.

    Returns:
        FeishuBot instance.

    Example:
        >>> from codebase._utils._send_feishu_message import get_feishu_bot
        >>> bot = get_feishu_bot()
        >>> bot.send_message("Test message")
    """
    global _global_bot

    if _global_bot is None:
        _global_bot = FeishuBot(config_path=config_path)

    return _global_bot


def send_feishu_message(
    message: str,
    add_keyword: bool = True,
    config_path: Optional[str] = None
) -> bool:
    """
    Send a text message to Feishu bot (convenience function).

    This is a simplified interface for quick message sending.
    For more control, use FeishuBot class directly.

    Args:
        message: The text message to send.
        add_keyword: Whether to automatically add keyword to message.
        config_path: Path to configuration file.

    Returns:
        True if message was sent successfully, False otherwise.

    Example:
        >>> from codebase._utils._send_feishu_message import send_feishu_message
        >>> send_feishu_message("API test completed successfully")
        True
    """
    bot = get_feishu_bot(config_path)
    return bot.send_message(message, add_keyword=add_keyword)
