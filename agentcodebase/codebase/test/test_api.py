import os
import time
from enum import Enum
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from codebase._utils._config import get_config
from codebase._utils._logging import get_logger
from codebase._utils._send_feishu_message import send_feishu_message


class StreamType(Enum):
    """Enumeration for API stream check types."""
    NON_STREAM_ONLY = "non_stream_only"
    STREAM_ONLY = "stream_only"
    BOTH = "both"

    @classmethod
    def from_value(cls, value):
        """
        Convert string or enum value to StreamType enum.

        Supports multiple string aliases for convenience:
        - "base", "non_stream", "non_stream_only" → NON_STREAM_ONLY
        - "stream", "stream_only" → STREAM_ONLY
        - "all", "both" → BOTH

        Args:
            value: String, StreamType enum, or None.

        Returns:
            StreamType enum value.

        Raises:
            ValueError: If value cannot be converted.

        Examples:
            >>> StreamType.from_value("base")
            <StreamType.NON_STREAM_ONLY: 'non_stream_only'>
            >>> StreamType.from_value("all")
            <StreamType.BOTH: 'both'>
            >>> StreamType.from_value(StreamType.STREAM_ONLY)
            <StreamType.STREAM_ONLY: 'stream_only'>
        """
        if value is None:
            return cls.NON_STREAM_ONLY

        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            value_lower = value.lower().strip()

            mapping = {
                "base": cls.NON_STREAM_ONLY,
                "non_stream": cls.NON_STREAM_ONLY,
                "non_stream_only": cls.NON_STREAM_ONLY,
                "stream": cls.STREAM_ONLY,
                "stream_only": cls.STREAM_ONLY,
                "all": cls.BOTH,
                "both": cls.BOTH,
            }

            if value_lower in mapping:
                return mapping[value_lower]

            raise ValueError(
                f"Invalid stream_type value: '{value}'. "
                f"Supported values: {list(mapping.keys())}"
            )

        raise ValueError(
            f"Expected string or StreamType, got {type(value).__name__}"
        )


@dataclass
class APIConfig:
    """Configuration for API endpoint."""
    api_key: str
    base_url: str

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key cannot be empty")
        if not self.base_url:
            raise ValueError("Base URL cannot be empty")


class APITester:
    """
    API availability tester with support for single checks, periodic checks,
    stream testing, and Feishu alerting.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        api_configs: Optional[List[Tuple[str, str]]] = None,
        logger=None,
        config_path: Optional[str] = None
    ):
        """
        Initialize APITester with API configurations.

        Args:
            api_key: Optional API key. If provided along with base_url, overrides api_configs.
            base_url: Optional base URL. If provided along with api_key, overrides api_configs.
            api_configs: List of (api_key, base_url) tuples.
            logger: Custom logger instance. If None, creates default logger.
            config_path: Path to configuration file. If None, uses default path.

        Raises:
            ValueError: If no valid API configuration can be loaded.
        """
        self.logger = logger or get_logger(__name__)

        self.api_configs = self._load_api_configs(
            api_key, base_url, api_configs, config_path
        )
        self._is_running = False

    def _load_api_configs(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        api_configs: Optional[List[Tuple[str, str]]],
        config_path: Optional[str] = None
    ) -> List[APIConfig]:
        """
        Load API configurations with the following priority:

        1. If both api_key and base_url are provided: Create single config (overrides api_configs)
        2. If api_configs is provided: Use the list of configs
        3. Try loading from config file (config.yaml)
        4. Try loading from .env file
        5. Try loading from environment variables

        Args:
            api_key: Optional API key.
            base_url: Optional base URL.
            api_configs: List of (api_key, base_url) tuples.
            config_path: Path to configuration YAML file.

        Returns:
            List of APIConfig objects.

        Raises:
            ValueError: If no valid API configuration found.
        """
        # Priority 1: If both api_key and base_url are provided, override everything else
        if api_key and base_url:
            self.logger.info("Using provided api_key and base_url (overrides api_configs)")
            return [APIConfig(api_key, base_url)]

        # Priority 2: Use api_configs if provided
        if api_configs:
            self.logger.info(f"Using provided api_configs (count: {len(api_configs)})")
            return [APIConfig(key, url) for key, url in api_configs]

        # Priority 3: Try loading from config file
        try:
            config = get_config(config_path)

            if hasattr(config, 'openai'):
                openai_config = config.openai
                cfg_api_key = openai_config.api_key
                cfg_base_url = openai_config.base_url
            else:
                cfg_api_key = config.get('openai.api_key')
                cfg_base_url = config.get('openai.base_url')

            if cfg_api_key and cfg_base_url:
                self.logger.info("Loaded configuration from config.yaml")
                return [APIConfig(cfg_api_key, cfg_base_url)]

        except Exception as e:
            self.logger.debug(f"Could not load from config file: {e}")

        # Priority 4: Try loading from .env file
        env_path = Path.cwd() / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)
            self.logger.info(f"Loaded configuration from .env file: {env_path}")

        env_api_key = os.getenv('OPENAI_API_KEY')
        env_base_url = os.getenv('OPENAI_BASE_URL')

        if env_api_key and env_base_url:
            self.logger.info("Using configuration from environment variables")
            return [APIConfig(env_api_key, env_base_url)]

        # No configuration found
        raise ValueError(
            "No API configuration found. Please provide one of the following:\n"
            "1. Both api_key and base_url parameters\n"
            "2. api_configs parameter (list of tuples)\n"
            "3. Config file (config.yaml with openai.api_key and openai.base_url)\n"
            "4. .env file with OPENAI_API_KEY and OPENAI_BASE_URL\n"
            "5. Environment variables OPENAI_API_KEY and OPENAI_BASE_URL"
        )

    def test_single_api(
        self,
        config: APIConfig,
        model: str,
        message: str,
        stream_type: StreamType = StreamType.NON_STREAM_ONLY
    ) -> Tuple[bool, Optional[str]]:
        """
        Test a single API endpoint with specified model and message.

        Args:
            config: APIConfig object.
            model: Model name to test.
            message: Test message.
            stream_type: Type of stream check to perform.

        Returns:
            Tuple of (success: bool, error_message: Optional[str]).
            If test passed, returns (True, None).
            If test failed, returns (False, error_message).
        """
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)

        try:
            if stream_type in [StreamType.NON_STREAM_ONLY, StreamType.BOTH]:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    stream=False
                )
                self.logger.info(
                    f"Non-stream test passed - Model: {model}, "
                    f"Base URL: {config.base_url}, Response: {response.choices[0].message.content}"
                )

            if stream_type in [StreamType.STREAM_ONLY, StreamType.BOTH]:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        self.logger.debug(f"Stream chunk: {chunk.choices[0].delta.content}")

                self.logger.info(
                    f"Stream test passed - Model: {model}, Base URL: {config.base_url}"
                )

            return True, None

        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"API test failed - Model: {model}, Base URL: {config.base_url}, Error: {error_msg}"
            )
            return False, error_msg

    def run_test(
        self,
        models: Optional[Union[List[str], str]] = None,
        messages: Optional[Union[List[str], str]] = None,
        stream_type: Union[StreamType, str] = StreamType.NON_STREAM_ONLY,
        enable_feishu_alert: bool = False
    ) -> dict:
        """
        Run comprehensive API tests across all configurations and models.

        Args:
            models: List of model names to test. If None, loads from config or uses defaults.
            messages: Test message(s). If str, uses same message for all tests.
                     If list, must match number of configs * models or be single message.
            stream_type: Type of stream check to perform. Can be StreamType enum or string.
                        Supported string values: "base", "stream", "all" (or full names).
            enable_feishu_alert: Whether to send alerts to Feishu on failure.

        Returns:
            Dictionary with test results organized by config and model.
        """
        # Convert stream_type to enum if it's a string
        stream_type = StreamType.from_value(stream_type)
        # Load models from config if not provided
        if models is None:
            self.logger.warning("Error, Models are not set, using default model 'gpt-3.5-turbo' instead.")
            models = ["gpt-3.5-turbo"]
        if isinstance(models, str):
            models = [models]

        # Set default message
        if messages is None:
            messages = "Hello! Introduce who are you in English with approximately 50 words."
        if isinstance(messages, str):
            messages = [messages] * (len(self.api_configs) * len(models))

        self.logger.info(f"Starting API availability test with models: {models}")
        results = {}

        for config_idx, config in enumerate(self.api_configs):
            config_key = f"config_{config_idx + 1}"
            results[config_key] = {}

            for model_idx, model in enumerate(models):
                msg_idx = config_idx * len(models) + model_idx
                message = messages[msg_idx] if msg_idx < len(messages) else messages[0]

                self.logger.info(
                    f"Testing - Config: {config.base_url}, Model: {model}, Message: {message}"
                )

                passed, error_msg = self.test_single_api(config, model, message, stream_type)
                results[config_key][model] = {
                    "passed": passed,
                    "api_key": config.api_key[:10] + "...",
                    "base_url": config.base_url,
                    "error": error_msg
                }

                if not passed and enable_feishu_alert:
                    alert_msg = (
                        f"API Test Failed\n"
                        f"Base URL: {config.base_url}\n"
                        f"Model: {model}\n"
                        f"Message: {message}\n"
                        f"Error: {error_msg}"
                    )
                    send_feishu_message(alert_msg)

                    # end running for heartbeat test
                    if self._is_running:
                        self._is_running = False

        success_count = sum(
            1 for config_results in results.values()
            for model_result in config_results.values()
            if model_result["passed"]
        )
        total_count = sum(len(config_results) for config_results in results.values())

        self.logger.info(
            f"Test completed: {success_count}/{total_count} tests passed"
        )

        return results

    def start_periodic_test(
        self,
        interval_seconds: int = 60,
        models: Optional[Union[List[str], str]] = None,
        messages: Optional[Union[List[str], str]] = None,
        stream_type: Union[StreamType, str] = StreamType.NON_STREAM_ONLY,
        enable_feishu_alert: bool = False
    ):
        """
        Start periodic API testing (heartbeat check).

        Args:
            interval_seconds: Interval between tests in seconds.
            models: List of model names to test.
            messages: Test message(s).
            stream_type: Type of stream check to perform. Can be StreamType enum or string.
            enable_feishu_alert: Whether to send alerts to Feishu on failure.
        """
        # Convert stream_type to enum if it's a string
        stream_type = StreamType.from_value(stream_type)
        if self._is_running:
            self.logger.warning("Periodic test is already running")
            return

        self._is_running = True
        self.logger.info(f"Starting periodic test with interval: {interval_seconds}s")

        try:
            while self._is_running:
                self.run_test(models, messages, stream_type, enable_feishu_alert)
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Periodic test stopped by user")
            self._is_running = False

    def stop_periodic_test(self):
        """Stop periodic API testing."""
        self._is_running = False
        self.logger.info("Stopping periodic test...")
