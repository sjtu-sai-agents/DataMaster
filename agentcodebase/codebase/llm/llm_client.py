import asyncio
import time
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from openai import AsyncOpenAI

from codebase._utils._config import get_config
from codebase._utils._logging import get_logger


@dataclass
class RetryConfig:
    """
    Configuration for retry mechanism.

    Args:
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries in seconds.
        backoff_factor: Multiplier for exponential backoff.
        retry_on_timeout: Whether to retry on timeout errors.
        retry_on_rate_limit: Whether to retry on rate limit errors.
        retry_on_server_error: Whether to retry on 5xx server errors.
    """
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if request should be retried based on error type and attempt count.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (0-indexed).

        Returns:
            True if should retry, False otherwise.
        """
        if attempt >= self.max_retries:
            return False

        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Check for timeout errors
        if self.retry_on_timeout and 'timeout' in error_msg:
            return True

        # Check for rate limit errors
        if self.retry_on_rate_limit and 'rate_limit' in error_msg:
            return True

        # Check for server errors (5xx)
        if self.retry_on_server_error:
            if '5' in error_msg or 'server error' in error_msg:
                return True

        return False

    def get_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        return self.retry_delay * (self.backoff_factor ** attempt)


@dataclass
class ModelConfig:
    """
    Configuration for LLM model.

    Args:
        model_name: Name of the model to use.
        api_key: API key for authentication.
        base_url: Base URL for API endpoint.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        rate_limit: Maximum requests per minute.
        retry_config: Retry configuration.
    """
    model_name: str
    api_key: str
    base_url: str
    max_tokens: int = 5042
    temperature: float = 1.0
    rate_limit: int = 200
    retry_config: RetryConfig = field(default_factory=RetryConfig)


class RateLimiter:
    """
    Rate limiter for controlling API request frequency.

    Implements token bucket algorithm to prevent API rate limit violations.

    Args:
        max_per_minute: Maximum number of requests allowed per minute.
    """

    def __init__(self, max_per_minute: int):
        """
        Initialize rate limiter.

        Args:
            max_per_minute: Maximum requests per minute.
        """
        self.interval = 60.0 / max_per_minute
        self.lock = asyncio.Lock()
        self.last = 0

    async def acquire(self):
        """
        Acquire a token, waiting if necessary to respect rate limit.

        This method ensures that requests are spaced out to respect the
        configured rate limit.
        """
        async with self.lock:
            now = time.monotonic()
            wait_time = self.interval - (now - self.last)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last = time.monotonic()


class LLMClient:
    """
    Asynchronous OpenAI API client with advanced features.

    Features:
    - Automatic rate limiting
    - Intelligent retry mechanism with exponential backoff
    - Request timing and metadata tracking
    - Flexible parameter passing via **kwargs
    - Comprehensive error handling

    Example:
        >>> client = LLMClient()
        >>> response, metadata = await client.chat_completion("Hello!")
        >>> print(f"Response: {response}")
        >>> print(f"Duration: {metadata['duration_ms']}ms")
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize LLM client with configuration.

        Args:
            config: ModelConfig object with model settings.
                   If None, loads from config file.
            config_path: Path to configuration YAML file.
                        If None, uses default path.

        Raises:
            ValueError: If API key is not provided or found in config.
        """
        self.logger = get_logger(__name__)

        if config is None:
            config = self._load_config_from_file(config_path)

        self._validate_config(config)
        self.config = config

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        self.rate_limiter = RateLimiter(max_per_minute=config.rate_limit)
        self.logger.info(
            f"LLMClient initialized: model={config.model_name}, "
            f"base_url={config.base_url}, "
            f"max_retries={config.retry_config.max_retries}"
        )

    def _load_config_from_file(self, config_path: Optional[str]) -> ModelConfig:
        """
        Load model configuration from config file.

        Args:
            config_path: Path to configuration file.

        Returns:
            ModelConfig object.

        Raises:
            ValueError: If configuration is invalid or missing.
        """
        try:
            config = get_config(config_path)

            if hasattr(config, 'openai'):
                openai_config = config.openai
                api_key = openai_config.api_key
                base_url = openai_config.base_url
                model_name = openai_config.models[0] if openai_config.models else "gpt-4"
            else:
                api_key = config.get('openai.api_key')
                base_url = config.get('openai.base_url')
                model_name = config.get('openai.models', ['gpt-4'])[0]

            max_tokens = config.get('openai.max_tokens', 5042)
            temperature = config.get('openai.temperature', 1.0)
            rate_limit = config.get('openai.rate_limit', 200)

            return ModelConfig(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                rate_limit=rate_limit
            )

        except Exception as e:
            raise ValueError(
                f"Failed to load configuration: {e}. "
                "Please ensure config.yaml contains valid 'openai' section."
            )

    def _validate_config(self, config: ModelConfig):
        """
        Validate model configuration.

        Args:
            config: ModelConfig to validate.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not config.api_key:
            raise ValueError("API key is required")

        if not config.base_url:
            raise ValueError("Base URL is required")

        if not config.model_name:
            raise ValueError("Model name is required")

    async def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Call OpenAI chat completion API with retry mechanism and timing.

        Args:
            prompt: User prompt to send to the model.
            system_prompt: Optional system prompt for context.
            **kwargs: Additional parameters to pass to API call.
                Common parameters include:
                - model: Override model name
                - temperature: Override sampling temperature
                - max_tokens: Override max tokens
                - top_p: Nucleus sampling parameter
                - n: Number of completions to generate
                - stream: Whether to stream responses
                - stop: Sequences where generation stops
                - presence_penalty: Penalty for new topics
                - frequency_penalty: Penalty for repetition

        Returns:
            Tuple of (response_content, metadata) if successful, None if failed.
            Metadata includes:
                - duration_ms: Request duration in milliseconds
                - retry_count: Number of retry attempts
                - timestamp: Request timestamp
                - api_response: Full API response
                - usage: Token usage statistics

        Example:
            >>> result = await client.chat_completion(
            ...     "Explain Python",
            ...     temperature=0.7,
            ...     max_tokens=1000
            ... )
            >>> if result:
            ...     response, metadata = result
            ...     print(f"Duration: {metadata['duration_ms']}ms")
        """
        retry_config = self.config.retry_config
        last_error = None

        for attempt in range(retry_config.max_retries + 1):
            await self.rate_limiter.acquire()

            start_time = time.monotonic()
            timestamp = datetime.now().isoformat()

            try:
                messages: List[Dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                self.logger.debug(
                    f"Sending request to {self.config.model_name} "
                    f"(attempt {attempt + 1}/{retry_config.max_retries + 1}): "
                    f"prompt_length={len(prompt)}"
                )

                # Prepare API parameters with defaults and overrides
                api_params = {
                    "model": kwargs.get("model", self.config.model_name),
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                }

                # Add any additional kwargs
                for key, value in kwargs.items():
                    if key not in ["model", "temperature", "max_tokens"]:
                        api_params[key] = value

                response = await self.client.chat.completions.create(**api_params)

                content = response.choices[0].message.content
                api_response = response.model_dump()

                # Calculate duration
                duration_ms = (time.monotonic() - start_time) * 1000

                # Build enhanced metadata
                metadata = {
                    "duration_ms": round(duration_ms, 2),
                    "retry_count": attempt,
                    "timestamp": timestamp,
                    "api_response": api_response,
                    "usage": {
                        "prompt_tokens": api_response.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": api_response.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": api_response.get("usage", {}).get("total_tokens", 0),
                    },
                    "model": api_response.get("model", self.config.model_name),
                    "finish_reason": api_response.get("choices", [{}])[0].get("finish_reason", "unknown"),
                }

                self.logger.info(
                    f"Request completed successfully: "
                    f"duration={duration_ms:.2f}ms, "
                    f"tokens={metadata['usage']['total_tokens']}, "
                    f"attempts={attempt + 1}"
                )

                return content, metadata

            except Exception as e:
                last_error = e
                duration_ms = (time.monotonic() - start_time) * 1000

                self.logger.warning(
                    f"API call failed (attempt {attempt + 1}): {type(e).__name__}: {str(e)}"
                )

                # Check if we should retry
                if retry_config.should_retry(e, attempt):
                    delay = retry_config.get_retry_delay(attempt)
                    self.logger.info(
                        f"Retrying in {delay:.2f}s... "
                        f"(attempt {attempt + 1}/{retry_config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # No more retries or non retry error
                    break

        # All retries exhausted
        self.logger.error(
            f"API call failed after {retry_config.max_retries + 1} attempts: {last_error}"
        )
        return None

    async def safe_chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        timeout: int = 3600,
        **kwargs
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Call chat completion API with timeout protection.

        This method wraps chat_completion with asyncio.wait_for to prevent
        indefinite hanging on API issues.

        Args:
            prompt: User prompt to send to the model.
            system_prompt: Optional system prompt for context.
            timeout: Timeout in seconds (default: 3600).
            **kwargs: Additional parameters to pass to API call.

        Returns:
            Tuple of (response_content, metadata) if successful, None if failed or timed out.

        Example:
            >>> result = await client.safe_chat_completion(
            ...     "Hello",
            ...     timeout=60,
            ...     temperature=0.8
            ... )
            >>> if result:
            ...     response, metadata = result
        """
        try:
            return await asyncio.wait_for(
                self.chat_completion(prompt, system_prompt, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Request timeout after {timeout}s: {prompt[:50]}..."
            )
            return None
        except Exception as e:
            self.logger.error(f"Safe API call failed: {e}")
            return None
