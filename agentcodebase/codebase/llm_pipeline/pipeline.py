import json
import os
import re
import threading
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable
from tqdm import tqdm

from codebase.llm.llm_client import LLMClient, ModelConfig
from codebase._utils._config import get_config
from codebase._utils._logging import get_logger


class LLMPipeline:
    """
    Automated data generation pipeline for LLM-based data synthesis.

    This pipeline supports concurrent LLM API processing, result extraction,
    and progress tracking.
    """

    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize data generation pipeline.

        Args:
            config_path: Path to configuration file. If None, uses default config.yaml.
            **kwargs: Parameters to override configuration values.
        """
        self.logger = get_logger(__name__)
        self.config = get_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Override config with kwargs if provided
        if kwargs:
            self._update_config_with_kwargs(kwargs)

        # Setup output directory
        self.output_dir = self._setup_output_dir()

        # Get prompt paths
        self._load_prompt_paths()

        # Initialize file lock and results storage
        self.file_lock = threading.Lock()
        self.results = []

        self.logger.info(
            f"DataGenerationPipeline initialized: " f"output_dir={self.output_dir}"
        )

    def _update_config_with_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Update configuration with provided keyword arguments.

        Args:
            kwargs: Keyword arguments to override config values.
        """
        for key, value in kwargs.items():
            if value is not None:
                self.logger.info(f"Overriding config: {key}={value}")
                # Note: Config is immutable, so we store overrides separately
                if not hasattr(self, "_config_overrides"):
                    self._config_overrides = {}
                self._config_overrides[key] = value

    def _get_config_value(self, key_path: str, default=None) -> Any:
        """
        Get configuration value with optional override.

        Args:
            key_path: Dot-notation path to config value (e.g., 'llm_pipeline.model.model').
            default: Default value if not found.

        Returns:
            Configuration value or default.
        """
        # Check overrides first
        if hasattr(self, "_config_overrides"):
            if key_path in self._config_overrides:
                return self._config_overrides[key_path]

            # Try without prefix
            short_key = key_path.replace("llm_pipeline.", "")
            if short_key in self._config_overrides:
                return self._config_overrides[short_key]

        # Use config.get()
        return self.config.get(key_path, default)

    def _load_prompt_paths(self) -> None:
        """Load prompt file paths from configuration."""
        self.system_prompt_path = self._get_config_value(
            "llm_pipeline.prompts.system_prompt_path"
        )
        self.user_prompt_path = self._get_config_value(
            "llm_pipeline.prompts.user_prompt_path"
        )

    def _setup_output_dir(self) -> str:
        """
        Setup output directory for generated data.

        Returns:
            Path to output directory.
        """
        output_dir = self._get_config_value(
            "llm_pipeline.output_data.output_dir", "output"
        )
        experiment_name = self._get_config_value(
            "llm_pipeline.output_data.experiment_name", "experiment"
        )
        need_timestamp = self._get_config_value(
            "llm_pipeline.output_data.need_time_stamp", False
        )

        if need_timestamp:
            experiment_dir = os.path.join(output_dir, experiment_name, self.timestamp)
        else:
            experiment_dir = os.path.join(output_dir, experiment_name)

        self.experiment_dir = experiment_dir
        self.logger.info(f"Experiment directory: {experiment_dir}")

        self.experiment_path = os.path.join(experiment_dir, "result.jsonl")
        os.makedirs(experiment_dir, exist_ok=True)

        return experiment_dir

    def _get_model_config(self) -> ModelConfig:
        """
        Get model configuration from config file.

        Returns:
            ModelConfig object for LLM client.

        Raises:
            ValueError: If required model configuration is missing.
        """
        try:
            # Try to access nested config using attribute or get method
            if hasattr(self.config, "llm_pipeline"):
                pipeline_config = self.config.llm_pipeline
                model_config = pipeline_config.model

                api_key = model_config.api_key
                base_url = model_config.base_url
                model_name = model_config.model
                max_tokens = model_config.max_tokens
                temperature = model_config.temperature
                rate_limit = model_config.rate_limit
            else:
                api_key = self.config.get("llm_pipeline.model.api_key")
                base_url = self.config.get("llm_pipeline.model.base_url")
                model_name = self.config.get("llm_pipeline.model.model", "gpt-4o-mini")
                max_tokens = self.config.get("llm_pipeline.model.max_tokens", 5012)
                temperature = self.config.get("llm_pipeline.model.temperature", 0.7)
                rate_limit = self.config.get("llm_pipeline.model.rate_limit", 20)

            # Apply overrides if any
            api_key = self._get_config_value("llm_pipeline.model.api_key", api_key)
            base_url = self._get_config_value("llm_pipeline.model.base_url", base_url)
            model_name = self._get_config_value("llm_pipeline.model.model", model_name)
            max_tokens = self._get_config_value(
                "llm_pipeline.model.max_tokens", max_tokens
            )
            temperature = self._get_config_value(
                "llm_pipeline.model.temperature", temperature
            )
            rate_limit = self._get_config_value(
                "llm_pipeline.model.rate_limit", rate_limit
            )

            if not api_key or not base_url:
                raise ValueError("API key and base URL are required in configuration")

            return ModelConfig(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                temperature=temperature,
                rate_limit=rate_limit,
            )

        except Exception as e:
            raise ValueError(
                f"Failed to load model configuration: {e}. "
                "Please ensure config.yaml contains 'llm_pipeline.model' section "
                "with api_key, base_url, and model fields."
            )

    def _load_prompt_from_file(self, file_path: str, **format_variables) -> str:
        """
        Load prompt content from file.

        Args:
            file_path: Path to prompt file.
            **format_variables: Variables for string formatting.

        Returns:
            Prompt content as string.
        """
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                prompt = file.read().strip()
                try:
                    prompt = prompt.format(**format_variables)
                except KeyError as e:
                    self.logger.warning(
                        f"Prompt formatting error, missing key: {e}. Using original prompt."
                    )
                except Exception as e:
                    self.logger.error(f"Error formatting prompt: {e}")
                return prompt

        self.logger.error(f"Failed to load prompt file: {file_path}")
        return ""

    def save_result(self, result: Dict[str, Any]) -> None:
        """
        Save single generation result to jsonl file.

        Args:
            result: Single generation result dictionary.
        """
        with self.file_lock:
            with open(self.experiment_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
            self.results.append(result)

    def extract_all(self, pattern_name: str, text: str) -> List[str]:
        """
        Extract all contents of <pattern_name>...</pattern_name> tags from text.

        Supports multiline, attributes, and case-insensitive matching.

        Args:
            pattern_name: Name of the XML tag pattern.
            text: Text to extract from.

        Returns:
            List of extracted content (stripped).

        Examples:
            >>> extract_all("draft", "<draft>hello</draft>")
            ["hello"]
        """
        tag = re.escape(pattern_name.strip().lower())
        pattern = rf"<{tag}\b[^>]*>\s*(.*?)\s*</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches]

    def make_default_extractor(
        self, pattern_name: str
    ) -> Callable[[str, bool], Optional[Union[str, List[str]]]]:
        """
        Create a default extract function bound to a specific pattern.

        Args:
            pattern_name: Name of the pattern to extract.

        Returns:
            Extract function that takes response and returns list or single value.
        """
        logger = self.logger

        def extract(response: str, to_list: bool = True) -> Union[List[str], str, None]:
            """
            Extract pattern from response.

            Args:
                response: LLM response text.
                to_list: If True, return list; if False, return first match.

            Returns:
                Extracted content(s) or None/empty list if not found.
            """
            try:
                extract_list = self.extract_all(pattern_name, response)

                if not extract_list:
                    logger.warning(f"No <{pattern_name}> found in response.")
                    return [] if to_list else None

                return extract_list if to_list else extract_list[0]

            except Exception as e:
                logger.error(f"Extraction failed for <{pattern_name}>: {e}")
                return [] if to_list else None

        return extract

    async def run_single_queries(
        self,
        user_prompt: str,
        system_prompt: str,
        extract_function=None,
        input_data=None,
        client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        """
        Run a single query with LLM client.

        Args:
            user_prompt: User prompt for the model.
            system_prompt: System prompt for context.
            extract_function: Optional function to extract results.
            input_data: Input data dictionary.
            client: LLM client instance.

        Returns:
            Result dictionary with response and metadata.
        """
        client_to_use: LLMClient = client if client is not None else self.client
        response, naive_response = await client_to_use.safe_chat_completion(
            prompt=user_prompt, system_prompt=system_prompt
        )

        # Construct result dictionary
        result = {
            "input": input_data,
            "response": response,
            "naive_response": naive_response,
            "timestamp": datetime.now().isoformat(),
        }

        # Extract answer if extract function provided
        if extract_function:
            result["extracted"] = extract_function(response)

        # Save result continuously
        self.save_result(result)

        return result

    async def run_single_task(
        self,
        i: int,
        input_data: Dict[str, Any],
        extract_function: Optional[Callable] = None,
        client: Optional[LLMClient] = None,
    ) -> Dict[str, Any]:
        """
        Process a single task.

        Args:
            i: Task index.
            input_data: Input data dictionary.
            extract_function: Optional function to extract results.
            client: LLM client instance.

        Returns:
            Task result dictionary.
        """
        try:
            self.logger.debug(f"Processing task {i+1}")
            self.logger.debug(f"Input data: {input_data}")

            # Load system prompt
            self.logger.debug("Loading system prompt")
            system_prompt_kwargs = input_data.get("system_prompt_kwargs", {})
            system_prompt = self._load_prompt_from_file(
                self.system_prompt_path, **system_prompt_kwargs
            )
            self.logger.debug(f"System prompt: {system_prompt[:100]}...")

            # Load user prompt
            self.logger.debug("Loading user prompt")
            user_prompt_kwargs = input_data.get("user_prompt_kwargs", {})
            user_prompt = self._load_prompt_from_file(
                self.user_prompt_path, **user_prompt_kwargs
            )
            self.logger.debug(f"User prompt: {user_prompt[:100]}...")

            result = await self.run_single_queries(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                input_data=input_data,
                extract_function=extract_function,
                client=client,
            )
            self.logger.debug(f"Got result: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error processing task {i+1}: {e}")
            error_result = {
                "input": input_data,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self.save_result(error_result)
            return error_result

    async def run_all_tasks(
        self,
        data_pool: List[Dict[str, Any]],
        concurrency_limit: int = 5,
        extract_function: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all tasks with concurrency control.

        Args:
            data_pool: List of input data dictionaries.
            concurrency_limit: Maximum number of concurrent tasks.
            extract_function: Optional function to extract results.

        Returns:
            List of task results.
        """
        sem = asyncio.Semaphore(concurrency_limit)
        model_config = self._get_model_config()

        async def worker(i, input_data):
            """Worker function for concurrent task processing."""
            async with sem:
                client = LLMClient(config=model_config)
                return await self.run_single_task(
                    i, input_data, extract_function, client
                )

        tasks = [worker(i, x) for i, x in enumerate(data_pool)]
        results = []

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            results.append(await coro)

        return results

    def run(
        self,
        data_pool: List[Dict[str, Any]],
        concurrency_limit: int = 5,
        extract_function: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run data generation pipeline with concurrent processing.

        Args:
            data_pool: List of input data dictionaries.
            concurrency_limit: Maximum number of concurrent tasks (default: 5).
            extract_function: Optional function to extract results from responses.

        Returns:
            List of processed results.

        Example:
            >>> pipeline = DataGenerationPipeline()
            >>> data = [{"input": "test"}, {"input": "test2"}]
            >>> results = pipeline.run(data_pool=data, concurrency_limit=3)
        """
        self.logger.info("Starting data generation pipeline")
        self.logger.info(f"Concurrency limit: {concurrency_limit}")
        self.logger.info(f"Total tasks: {len(data_pool)}")

        results = asyncio.run(
            self.run_all_tasks(
                data_pool=data_pool,
                concurrency_limit=concurrency_limit,
                extract_function=extract_function,
            )
        )

        # Process any exception results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {i+1} failed with exception: {result}")
                processed_results.append(
                    {"error": f"Task failed with exception: {result}", "index": i}
                )
            else:
                processed_results.append(result)

        self.logger.info("Data generation pipeline completed")
        return processed_results
