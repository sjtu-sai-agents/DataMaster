import yaml
from pathlib import Path
from typing import Any, Optional
from threading import Lock


class Config:
    """
    Global configuration manager with singleton pattern.

    Loads configuration from YAML file and provides dot notation access.
    This class is designed to be shared across all application modules.
    """

    _instance: Optional['Config'] = None
    _lock = Lock()
    _config_data: dict = {}
    _config_path: Optional[Path] = None

    def __new__(cls, config_path: Optional[str] = None):
        """
        Create or return the singleton instance.

        Args:
            config_path: Path to the configuration YAML file.
                        Only used on first initialization.

        Returns:
            Config singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration by loading from YAML file.

        Args:
            config_path: Path to the configuration YAML file.
                        If None, uses default path: config/config.yaml

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file has invalid YAML syntax.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if config_path is None:
                project_root = Path(__file__).parent.parent.parent
                config_path = project_root / "config" / "config.yaml"
            else:
                config_path = Path(config_path)

            self._config_path = config_path
            self._load_config()
            self._initialized = True

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file has invalid YAML syntax.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse configuration file: {self._config_path}\nError: {e}"
            )

    def reload(self) -> None:
        """
        Reload configuration from file.

        This method is useful for picking up configuration changes
        without restarting the application.
        """
        with self._lock:
            self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'feishu.url').
            default: Default value if key doesn't exist.

        Returns:
            Configuration value or default if not found.

        Example:
            >>> config = Config()
            >>> feishu_url = config.get('feishu.url')
            >>> api_key = config.get('openai.api_key', 'default-key')
        """
        keys = key.split('.')
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_all(self) -> dict:
        """
        Get all configuration data.

        Returns:
            Dictionary containing all configuration values.
        """
        return self._config_data.copy()

    def __getattr__(self, name: str) -> Any:
        """
        Provide direct attribute access to top-level configuration keys.

        Args:
            name: Top-level configuration key.

        Returns:
            Configuration value or ConfigDict for nested access.

        Example:
            >>> config = Config()
            >>> config.feishu.url  # Access nested values
            >>> config.feishu  # Returns ConfigDict for feishu section
        """
        if name in self._config_data:
            value = self._config_data[name]
            return ConfigDict(value) if isinstance(value, dict) else value
        raise AttributeError(
            f"Configuration has no attribute '{name}'. "
            f"Available keys: {list(self._config_data.keys())}"
        )

    def __contains__(self, key: str) -> bool:
        """
        Check if configuration key exists.

        Args:
            key: Configuration key in dot notation.

        Returns:
            True if key exists, False otherwise.
        """
        return self.get(key) is not None

    def __repr__(self) -> str:
        """Return string representation of configuration."""
        return f"Config(path='{self._config_path}', keys={list(self._config_data.keys())})"


class ConfigDict:
    """
    Helper class for nested configuration access.

    Provides dict-like and attribute access to nested configuration sections.
    """

    def __init__(self, data: dict):
        """
        Initialize ConfigDict with configuration data.

        Args:
            data: Configuration dictionary for this section.
        """
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from this configuration section.

        Args:
            key: Key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            Configuration value or default.
        """
        value = self._data.get(key, default)
        return ConfigDict(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        """
        Provide attribute access to configuration values.

        Args:
            name: Configuration key.

        Returns:
            Configuration value or another ConfigDict for nested access.
        """
        if name in self._data:
            value = self._data[name]
            return ConfigDict(value) if isinstance(value, dict) else value
        raise AttributeError(
            f"Configuration section has no attribute '{name}'. "
            f"Available keys: {list(self._data.keys())}"
        )

    def __getitem__(self, key: str) -> Any:
        """Provide dictionary-style access."""
        value = self._data[key]
        return ConfigDict(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in this section."""
        return key in self._data

    def __repr__(self) -> str:
        """Return string representation of this section."""
        return f"ConfigDict(keys={list(self._data.keys())})"

    def to_dict(self) -> dict:
        """
        Convert ConfigDict to plain dictionary.

        Returns:
            Plain dictionary representation.
        """
        return self._data.copy()


# Global configuration instance
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global configuration instance.

    This is the main interface for accessing configuration throughout the application.

    Args:
        config_path: Path to configuration file. Only used on first call.

    Returns:
        Global Config instance.

    Example:
        >>> from codebase._utils._config import get_config
        >>> config = get_config()
        >>> feishu_url = config.feishu.url
        >>> api_key = config.get('openai.api_key')
    """
    global _global_config

    if _global_config is None:
        _global_config = Config(config_path)

    return _global_config
