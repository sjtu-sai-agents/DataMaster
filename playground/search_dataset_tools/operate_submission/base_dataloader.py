import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface that all data loaders must implement.
    Subclasses only need to implement:
    - setup(): Load and process data
    - describe(): Provide a description

    Data splitting is NOT handled here - it's the responsibility of the training code.

    Attributes:
        config: Additional keyword arguments for customization
        train_data: Processed training data (set by setup())
        test_data: Processed test data (set by setup())
    """

    def __init__(self, **kwargs):
        """
        Initialize the BaseDataLoader.

        Args:
            **kwargs: Additional configuration (accessible via self.config)
        """
        self.config = kwargs
        self.train_data = None
        self.test_data = None
        self._is_setup = False

    # ========================================================================
    # Abstract methods - must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def setup(self):
        """
        Setup the data loader.

        Subclasses must implement this method to:
        - Load data from disk (CSV, images, etc.)
        - Perform feature engineering
        - Define data augmentation strategies
        - Set self.train_data and self.test_data

        Note: Do NOT perform train/val split here. Splitting is handled by the training code.
        """
        raise NotImplementedError(
            "Methods should implement `setup` methods for specific data loader class!"
        )

    @abstractmethod
    def describe(self) -> str:
        """
        Return a description of this data loader.

        Should describe:
        - What task this loader is for
        - What data processing/augmentation tricks is applied
        - What new data source and datsets are added.
        - Any notable features of the implementation

        Returns:
            A string description of the data loader
        """
        raise NotImplementedError(
            "Methods should implement `describe` methods for specific data loader class!"
        )

    def get_data(self):
        """
        Return the processed training and test data.

        This is the main interface - returns the data after all
        preprocessing, feature engineering, and augmentation.

        Automatically calls setup() if not already called.
        """
        if not self._is_setup:
            logger.warning(
                "Using Base Methods get_data: load self.train_data & self.test_data"
            )
            self.setup()
            self._is_setup = True

        return self.train_data, self.test_data

    def __str__(self) -> str:
        return self.describe()