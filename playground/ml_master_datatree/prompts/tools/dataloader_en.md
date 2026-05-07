The code uses a **separation pattern** structure that separates the **algorithm component** and the **data component**. During execution, they are automatically assembled in the following order:

1. **base_dataloader.py** - BaseDataLoader abstract class (provided by the system)
2. **"\n\n"**
3. **code_{node_id}_dataloader.py** - MyDataLoader class
4. **code_{node_id}_template.py** - Training script

```
Complete code at runtime = base_dataloader + "\n\n" + Your DataLoader + template
```

### `MyDataLoader`

In `code_{node_id}_dataloader.py`:

```python
# Note: No import statements needed! They will be automatically assembled

# You can define any auxiliary classes
class FeatureExtractor:
    def extract(self, text):
        return ...

# Must have a MyDataLoader class, the name must be consistent
class MyDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Your initialization code

    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        self.train_data = ...
        self.test_data = ...

    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return "Describe your data processing approach"
```

### Important Constraints

1. **The class name must be `MyDataLoader`**, cannot be changed
2. **Must inherit from `BaseDataLoader`**
3. **No need to write import statements** - will be automatically assembled
4. You can write any auxiliary functions and auxiliary classes
5. The system will only import and use the `MyDataLoader` class
6. **Do not implement training logic in MyDataLoader** - this part is in the template

### BaseDataLoader Interface

```python
class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.

    Subclasses must implement:
    - setup(): Load and process data
    - describe(): Provide description
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.train_data = None
        self.test_data = None
        self._is_setup = False

    @abstractmethod
    def setup(self):
        """
        Set up the data loader.

        This method must be implemented to:
        - Load data from disk (CSV, images, etc.)
        - Perform feature engineering
        - Define data augmentation strategies
        - Set self.train_data and self.test_data
        """
        raise NotImplementedError("Must implement setup method")

    @abstractmethod
    def describe(self) -> str:
        """
        Return a description of the data loader.

        Should describe:
        - What task this loader is for
        - What data processing/augmentation techniques are applied
        - What new data sources are added
        - Any significant features implemented
        """
        raise NotImplementedError("Must implement describe method")

    def get_data(self):
        """
        Return the processed training and test data.

        This is the main interface - returns data that has undergone
        all preprocessing, feature engineering, and augmentation.
        """
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        return self.train_data, self.test_data
```

### CRITICAL: Fixed Validation Set Requirements

**You must use the pre-split validation set `input/val.csv`; random splitting is strictly prohibited!**

Within your `MyDataLoader.setup()`:

1.  **Check if `input/val.csv` exists**:
    ```python
    if os.path.exists('input/val.csv'):
        val_df = pd.read_csv('input/val.csv')
        # Remove val samples from train
        val_images = set(val_df['image'].values)
        train_df = train_full_df[~train_full_df['image'].isin(val_images)]
    ```

2.  **Prohibit the use of `train_test_split` for random partitioning**:
    * If `val.csv` exists, use it directly.
    * All nodes must be evaluated on the same validation set so that metrics accurately reflect improvements.

3.  **If the parent node's code already correctly uses `val.csv`**:
    * **Retain this logic.**
    * Only modify parts such as data augmentation, external data loading, or feature engineering.
    * Do not rewrite the entire `setup()` function.

4.  **If `input/val.csv` does not exist**:
    * You may only split `train/val` from the **original competition training set**.
    * **First**, split the original competition data into `X_train_orig, X_val, y_train_orig, y_val`.
    * **Then**, append external data to `X_train_orig` / `y_train_orig`.
    * The final `X_val` / `y_val` must contain **only** original competition data and no external samples.

5.  **Strictly Forbidden Implementation**:
    ```python
    X_combined = np.concatenate([X_orig, X_external])
    y_combined = np.concatenate([y_orig, y_external])
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, ...)
    ```
    The implementation above leaks external data into the validation set, making the validation metrics untrustworthy.