代码采用**分离模式**结构，将**算法部分**和**数据部分相互分离**，执行时会按以下顺序自动拼装：

1. **base_dataloader.py** - BaseDataLoader 抽象类（系统提供）
2. **"\n\n"**
3. **code_{node_id}_dataloader.py** - MyDataLoader 类
4. **code_{node_id}_template.py** - 训练脚本

```
执行时的完整代码 = base_dataloader + "\n\n" + 你的 DataLoader + template
```

### `MyDataLoader`

在 `code_{node_id}_dataloader.py` 中：

```python
# 注意：不需要 import 语句！会自动拼装

# 你可以定义任意辅助类
class FeatureExtractor:
    def extract(self, text):
        return ...

# 必须有一个 MyDataLoader 类，名称必须一致
class MyDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 你的初始化代码

    def setup(self):
        """
        加载数据、特征工程、数据增强等
        必须设置 self.train_data 和 self.test_data
        """
        self.train_data = ...
        self.test_data = ...

    def describe(self) -> str:
        """
        返回你的数据处理方式描述
        """
        return "描述你的数据处理方式"
```

### 重要约束

1. **类名必须叫 `MyDataLoader`**，不能改
2. **必须继承自 `BaseDataLoader`**
3. **不需要写 import 语句** - 会自动拼装
4. 你可以写任意的辅助函数和辅助类
5. 系统只会导入和使用 `MyDataLoader` 类
6. **训练逻辑不要在 MyDataLoader 中实现** - 这部分在 template 中

### BaseDataLoader 接口

```python
class BaseDataLoader(ABC):
    """
    数据加载器的抽象基类。
    
    子类必须实现：
    - setup(): 加载和处理数据
    - describe(): 提供描述
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.train_data = None
        self.test_data = None
        self._is_setup = False

    @abstractmethod
    def setup(self):
        """
        设置数据加载器。
        
        必须实现此方法来：
        - 从磁盘加载数据（CSV、图片等）
        - 执行特征工程
        - 定义数据增强策略
        - 设置 self.train_data 和 self.test_data
        """
        raise NotImplementedError("必须实现 setup 方法")
    
    @abstractmethod
    def describe(self) -> str:
        """
        返回数据加载器的描述。
        
        应该描述：
        - 这个加载器用于什么任务
        - 应用了什么数据处理/增强技巧
        - 添加了什么新数据源
        - 实现的任何显著特征
        """
        raise NotImplementedError("必须实现 describe 方法")
    
    def get_data(self):
        """
        返回处理后的训练和测试数据。
        
        这是主要接口 - 返回经过所有预处理、
        特征工程和增强后的数据。
        """
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        return self.train_data, self.test_data
```
