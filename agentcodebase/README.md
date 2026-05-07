# CodeBase for Agentic Researcher

## Introduction

高效，持续演进的代码结构化复用，并形成个性化的个人代码 CodeBase 是提升科研和工程效率开发的关键手段。本仓库将汇集本人在科研过程中不断抽象解耦出来的可重复使用的 Python 代码组件等等，也希望开源并且贡献至社区上。

## Usage

作为一个独立的文件夹（类似于 `utils` 的功能），可以作为一个 Python 包封装并进行调用。

```bash
# clone the project
git clone git@github.com:anonymous-org/AgentCodeBase.git

# python environment
uv sync
source .venv/bin/activate
```

其内部的 `codebase` 文件夹是一个完整的 Python 可移植包。

### Config

所有的 codebase 组件都使用 YAML 文件作为读取配置文件的方式（支持注释，可读性高），并且目前所有的配置都集成到了 `config/config.yaml` 一个文件中。

```bash
cp config/config.example.yaml config/config.yaml
```

- [`FEISHU_URL`]("https://open.feishu.cn/community/articles/7271149634339422210")：飞书 Webhook URL 获取

## Utilities

### `APITester`

API 可用性测试工具

- 多模型并发测试
- 流式/非流式 API 检查
- 智能重试和错误处理
- 飞书报警集成
- 定时心跳监控
- 灵活的配置加载方式


```python
from codebase import APITester, StreamType

# 方式 1: 从 config.yaml 自动加载配置
tester = APITester()

# 方式 2: 直接传入 API Key 和 Base URL
tester = APITester(
    api_key="sk-proj-xxx",
    base_url="https://api.openai.com/v1"
)

# 方式 3: 传入多个配置
tester = APITester(
    api_configs=[
        ("sk-key1", "https://api.openai.com/v1"),
        ("sk-key2", "https://api.example.com/v1")
    ]
)

# 单次测试 - 使用字符串指定流式类型
results = tester.run_test(
    models=["gpt-4o", "gpt-4o-mini"],
    stream_type="base",  # "base", "stream", or "all"
    enable_feishu_alert=True
)

# 定时心跳测试
tester.start_periodic_test(
    interval_seconds=300,  # 每5分钟测试一次
    models=["gpt-4o"],
    stream_type="all",
    enable_feishu_alert=True
)
```

### `LLMClient`

异步 OpenAI API 客户端,支持速率限制、智能重试、请求计时和灵活的参数传递。

- 自动速率限制
- 智能重试机制(指数退避)
- 请求计时和性能监控
- 支持 **kwargs 传递任意 OpenAI API 参数
- 增强的错误处理和日志记录
- 异步并发支持

```python
import asyncio
from codebase import LLMClient, ModelConfig, RetryConfig

async def main():
    # 基础使用 - 从 config.yaml 加载配置
    client = LLMClient()

    # 调用 API
    response, metadata = await client.chat_completion("Hello, world!")

    print(f"Response: {response}")
    print(f"Duration: {metadata['duration_ms']}ms")
    print(f"Tokens: {metadata['usage']['total_tokens']}")
    print(f"Retries: {metadata['retry_count']}")

    # 使用 **kwargs 传递额外参数
    response, metadata = await client.chat_completion(
        "Write a poem about AI",
        system_prompt="You are a poet",
        temperature=0.9,
        max_tokens=2000,
        top_p=0.95,
        presence_penalty=0.5,
        frequency_penalty=0.3
    )

    # 自定义重试配置
    retry_config = RetryConfig(
        max_retries=5,
        retry_delay=2.0,
        backoff_factor=1.5
    )

    config = ModelConfig(
        model_name="gpt-4",
        api_key="sk-xxx",
        base_url="https://api.openai.com/v1",
        retry_config=retry_config
    )

    client = LLMClient(config=config)
    response, metadata = await client.chat_completion("Test")

asyncio.run(main())
```

### `LLMPipeline`

LLM 数据生成流水线,支持并发处理、结果提取和进度跟踪。

- 异步并发数据生成
- 自动结果提取(支持正则表达式)
- 实时保存到 JSONL 文件
- 进度条显示
- 灵活的配置管理
- 错误处理和重试


```python
from codebase import LLMPipeline

# 初始化 pipeline
pipeline = LLMPipeline()

# 准备输入数据
data_pool = [
    {
        "input": "Generate data 1",
        "user_prompt_kwargs": {"topic": "AI"},
        "system_prompt_kwargs": {"role": "assistant"}
    },
    {
        "input": "Generate data 2",
        "user_prompt_kwargs": {"topic": "ML"},
        "system_prompt_kwargs": {"role": "expert"}
    }
]

# 创建提取函数
extractor = pipeline.make_default_extractor("result")

# 运行 pipeline
results = pipeline.run(
    data_pool=data_pool,
    concurrency_limit=5,  # 并发数量
    extract_function=extractor
)

# 结果会自动保存到 output/{experiment_name}/result.jsonl
```

### `Matplotlib` and `Seaborn` Modules Support

`mplstyle` files borrowed from [Public Github](https://github.com/hosilva/mplstyle)

See detailed human documents and prompts for agents in:

- [Human Documents](./codebase/draw/README4HUMAN.md)
- [Prompts for Agents](./codebase/draw/README4AGENTS.md)


## Todo List

- [x] Add Demo usage for class `LLMClient`
- [x] Find and Optimize custon mplstyle file
- [x] Add detailed code and docs for image drawing
- [x] Add Config and system level support for `prompts` (or maybe skills...)
