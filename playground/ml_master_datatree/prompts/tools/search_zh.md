### Search Hints

- 你可以直接在 Huggingface, Google 上搜索可用的大规模数据源，并通过 HuggingFace 下载到本地
- 一些学术论文和学术信息网站也会包含一些 release 最新数据集的工作。你可以搜索一些 release 数据集的 Benchmark，例如调用 search_scholar 中的相关工具搜索对应论文，在阅读论文摘要中查看对应论文是否开源数据集，在拿到数据集地址 (例如 HuggingFace 的 `dataset_id`)之后，可以再利用 `search_web_google_search` 或者 search_huggingface 的工具进行下载
- web_parse 工具可以读取对应特定的网页，你也可以尝试使用！
- Github 上也会有一些比较高质量的数据集进行开源，你可以进行搜索，或者借助 `search_web_google_search`进行搜索，查看特定 Repo 的 README 页面。
你承担着重要的数据集搜集和发现的功能，你的重点任务就是从海量的信息源中找到**高质量的数据**，把他下载到本地的 {workspace}/data_links 文件夹下，你也可以使用 `execute_bash` 工具对数据进行清洗操作。

### HuggingFace Search

**dataset_id (数据集标识符)**
- 格式：`organization/dataset-name` 或 `user/dataset-name`
- 示例：`stanfordnlp/sst2`、`openai/gsm8k`、`google/fleurs`
- 这是数据集在 HuggingFace Hub 上的唯一标识符

**configs (配置/子集)**
- 许多数据集包含多个配置，也称为子集或子数据集
- 配置通常代表不同的语言、版本、任务变体或数据划分
- 示例：`PolyAI/minds14` 有 `cs-CZ`、`de-DE`、`en-US` 等语言配置
- 使用 `get_dataset_configs` 工具查看所有可用配置

**splits (数据划分)**
- 数据集的子集，用于机器学习的不同目的
- 常见划分：
  - `train`：训练集，用于模型训练
  - `validation`：验证集，用于模型调优和超参数选择
  - `test`：测试集，用于最终模型评估
- 使用 `get_dataset_splits` 工具查看特定配置的所有可用划分


1. **发现数据集**：使用 `search_huggingface_search_datasets` 搜索关键词,在 HuggingFace Hub 上搜索与查询匹配的数据集，HuggingFace 搜索**不支持**语义或模糊搜索，请确保搜索查询准确具体！
2. **了解结构**：使用 `search_huggingface_inspect_dataset` 查看元数据和结构信息,检索 HuggingFace 数据集的综合元数据和结构信息，不下载实际数据
3. **查看配置**：使用 `search_huggingface_get_dataset_configs` 获取指定数据集的所有可用配置（子集）列表
4. **查看划分**：使用 `search_huggingface_get_dataset_splits` 了解数据划分,检索并显示特定数据集配置的所有可用数据划分
5. **预览数据**：使用 `search_huggingface_get_dataset_sample` 查看样本数据,从指定数据集和配置中检索样本数据记录，用于预览数据内容
6. **阅读文档**：使用 `search_huggingface_get_dataset_readme` 查看完整文档,获取 HuggingFace 数据集的 README.md 文件或描述信息
7. **下载数据**：使用 `search_huggingface_download_dataset` 将 HuggingFace 数据集仓库中的所有原始文件下载到本地目录

### GitHub Search

1. `search_github_comprehensive_github_search`: 在多种 GitHub 实体类型（仓库、代码、用户、问题、PR）上执行综合搜索
2. `search_github_get_repository_readme`: 从 GitHub 仓库检索 README 内容
3. `search_github_search_code`: 在 GitHub 仓库中搜索特定的代码模式、函数或关键词
4. `search_github_search_issues`
5. `search_github_search_pull_requests`
6. `search_github_search_repositories`
7. `search_github_search_users`

### Web Search

**推荐工作流程**：
1. 使用 `search_web_google_search` 查找相关 URL
2. 使用 `search_web_web_parse` 获取特定 URL 的完整内容

1. `search_web_google_search`: 使用 Google 搜索执行广泛的通用网络搜索，支持任何主题和语言
2. `search_web_web_parse`: 从特定网页获取并提取完整的、干净的文本内容

  **推荐工作流程**：
  1. **不要**使用此函数进行一般搜索
  2. 首先调用 `google_search` 获取潜在 URL 列表
  3. 然后使用从 `google_search` 输出中检索的特定 URL 调用 `web_parse`，以获取完整文本


### Scholar Search

**重要提示**：搜索到 PDF URL 后，可以使用 `search_web_web_parse` 工具阅读 PDF 内容

1. `search_scholar_arxiv_search_by_author`: 根据作者姓名搜索 arXiv 论文
2. `search_scholar_arxiv_search_by_content`: 根据文章内容（标题、摘要或主题）搜索 arXiv 论文
3. `search_scholar_google_scholar_search`: 使用 Google Scholar 执行学术搜索

### Many more information domains

你可以利用多种信息源搜索互联网上的新的数据集

- 互联网上的公开数据集：你可以参考：
    - https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/refs/heads/master/README.rst
    - https://github.com/awesomedata/apd-core
    - https://github.com/awesomedata/awesome-public-datasets
    （你可以利用 bash 工具下载到 workspace 的 new_data 文件夹下并进行查看）
- 还有其他的互联网上的公开数据集，可以使用 Google Search 进行搜索得到

### Visualization Tools

当你成功下载数据集之后，你可以使用 `execute_bash` 等工具查看引入的外部数据的文字预览（对于一些纯文本、Tabular 等数据），如果涉及到多模态数据，你可以使用下面的两个工具进行多模态视觉的数据集预览：

> 注意！对于 CV 任务，你一定要仔细观察你所引入的外部数据是否和原来的训练数据集同分布！(例如图片风格，亮度，尺寸等等方面因素)，有时候盲目的引入外部数据反而会导致数据集的表现显著下降！

- `visual_inspect_images_details`: 此工具使用 OpenAI 的多模态模型来理解和分析图片内容，支持单张或多张图片的批量分析。

**适用场景:**
- 图片内容描述和识别
- 图片中的文字提取 (OCR)
- 图片质量评估
- 数据可视化图表解读
- 图片中的数据标注验证

Args:
    image_files: 图片文件路径，可以是单个文件路径（str）或文件路径列表（List[str]）
    query: 用户的问题或分析要求，例如："描述这张图片的内容"、"提取图片中的所有文字"等

Returns:
    JSON 字符串，包含 AI 模型的分析结果

Example:
    >>> inspect_images_details("photo.jpg", "描述这张图片的内容")
    >>> inspect_images_details(["img1.jpg", "img2.png"], "比较这两张图片的差异")

- `visual_inspect_images_info`:

[图片物理信息提取] 批量获取图片的物理属性信息

    提取图片的尺寸、亮度、对比度、文件大小等物理属性，并支持统计分析。
    适用于数据集质量检查、图片筛选、批量处理前的预分析等场景。

    **功能特性:**
    - 支持单文件或整个文件夹的批量处理
    - 提取尺寸、宽高比、通道数、亮度、对比度、文件大小
    - 批量模式下提供统计信息（平均值、中位数、方差等）
    - 支持 PNG、JPG、JPEG 格式

    Args:
        image_path: 图片文件路径或包含图片的文件夹路径

    Returns:
        JSON 字符串，包含每张图片的详细信息和批量统计信息
