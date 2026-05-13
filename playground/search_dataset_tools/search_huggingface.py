#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 数据集搜索 MCP 工具

使用 FastMCP 封装的 HuggingFace 数据集操作工具集。
支持搜索、验证、获取样本、下载等功能。

环境变量:
    HF_TOKEN: HuggingFace API token (用于访问 gated 数据集)
    HF_DATASETS_CACHE: 数据集缓存目录 (默认: ~/.cache/huggingface/datasets)
    HF_ENDPOINT: HuggingFace 镜像端点 (如: https://hf-mirror.com)

官方文档参考:
    https://huggingface.co/docs/datasets/loading
"""

import os
import sys
import yaml
import logging
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP

# 配置日志 - 只输出到 stderr，避免污染 stdout（MCP 协议要求）
logging.basicConfig(
    level=logging.WARNING,  # 只显示警告和错误
    format='%(levelname)s - %(name)s - %(message)s',
    stream=sys.stderr  # 明确输出到 stderr
)
logger = logging.getLogger(__name__)

# 现在才导入 HuggingFace 库
from huggingface_hub import (
    list_datasets,
    dataset_info,
    hf_hub_download,
    scan_cache_dir,
    login,
    DatasetInfo,
    HfFileSystem,
)
from datasets import (
    load_dataset,
    load_dataset_builder,
    get_dataset_config_names,
    get_dataset_split_names,
)

mcp = FastMCP("huggingface-search")
_HF_SANDBOX_URL = os.getenv("HF_SANDBOX_URL", "http://localhost:8899").rstrip("/")


# ============================================================================
# Sandbox 代理 – 优先通过本地服务转发请求
# ============================================================================


def _sandbox_call(endpoint: str, payload: dict, timeout: int = 30) -> dict | None:
    """Call local sandbox API; return parsed JSON on success, None on failure (triggers fallback)."""
    if not _HF_SANDBOX_URL:
        return None
    try:
        import httpx
        resp = httpx.post(f"{_HF_SANDBOX_URL}{endpoint}", json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ============================================================================
# 辅助函数
# ============================================================================


def get_hf_client() -> None:
    """初始化 HF 客户端认证"""
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token)
        except Exception as e:
            logger.warning(f"HF token login failed: {e}")


def get_cached_datasets() -> List[Dict[str, Any]]:
    """使用官方 API 获取所有已缓存的数据集列表"""
    cached = []
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_type == "dataset":
                size_mb = round(repo.size_on_disk / (1024 * 1024), 2)
                cached.append(
                    {
                        "dataset_id": repo.repo_id,
                        "cache_path": repo.repo_path,
                        "size_mb": size_mb,
                    }
                )
    except Exception as e:
        logger.error(f"Loading cache error: {e}")
        return []
    return cached


def try_download_raw_file(dataset_id: str, output_dir: str) -> Optional[str]:
    """尝试从 HuggingFace Hub 直接下载原始文件。

    Args:
        dataset_id: 数据集 ID
        output_dir: 输出目录

    Returns:
        下载结果字符串，如果失败返回 None
    """
    try:
        # 使用 HfFileSystem 列出数据集仓库中的所有文件
        token = os.getenv("HF_TOKEN")
        fs = HfFileSystem(token=token)

        repo_path = f"datasets/{dataset_id}"
        try:
            all_files = list(fs.ls(repo_path, recursive=True))
        except Exception:
            return None

        # 下载匹配的文件
        results = []
        for file_path in all_files:
            if file_path["type"] == "file":
                file_name = file_path["name"]
                try:
                    local_path = hf_hub_download(
                        repo_id=dataset_id,
                        # * 替换绝对路径为相对路径
                        filename=file_name.replace(f"datasets/{dataset_id}/", ""),
                        repo_type="dataset",
                        token=token,
                        local_dir=output_dir,
                        local_dir_use_symlinks=False,
                    )
                    file_size_mb = round(os.path.getsize(local_path) / (1024 * 1024), 2)
                    results.append(
                        f"Raw file: {os.path.basename(local_path)} ({file_size_mb} MB) -> {local_path}"
                    )
                except Exception as e:
                    results.append(f"Failed to download {file_path}: {str(e)}")

        if results:
            return "\n".join(results)
        return None

    except Exception:
        return None


def _format_ds_basic(ds) -> str:
    """格式化数据集基本信息为单行"""
    return (
        f"{ds.id} | Author: {ds.author or 'unknown'} | "
        f"Downloads: {ds.downloads or 0:,} | Likes: {ds.likes or 0} | "
        f"URL: https://huggingface.co/datasets/{ds.id}"
    )


# ============================================================================
# MCP 工具函数
# ============================================================================


@mcp.tool()
def search_datasets(
    query: str,
    limit: int = 100,
    author: Optional[str] = None
):
    """
    Search for datasets on the HuggingFace Hub.

    This tool searches for datasets matching the given query string on the HuggingFace Hub.
    It returns a list of datasets with basic metadata including author, download count,
    likes, and direct URLs. Use this tool to discover datasets before inspecting or
    downloading them.

    IMPORTANT: HuggingFace search does NOT support semantic or fuzzy search. Please ensure
    your search query is accurate and specific for best results.

    Args:
        query (str): Search keyword(s) for datasets. Use specific terms for better results.
                     Examples: "sentiment analysis", "translation", "image classification".
        limit (int, optional): Maximum number of results to return. Default is 100.
                               Recommended range: 10-100 for optimal performance.
        author (str, optional): Filter results by dataset author or organization.
                                Examples: "stanfordnlp", "google", "openai".

    Returns:
        str: A formatted string containing search results with dataset information including
             dataset ID, author, download count, likes, and URL. Returns "No accessible
             datasets found" message if no results match the query.

    Example:
        >>> search_datasets("sentiment", limit=5, author="stanfordnlp")
        "Found 3 datasets for 'sentiment':\\n\\n1. stanfordnlp/sst2 | Author: ..."

    Note:
        - Gated datasets (requiring HF_TOKEN) are automatically excluded from results
        - Search is case-insensitive but requires exact keyword matching
        - For comprehensive dataset exploration, consider combining with inspect_dataset()
    """
    # ── sandbox 代理优先 ──
    sb = _sandbox_call("/search", {"query": query, "limit": limit, "author": author})
    if sb is not None:
        return sb.get("text", str(sb))

    # ── 回退到原有实现 ──
    try:
        search_params = {
            "search": f"{query} author:{author}" if author else query,
            "limit": limit,
        }
        results = [ds for ds in list_datasets(**search_params)]

        if not results:
            return f"No accessible datasets found for query: '{query}'"

        lines = [f"Found {len(results)} datasets for '{query}':", ""]
        lines.extend(f"{i}. {_format_ds_basic(ds)}" for i, ds in enumerate(results, 1))
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching datasets: {str(e)}"


@mcp.tool()
def inspect_dataset(
    dataset_id: str,
    config: Optional[str] = None
):
    """
    Retrieve comprehensive metadata and structural information for a HuggingFace dataset.

    This tool provides detailed information about a specific dataset without downloading
    the actual data. It combines metadata from the HuggingFace Hub with dataset structure
    information from the dataset builder. Use this tool to understand a dataset's content,
    structure, and accessibility before downloading or sampling.

    The returned information includes:
    - Basic metadata: dataset ID, author, creation/modification dates, access status
    - Statistics: download counts, likes, trending score
    - Accessibility: whether the dataset is gated (requires HF_TOKEN)
    - Data structure: features (columns and types), splits, sample counts
    - Documentation: description, citation, license, homepage
    - Available configurations and file listing

    Args:
        dataset_id (str): The unique identifier for the dataset on HuggingFace Hub.
                         Must be in the format 'organization/dataset-name' or 'user/dataset-name'.
                         Examples: "stanfordnlp/sst2", "imdb", "google/fleurs".
        config (str, optional): The configuration name to inspect. Many datasets have multiple
                               configurations representing different languages, versions, or subsets.
                               Use get_dataset_configs() to list available configurations.
                               If not specified, uses the default configuration.
                               Example: "en-US" for PolyAI/minds14, "main" for imdb.

    Returns:
        str: A YAML-formatted string containing comprehensive dataset metadata including:
             - Hub metadata: id, author, created_at, last_modified, private, disabled, gated,
               downloads, likes, tags, trending_score, files
             - Builder metadata: dataset_name, config_name, description, citation, homepage,
               license, features (with types), splits, configs
             Returns an error message if the dataset is not found or inaccessible.

    Example:
        >>> inspect_dataset("imdb", config="main")
        id: imdb
        author: openai
        created_at: '2021-08-20T20:05:47.000Z'
        ...

    Note:
        - For gated datasets, set HF_TOKEN environment variable before calling this tool
        - The 'config' parameter is required for datasets with multiple configurations
        - Features are displayed as string representations of their types
        - Split information includes actual data when loaded for inspection
    """
    # ── sandbox 代理优先 ──
    sb = _sandbox_call("/inspect", {"dataset_id": dataset_id, "config": config})
    if sb is not None:
        return sb.get("text", str(sb))

    # ── 回退到原有实现 ──
    error_msg = ""
    try:
        get_hf_client()
        hub_info: DatasetInfo = dataset_info(dataset_id)
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Error accessing dataset '{dataset_id}': {error_msg}")
        hub_info = None

    try:
        builder = load_dataset_builder(dataset_id, name=config)
        info = builder.info
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Error accessing dataset '{dataset_id}': {error_msg}")
        builder = None

    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as e:
        logger.error(f"Error listing configs for '{dataset_id}': {str(e)}")
        configs = []

    hub_metadata = (
        {
            "id": hub_info.id,
            "author": hub_info.author,
            "created_at": (
                hub_info.created_at.isoformat() if hub_info.created_at else None
            ),
            "last_modified": (
                hub_info.last_modified.isoformat() if hub_info.last_modified else None
            ),
            "private": hub_info.private,
            "disabled": hub_info.disabled,
            "gated": hub_info.gated,
            "downloads": hub_info.downloads,
            "downloads_all_time": hub_info.downloads_all_time,
            "likes": hub_info.likes,
            "tags": hub_info.tags,
            "paperswithcode_id": hub_info.paperswithcode_id,
            "trending_score": hub_info.trending_score,
            "files": [s.rfilename for s in hub_info.siblings],
        }
        if hub_info
        else {}
    )

    builder_info = (
        {
            "dataset_name": builder.dataset_name,
            "config_name": builder.name,
            "cache_dir": str(builder.cache_dir),
            "base_path": str(builder.base_path),
            "config_kwargs": builder.config_kwargs,
            "description": info.description,
            "citation": info.citation,
            "homepage": info.homepage,
            "license": info.license,
            "features": {k: str(v) for k, v in info.features.items()},
            # "splits": str(load_dataset(dataset_id, config)),
            "splits": {k: str(v) for k, v in info.splits.items()} if info.splits else {},
            "configs": configs,
        }
        if builder
        else {}
    )

    flattened_dict = {**hub_metadata, **builder_info}

    yaml_string = yaml.dump(
        flattened_dict, allow_unicode=True, sort_keys=False, default_flow_style=False
    )

    return yaml_string


@mcp.tool()
def get_dataset_configs(dataset_id: str):
    """
    Retrieve all available configurations for a given dataset.

    Many datasets on HuggingFace Hub contain multiple configurations, also known as
    subsets or sub-datasets. Configurations typically represent different languages,
    versions, task variants, or data splits within the same dataset repository.

    Use this tool to discover available configurations before calling inspect_dataset(),
    get_dataset_splits(), or get_dataset_sample() with a specific configuration.

    Args:
        dataset_id (str): The unique identifier for the dataset on HuggingFace Hub.
                         Must be in the format 'organization/dataset-name' or 'user/dataset-name'.
                         Examples: "PolyAI/minds14", "google/fleurs", "mozilla-foundation/common_voice_16_1".

    Returns:
        List: A list of available configuration names as strings. For datasets with no
             multiple configurations, returns a message indicating the dataset uses
             the default configuration. Returns an error message if the dataset is
             not found or inaccessible.

    Example:
        >>> get_dataset_configs("PolyAI/minds14")
        ['cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR', 'it-IT', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN', 'all']

        >>> get_dataset_configs("stanfordnlp/sst2")
        "This dataset uses the default configuration (no multiple configs)."

    Note:
        - Configuration names are case-sensitive and must match exactly
        - Use the returned configuration name as the 'config' parameter in other tools
        - Some datasets require a config to be specified (will error otherwise)
        - The 'all' configuration (when available) typically combines all subsets
    """
    # ── sandbox 代理优先 ──
    sb = _sandbox_call("/configs", {"dataset_id": dataset_id})
    if sb is not None:
        text = sb.get("text", "")
        if text.startswith("["):
            import json as _json
            try:
                return _json.loads(text)
            except Exception:
                pass
        return text

    # ── 回退到原有实现 ──
    try:
        configs = get_dataset_config_names(dataset_id)
        if not configs:
            return "This dataset uses the default configuration (no multiple configs)."
        else:
            return configs
    except Exception as e:
        return f"Error listing configs for '{dataset_id}': {str(e)}"


@mcp.tool()
def get_dataset_splits(
    dataset_id: str,
    config: Optional[str] = None
):
    """
    Retrieve and display all available data splits for a specific dataset configuration.

    Splits are subsets of the dataset used for different purposes in machine learning,
    typically including 'train' for training, 'validation' or 'test' for evaluation.
    This tool loads the dataset and returns its structure showing available splits
    and their metadata.

    Use this tool to understand the split structure before sampling or downloading
    specific portions of the dataset.

    Args:
        dataset_id (str): The unique identifier for the dataset on HuggingFace Hub.
                         Must be in the format 'organization/dataset-name' or 'user/dataset-name'.
                         Examples: "stanfordnlp/sst2", "imdb", "imdb".
        config (str, optional): The configuration name to inspect. Required for datasets
                               with multiple configurations. Use get_dataset_configs()
                               to list available configurations.
                               Examples: "en-US" for PolyAI/minds14, "main" for imdb.
                               Defaults to the default configuration if not specified.

    Returns:
        str: A string representation of the dataset showing its structure including
             available splits, column names, and row counts. Returns an error message
             if the dataset is not found, inaccessible, or the configuration is invalid.

    Example:
        >>> get_dataset_splits("rotten_tomatoes")
        DatasetDict({
            train: Dataset({
                features: ['text', 'label'],
                num_rows: 8530
            })
            test: Dataset({
                features: ['text', 'label'],
                num_rows: 1066
            })
        })

    Note:
        - This tool downloads the dataset metadata but not the full data
        - The returned string shows split names, features, and row counts
        - For large datasets, this may take a few seconds to retrieve metadata
        - Use get_dataset_sample() to preview actual data from a specific split
    """
    # ── sandbox 代理优先 ──
    sb = _sandbox_call("/splits", {"dataset_id": dataset_id, "config": config})
    if sb is not None:
        return sb.get("text", str(sb))

    # ── 回退到原有实现 ──
    try:
        dataset = load_dataset(dataset_id, name=config)
        return str(dataset)
    except Exception as e:
        return f"Error listing splits for '{dataset_id}': {str(e)}"


@mcp.tool()
def get_dataset_readme(dataset_id: str):
    """
    Retrieve the README.md file or description for a HuggingFace dataset.

    This tool fetches the documentation for a dataset, which typically includes:
    - Dataset description and summary
    - Usage instructions and examples
    - Data source and collection methodology
    - Preprocessing steps and data statistics
    - Citation information and license details
    - Important notes and limitations

    Use this tool to understand how to properly use a dataset before downloading
    or processing it.

    Args:
        dataset_id (str): The unique identifier for the dataset on HuggingFace Hub.
                         Must be in the format 'organization/dataset-name' or 'user/dataset-name'.
                         Examples: "stanfordnlp/sst2", "imdb", "imdb".

    Returns:
        str: The full content of the dataset's README.md file, or the description
             from the dataset card if README.md is not available. Returns an error
             message if the dataset is not found or has no documentation.

    Example:
        >>> get_dataset_readme("imdb")
        # Example Dataset
        #
        # Example dataset README content...
        # [full README content]

    Note:
        - README files can be lengthy (up to several thousand words)
        - The content is returned in Markdown format
        - For datasets without README, attempts to retrieve description from card data
        - Use inspect_dataset() for structured metadata instead of documentation
    """
    # ── sandbox 代理优先 ──
    sb = _sandbox_call("/readme", {"dataset_id": dataset_id})
    if sb is not None:
        return sb.get("text", str(sb))

    # ── 回退到原有实现 ──
    try:
        # * 直接使用 hf_hub_download 进行下载
        try:
            readme_path = hf_hub_download(
                repo_id=dataset_id, filename="README.md", repo_type="dataset"
            )
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

            return content
        except Exception:
            pass

        try:
            # * 利用 dataset info 进行下载
            info = dataset_info(dataset_id)
            description = info.card_data.get("description", "") or info.card_data.get(
                "dataset_info", {}
            ).get("description", "")
            return description
        except Exception:
            pass

        return f"No README or description found for '{dataset_id}'"
    except Exception as e:
        return f"Error retrieving README for '{dataset_id}': {str(e)}"


@mcp.tool()
def get_dataset_sample(
    dataset_id: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    num_samples: int = 5,
):
    """
    Retrieve sample data records from a specific dataset and configuration.

    This tool fetches actual data samples from a dataset to preview the content,
    structure, and format. It displays column names, data types, and the specified
    number of sample records. Use this tool to understand the data format before
    downloading the full dataset or processing it.

    The tool automatically validates the requested split and falls back to the
    first available split if the specified split is not found.

    Args:
        dataset_id (str): The unique identifier for the dataset on HuggingFace Hub.
                         Must be in the format 'organization/dataset-name' or 'user/dataset-name'.
                         Examples: "stanfordnlp/sst2", "imdb", "imdb".
        config (str, optional): The configuration name to sample from. Required for datasets
                               with multiple configurations. Use get_dataset_configs()
                               to list available configurations.
                               Examples: "en-US" for PolyAI/minds14, "main" for imdb.
                               Defaults to the default configuration if not specified.
        split (str, optional): The specific data split to sample from, such as 'train',
                              'validation', or 'test'. Use get_dataset_splits() to see
                              available splits. If not specified or if the specified split
                              is not found, automatically uses the first available split.
        num_samples (int, optional): The number of sample records to retrieve and display.
                                    Default is 5. Maximum recommended is 100 to avoid
                                    excessive output. Each sample shows all columns.

    Returns:
        str: A formatted string containing:
             - Dataset ID and configuration used
             - Split name that was actually used
             - List of column names and their data types
             - The requested number of sample records with all field values
             Returns an error message if the dataset is inaccessible or has no data.

    Example:
        >>> get_dataset_sample("rotten_tomatoes", split="train", num_samples=2)
        Dataset Sample from 'cornell-movie-review-data/rotten_tomatoes'
        Config: default | Split: train
        Columns (2): text, label

        Column Types:
          text: str
          label: int

        [Sample 1]
          text: the rock is destined to be the 21st century's new " conan "...
          label: 1

        [Sample 2]
          text: gorgeously photographed and reported ...

    Note:
        - Complex data types (nested structures, images, audio) are converted to strings
        - If the specified split doesn't exist, the first available split is used
        - Long field values are displayed in full (no truncation)
        - For gated datasets, ensure HF_TOKEN is set before calling this tool
        - This tool downloads the actual data samples, not just metadata
    """
    # ── sandbox 代理优先 ──
    sb = _sandbox_call(
        "/sample",
        {"dataset_id": dataset_id, "config": config, "split": split, "num_samples": num_samples},
        timeout=120,
    )
    if sb is not None:
        return sb.get("text", str(sb))

    # ── 回退到原有实现 ──
    try:
        messages = []
        # get dataset split names
        dataset_split_names = get_dataset_split_names(dataset_id, config_name=config)
        if split and split in dataset_split_names:
            pass
        else:
            messages.append(f"Split: {split} not in {dataset_split_names}")
            split = dataset_split_names[0] if dataset_split_names else "unknown"
            logger.warning(f"Split: {split} not in {dataset_split_names}")

        dataset = load_dataset(dataset_id, name=config, split=split)
        samples = []
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            samples.append(
                {
                    k: (
                        v
                        if isinstance(v, (str, int, float, bool, list, dict))
                        or v is None
                        else str(v)
                    )
                    for k, v in example.items()
                }
            )

        if not samples:
            return f"No samples retrieved from '{dataset_id}' (split: '{split}')"

        columns = list(samples[0].keys())
        messages.extend(
            [
                f"Dataset Sample from '{dataset_id}'",
                f"Config: {config or 'default'} | Split: {split}",
                f"Columns ({len(columns)}): {', '.join(columns)}",
            ]
        )

        col_types = {k: type(v).__name__ for k, v in samples[0].items()}
        messages.append("\nColumn Types:")
        messages.extend(f"  {k}: {v}" for k, v in col_types.items())

        # 样本数据
        for i, sample in enumerate(samples, 1):
            messages.append(f"\n[Sample {i}]")
            for k, v in sample.items():
                messages.append(f"  {k}: {str(v)}")

        return "\n".join(messages)

    except Exception as e:
        error_msg = str(e)
        return f"Error calling tools: {error_msg}"


@mcp.tool()
def download_dataset(
    dataset_id: str,
    output_dir: str,
):
    """
    Download all raw files from a HuggingFace dataset repository to a local directory.

    This tool retrieves the complete file structure of a dataset repository from the
    HuggingFace Hub and downloads all files to the specified local directory. Unlike
    dataset conversion tools, this preserves the original file format and structure
    as stored on the Hub.

    The downloaded files may include data files (parquet, csv, json, txt), metadata
    files, configuration files, documentation, and any other assets associated with
    the dataset repository.

    Args:
        dataset_id (str): The unique identifier for the dataset on HuggingFace Hub.
                         Must be in the format 'organization/dataset-name' or 'user/dataset-name'.
                         Examples: "glue", "scq000/my-dataset", "mozilla-foundation/common_voice_16_1".
        output_dir (str): Local directory path where dataset files will be saved.
                                   The directory will be created if it doesn't exist.

    Returns:
        str: A summary string containing:
             - Dataset ID being downloaded
             - Absolute path to the output directory
             - List of downloaded files with their sizes in MB
             - HuggingFace Hub URL for the dataset
             Returns "Download Error" if the download fails completely.

    Example:
        >>> download_dataset("glue", "./data")
        Raw File Download: 'glue'
        Output: /home/user/projects/data
        --- Downloaded Files ---
        Raw file: dataset_infos.json (0.05 MB) -> /home/user/projects/data/dataset_infos.json
        Raw file: glue.py (0.02 MB) -> /home/user/projects/data/glue.py
        ...

    Note:
        - This tool downloads ALL files from the dataset repository, not just data files
        - Files are downloaded with their original directory structure preserved
        - For large datasets, this may take significant time and disk space
        - Gated datasets require HF_TOKEN environment variable to be set
        - Existing files in the output directory may be overwritten
        - Download progress is not displayed; use output path to verify completion
    """
    # ── sandbox 代理优先 ──
    sb = _sandbox_call(
        "/download",
        {"dataset_id": dataset_id, "output_dir": output_dir},
        timeout=600,
    )
    if sb is not None:
        return sb.get("text", str(sb))

    # ── 回退到原有实现 ──
    raw_download_result = try_download_raw_file(dataset_id, output_dir)
    if raw_download_result:
        summary = [
            f"Raw File Download: '{dataset_id}'",
            f"Output: {os.path.abspath(output_dir)}",
            "--- Downloaded Files ---",
            raw_download_result,
            f"\n{'='*60}\nURL: https://huggingface.co/datasets/{dataset_id}",
        ]
        return "\n".join(summary)
    else:
        return "Download Error"


if __name__ == "__main__":
    mcp.run()
