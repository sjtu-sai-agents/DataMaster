"""
Data generation pipeline module for automated data synthesis.

This module provides tools for:
- Automated data generation pipelines
- Concurrent LLM API processing
- Result extraction and saving
- Progress tracking and logging

Example usage:
    >>> from codebase.llm_pipeline import DataGenerationPipeline
    >>>
    >>> pipeline = DataGenerationPipeline(config_path="./config/data_generation.yaml")
    >>> results = pipeline.run(data_pool=[{"input": "test"}], concurrency_limit=5)
"""

from codebase.llm_pipeline.pipeline import LLMPipeline

__all__ = [
    "LLMPipeline",
]
