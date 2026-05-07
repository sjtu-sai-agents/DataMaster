"""
LLM client module for language model interactions.

This module provides utilities for working with language models including:
- LLM client configuration
- Logging setup for LLM operations

Example usage:
    >>> from codebase.llm import LLMClient
    >>>
    >>> client = LLMClient(model="gpt-4")
    >>> response = client.generate("Hello, world!")
"""

from codebase.llm.llm_client import LLMClient

__all__ = [
    'LLMClient',
]
