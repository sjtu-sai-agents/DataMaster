"""
Test module for API availability testing.

This module provides tools for testing OpenAI API endpoints including:
- Single API availability checks
- Stream/non-stream API testing
- Periodic heartbeat monitoring
- Feishu alert integration

Example usage:
    >>> from codebase.test import APITester, StreamType
    >>>
    >>> tester = APITester()
    >>> results = tester.run_test(
    ...     messages="Hello World",
    ...     stream_type=StreamType.BOTH,
    ...     enable_feishu_alert=True
    ... )
"""

from codebase.test.test_api import APITester, StreamType, APIConfig

__all__ = [
    'APITester',
    'StreamType',
    'APIConfig',
]
