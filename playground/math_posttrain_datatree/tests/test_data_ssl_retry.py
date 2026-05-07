"""Test for SSL retry logic in data.py when fetching from datasets-server."""

from __future__ import annotations

import ssl
from unittest.mock import MagicMock, patch

import pytest
import requests

from playground.math_posttrain_datatree.core.utils.data import (
    _datasets_server_get,
    _materialize_via_datasets_server,
    DATASETS_SERVER_RETRIES,
)


class TestDatasetServerSSLRetry:
    """Test SSL error handling and retry logic for datasets-server requests."""

    def test_ssl_eof_error_retry_eventually_succeeds(self):
        """Test that SSL EOF errors trigger retries and eventually succeed."""
        # Simulate SSLEOFError for first 2 attempts, then succeed on 3rd
        mock_responses = [
            requests.exceptions.SSLError(
                ssl.SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)')
            ),
            requests.exceptions.SSLError(
                ssl.SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1010)')
            ),
            MagicMock(status_code=200, json=lambda: {"rows": []}),  # Success on 3rd attempt
        ]

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            response = mock_responses[call_count]
            call_count += 1
            if isinstance(response, Exception):
                raise response
            return response

        # Mock both the session getter and rate limiting
        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session') as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get = mock_get
            mock_session_getter.return_value = mock_session

            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                with patch('time.sleep'):  # Skip actual sleep in tests
                    resp = _datasets_server_get(
                        "rows",
                        params={"dataset": "test/dataset", "config": "default", "split": "train", "offset": 0, "length": 100},
                        timeout=30,
                        dataset_id="test/dataset",
                    )
                    assert resp.status_code == 200
                    assert call_count == 3  # Should have tried 3 times

    def test_ssl_eof_error_exhausts_retries(self):
        """Test that SSL EOF errors are retried the maximum number of times before failing."""
        # Fail on all retry attempts
        def mock_get_always_fail(*args, **kwargs):
            raise requests.exceptions.SSLError(
                ssl.SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol')
            )

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session') as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get = mock_get_always_fail
            mock_session_getter.return_value = mock_session

            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                with patch('time.sleep'):  # Skip actual sleep in tests
                    with pytest.raises(requests.exceptions.SSLError) as exc_info:
                        _datasets_server_get(
                            "rows",
                            params={"dataset": "test/dataset", "config": "default", "split": "train", "offset": 0, "length": 100},
                            timeout=30,
                            dataset_id="test/dataset",
                        )
                    assert "UNEXPECTED_EOF_WHILE_READING" in str(exc_info.value)

    def test_materialize_handles_ssl_errors_gracefully_during_pagination(self):
        """Test that materialization continues with partial results when SSL errors occur during pagination."""
        # Simulate success for first page, then SSL error on second page
        call_count = 0

        def mock_get_partial_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: splits lookup succeeds
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "splits": [
                            {"config": "default", "split": "train"}
                        ]
                    }
                )
            elif call_count == 2:
                # Second call: first page of rows succeeds
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "rows": [
                            {"row": {"problem": "Test problem 1", "solution": "Test solution 1", "answer": "42"}},
                            {"row": {"problem": "Test problem 2", "solution": "Test solution 2", "answer": "43"}},
                        ]
                    }
                )
            else:
                # Third call onwards: SSL error
                raise requests.exceptions.SSLError(
                    ssl.SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol')
                )

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session') as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get = mock_get_partial_success
            mock_session_getter.return_value = mock_session

            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                with patch('time.sleep'):  # Skip actual sleep in tests
                    entry = {
                        "source_id": "test/dataset",
                        "config": "default",
                        "split": "train",
                    }

                    rows = _materialize_via_datasets_server(
                        entry,
                        "test/dataset",
                        max_rows=500,
                    )

                    # Should return the rows from the first page, even though second page failed
                    assert len(rows) == 2
                    assert rows[0]["problem"] == "Test problem 1"
                    assert rows[1]["problem"] == "Test problem 2"

    def test_connection_error_also_triggers_retry(self):
        """Test that other connection errors (not just SSL) also trigger retries."""
        # Simulate ConnectionError for first attempt, then succeed on 2nd
        mock_responses = [
            requests.exceptions.ConnectionError("Connection refused"),
            MagicMock(status_code=200, json=lambda: {"rows": []}),
        ]

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            response = mock_responses[call_count]
            call_count += 1
            if isinstance(response, Exception):
                raise response
            return response

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session') as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get = mock_get
            mock_session_getter.return_value = mock_session

            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                with patch('time.sleep'):
                    resp = _datasets_server_get(
                        "rows",
                        params={"dataset": "test/dataset", "config": "default", "split": "train", "offset": 0, "length": 100},
                        timeout=30,
                        dataset_id="test/dataset",
                    )
                    assert resp.status_code == 200
                    assert call_count == 2

    def test_timeout_error_triggers_retry(self):
        """Test that timeout errors trigger retries."""
        # Simulate timeout for first 2 attempts, then succeed
        mock_responses = [
            requests.exceptions.Timeout("Read timed out"),
            requests.exceptions.Timeout("Read timed out"),
            MagicMock(status_code=200, json=lambda: {"rows": []}),
        ]

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            response = mock_responses[call_count]
            call_count += 1
            if isinstance(response, Exception):
                raise response
            return response

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session') as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get = mock_get
            mock_session_getter.return_value = mock_session

            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                with patch('time.sleep'):
                    resp = _datasets_server_get(
                        "rows",
                        params={"dataset": "test/dataset", "config": "default", "split": "train", "offset": 0, "length": 100},
                        timeout=30,
                        dataset_id="test/dataset",
                    )
                    assert resp.status_code == 200
                    assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
