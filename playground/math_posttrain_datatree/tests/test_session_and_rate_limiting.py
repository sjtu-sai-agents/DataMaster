"""Comprehensive unit tests for new session management and rate limiting features."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from playground.math_posttrain_datatree.core.utils.data import (
    _create_datasets_server_session,
    _get_datasets_server_session,
    _rate_limit_request,
    _datasets_server_get,
    MIN_REQUEST_INTERVAL_SECONDS,
)


class TestSessionManagement:
    """Test session creation and management features."""

    def test_create_session_has_correct_configuration(self):
        """Test that created session has correct retry and pooling config."""
        session = _create_datasets_server_session()

        assert isinstance(session, requests.Session)

        # Check headers
        assert session.headers['Connection'] == 'keep-alive'
        assert 'User-Agent' in session.headers

        # Check adapters are mounted
        assert 'https://' in session.adapters
        assert 'http://' in session.adapters

        # Check adapter configuration
        adapter = session.get_adapter('https://')
        assert hasattr(adapter, 'max_retries')

    def test_session_singleton_behavior(self):
        """Test that the same session is reused (singleton pattern)."""
        # Clear any existing session
        import playground.math_posttrain_datatree.core.utils.data as data_module
        data_module._DATASETS_SERVER_SESSION = None

        session1 = _get_datasets_server_session()
        session2 = _get_datasets_server_session()
        session3 = _get_datasets_server_session()

        assert session1 is session2
        assert session2 is session3
        assert id(session1) == id(session2) == id(session3)

    def test_session_headers_preserved_across_requests(self):
        """Test that session headers are preserved."""
        session = _get_datasets_server_session()

        headers_before = dict(session.headers)

        # Simulate a request (mock it to avoid actual network call)
        with patch.object(session, 'get', return_value=MagicMock(status_code=200)):
            session.get('https://example.com')

        headers_after = dict(session.headers)

        assert headers_before == headers_after


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_enforces_minimum_interval(self):
        """Test that rate limiting enforces minimum time between requests."""
        import playground.math_posttrain_datatree.core.utils.data as data_module

        # Reset last request time
        data_module._LAST_REQUEST_TIME = 0.0

        start_time = time.time()

        # First request - should be immediate
        _rate_limit_request()
        time_after_first = time.time()
        first_duration = time_after_first - start_time

        # Second request - should be delayed
        _rate_limit_request()
        time_after_second = time.time()
        second_duration = time_after_second - time_after_first

        # First request should be fast (< 50ms overhead)
        assert first_duration < 0.05, "First request should not be delayed"

        # Second request should respect the rate limit
        assert second_duration >= MIN_REQUEST_INTERVAL_SECONDS * 0.9, \
            f"Second request should wait at least {MIN_REQUEST_INTERVAL_SECONDS}s"

    def test_rate_limit_does_not_delay_if_enough_time_passed(self):
        """Test that rate limiting doesn't delay if enough time has passed."""
        import playground.math_posttrain_datatree.core.utils.data as data_module

        # Set last request time to past
        data_module._LAST_REQUEST_TIME = time.time() - 1.0  # 1 second ago

        start_time = time.time()
        _rate_limit_request()
        duration = time.time() - start_time

        # Should not have been delayed (< 50ms overhead)
        assert duration < 0.05

    def test_rate_limit_multiple_rapid_calls(self):
        """Test rate limiting works correctly for multiple rapid calls."""
        import playground.math_posttrain_datatree.core.utils.data as data_module

        # Reset
        data_module._LAST_REQUEST_TIME = 0.0

        start_time = time.time()

        # Make 5 rapid calls
        for _ in range(5):
            _rate_limit_request()

        total_duration = time.time() - start_time

        # Total time should be at least 4 intervals (5 calls = 4 waits)
        expected_min_duration = MIN_REQUEST_INTERVAL_SECONDS * 4
        assert total_duration >= expected_min_duration * 0.9, \
            f"Total duration should be at least {expected_min_duration}s for 5 calls"


class TestImprovedSSLHandling:
    """Test the improved SSL error handling in _datasets_server_get."""

    def test_ssl_error_uses_longer_backoff(self):
        """Test that SSL errors use 2x longer backoff than regular errors."""
        call_count = 0

        def mock_get_ssl_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.SSLError("SSL error")

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session') as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get = mock_get_ssl_error
            mock_session_getter.return_value = mock_session

            with patch('time.sleep') as mock_sleep:
                with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                    with pytest.raises(requests.exceptions.SSLError):
                        _datasets_server_get(
                            "rows",
                            params={"dataset": "test"},
                            timeout=30,
                            dataset_id="test",
                        )

                    # Check that sleep was called with 2x backoff for SSL errors
                    # Expected: 1*2=2, 2*2=4, 3*2=6 seconds
                    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                    assert len(sleep_calls) == 3  # 4 attempts - 1 = 3 sleeps
                    assert sleep_calls[0] == 2.0  # 1 * 1.0 * 2
                    assert sleep_calls[1] == 4.0  # 2 * 1.0 * 2
                    assert sleep_calls[2] == 6.0  # 3 * 1.0 * 2

    def test_non_ssl_error_uses_normal_backoff(self):
        """Test that non-SSL errors use normal backoff."""
        call_count = 0

        def mock_get_timeout_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.Timeout("Timeout")

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session') as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get = mock_get_timeout_error
            mock_session_getter.return_value = mock_session

            with patch('time.sleep') as mock_sleep:
                with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                    with pytest.raises(requests.exceptions.Timeout):
                        _datasets_server_get(
                            "rows",
                            params={"dataset": "test"},
                            timeout=30,
                            dataset_id="test",
                        )

                    # Check that sleep was called with normal backoff
                    # Expected: 1, 2, 3 seconds
                    sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                    assert len(sleep_calls) == 3
                    assert sleep_calls[0] == 1.0  # 1 * 1.0
                    assert sleep_calls[1] == 2.0  # 2 * 1.0
                    assert sleep_calls[2] == 3.0  # 3 * 1.0

    def test_uses_session_instead_of_direct_requests(self):
        """Test that the function uses session instead of requests.get directly."""
        mock_session = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_session.get.return_value = mock_response

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session', return_value=mock_session):
            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                resp = _datasets_server_get(
                    "rows",
                    params={"dataset": "test"},
                    timeout=30,
                    dataset_id="test",
                )

                # Verify session.get was called (not requests.get)
                assert mock_session.get.called
                assert resp == mock_response

    def test_uses_tuple_timeout_for_connect_and_read(self):
        """Test that timeout is converted to tuple (connect, read)."""
        mock_session = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_session.get.return_value = mock_response

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session', return_value=mock_session):
            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                _datasets_server_get(
                    "rows",
                    params={"dataset": "test"},
                    timeout=30,
                    dataset_id="test",
                )

                # Check that timeout was passed as tuple
                call_kwargs = mock_session.get.call_args[1]
                assert 'timeout' in call_kwargs
                timeout_arg = call_kwargs['timeout']
                assert isinstance(timeout_arg, tuple)
                assert timeout_arg == (10, 30)  # (connect, read)

    def test_rate_limiting_is_called_before_each_request(self):
        """Test that rate limiting is enforced before each retry attempt."""
        attempts = 0

        def mock_get_fail_twice(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise requests.exceptions.ConnectionError("Connection error")
            return MagicMock(status_code=200)

        mock_session = MagicMock()
        mock_session.get = mock_get_fail_twice

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session', return_value=mock_session):
            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request') as mock_rate_limit:
                with patch('time.sleep'):
                    _datasets_server_get(
                        "rows",
                        params={"dataset": "test"},
                        timeout=30,
                        dataset_id="test",
                    )

                    # Rate limiting should be called 3 times (once per attempt)
                    assert mock_rate_limit.call_count == 3


class TestBackwardCompatibility:
    """Test that the changes maintain backward compatibility."""

    def test_function_signature_unchanged(self):
        """Test that _datasets_server_get signature is unchanged."""
        import inspect
        sig = inspect.signature(_datasets_server_get)

        assert 'endpoint' in sig.parameters
        assert 'params' in sig.parameters
        assert 'timeout' in sig.parameters
        assert 'dataset_id' in sig.parameters

    def test_timeout_accepts_int(self):
        """Test that timeout still accepts int (not just tuple)."""
        mock_session = MagicMock()
        mock_response = MagicMock(status_code=200)
        mock_session.get.return_value = mock_response

        with patch('playground.math_posttrain_datatree.core.utils.data._get_datasets_server_session', return_value=mock_session):
            with patch('playground.math_posttrain_datatree.core.utils.data._rate_limit_request'):
                # Should not raise TypeError
                _datasets_server_get(
                    "rows",
                    params={"dataset": "test"},
                    timeout=30,  # int, not tuple
                    dataset_id="test",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
