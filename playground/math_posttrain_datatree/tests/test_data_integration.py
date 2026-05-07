"""Integration tests for the improved data.py SSL error handling."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from playground.math_posttrain_datatree.core.utils.data import (
    _datasets_server_get,
    _get_datasets_server_session,
    _materialize_via_datasets_server,
    materialize_dataset_entry,
)


class TestDataIntegration:
    """Integration tests with real datasets-server requests."""

    def test_session_is_reused_across_requests(self):
        """Test that the same session instance is reused for multiple requests."""
        session1 = _get_datasets_server_session()
        session2 = _get_datasets_server_session()
        assert session1 is session2, "Session should be reused across calls"

    def test_datasets_server_get_real_request(self):
        """Test real request to datasets-server for splits."""
        resp = _datasets_server_get(
            "splits",
            params={"dataset": "openai/gsm8k"},
            timeout=30,
            dataset_id="openai/gsm8k",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "splits" in data
        assert len(data["splits"]) > 0

    def test_datasets_server_get_rows_request(self):
        """Test real request to datasets-server for rows."""
        resp = _datasets_server_get(
            "rows",
            params={
                "dataset": "openai/gsm8k",
                "config": "main",
                "split": "train",
                "offset": 0,
                "length": 10,
            },
            timeout=30,
            dataset_id="openai/gsm8k",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rows" in data
        assert len(data["rows"]) > 0

    def test_materialize_via_datasets_server_real(self):
        """Test materialization via datasets-server with real dataset."""
        entry = {
            "source_id": "openai/gsm8k",
            "config": "main",
            "split": "train",
        }

        rows = _materialize_via_datasets_server(
            entry,
            "openai/gsm8k",
            max_rows=50,
        )

        assert len(rows) > 0
        assert len(rows) <= 50
        # Check that rows have expected structure
        assert all(isinstance(row, dict) for row in rows)
        # GSM8K should have problem/answer fields
        assert any("question" in row or "problem" in row for row in rows[:5])

    def test_materialize_dataset_entry_with_cache(self, tmp_path):
        """Test full materialization pipeline with caching."""
        entry = {
            "source_id": "openai/gsm8k",
            "config": "main",
            "split": "train",
        }

        cache_dir = tmp_path / "test_cache"

        # First call - should fetch from server
        path1 = materialize_dataset_entry(
            entry,
            cache_dir=cache_dir,
            max_rows=30,
        )

        assert path1
        assert cache_dir.exists()
        from pathlib import Path
        assert Path(path1).exists()

        # Second call - should use cache
        path2 = materialize_dataset_entry(
            entry,
            cache_dir=cache_dir,
            max_rows=30,
        )

        assert path1 == path2  # Same path

    def test_multiple_concurrent_requests_no_ssl_error(self):
        """
        Test that multiple rapid requests don't cause SSL errors.
        This simulates the scenario from the bug report.
        """
        datasets = [
            ("openai/gsm8k", "main", "train"),
            ("openai/gsm8k", "main", "test"),
            ("openai/gsm8k", "socratic", "train"),
        ]

        all_rows = []
        for dataset_id, config, split in datasets:
            entry = {
                "source_id": dataset_id,
                "config": config,
                "split": split,
            }

            rows = _materialize_via_datasets_server(
                entry,
                dataset_id,
                max_rows=20,
            )

            all_rows.append(rows)
            assert len(rows) > 0, f"Failed to fetch rows for {dataset_id}/{config}/{split}"

        # All requests should succeed
        assert len(all_rows) == 3
        assert all(len(rows) > 0 for rows in all_rows)

    def test_invalid_dataset_handles_gracefully(self):
        """Test that invalid dataset requests are handled gracefully."""
        with pytest.raises(Exception):  # Should raise some error
            _datasets_server_get(
                "splits",
                params={"dataset": "nonexistent/dataset_12345"},
                timeout=10,
                dataset_id="nonexistent/dataset_12345",
            )

    @pytest.mark.parametrize("dataset_id,config,split", [
        ("openai/gsm8k", "main", "train"),
        ("openai/gsm8k", "socratic", "train"),
    ])
    def test_different_configs_materialize_correctly(self, dataset_id, config, split):
        """Test materialization works for different dataset configs."""
        entry = {
            "source_id": dataset_id,
            "config": config,
            "split": split,
        }

        rows = _materialize_via_datasets_server(
            entry,
            dataset_id,
            max_rows=15,
        )

        assert len(rows) > 0
        assert len(rows) <= 15
        # Verify dataset_config is set
        if config:
            assert any(row.get("dataset_config") == config for row in rows)


def test_materialize_dataset_entry_skips_datasets_server_when_disabled(tmp_path):
    entry = {
        "source_id": "math-ai/amc23",
        "split": "train",
    }

    fake_datasets = types.SimpleNamespace(
        get_dataset_config_names=lambda dataset_id: [],
        get_dataset_split_names=lambda **kwargs: ["train"],
        load_dataset=lambda *args, **kwargs: [],
    )

    with patch.dict(sys.modules, {"datasets": fake_datasets}):
        with patch(
            "playground.math_posttrain_datatree.core.utils.data._materialize_via_datasets_server"
        ) as mock_server:
            path = materialize_dataset_entry(
                entry,
                cache_dir=tmp_path / "cache",
                max_rows=10,
                data_access_config={
                    "hf_endpoint": "https://hf-mirror.com",
                    "datasets_server": {
                        "enabled": False,
                        "base_url": "https://datasets-server.huggingface.co",
                    },
                },
            )

    assert path
    mock_server.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
