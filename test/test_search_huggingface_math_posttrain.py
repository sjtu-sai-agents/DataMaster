from __future__ import annotations

from types import SimpleNamespace

import requests

from search_dataset_tools import search_huggingface_math_posttrain as hf_search


def test_search_datasets_falls_back_to_huggingface_hub_when_rest_api_fails(monkeypatch) -> None:
    def fake_request_json(*args, **kwargs):
        raise requests.ConnectionError("[Errno 111] Connection refused")

    fake_result = SimpleNamespace(
        id="test-org/aime-dataset",
        author="test-org",
        downloads=123,
        likes=7,
    )

    def fake_list_datasets(**kwargs):
        assert kwargs["search"] == "AIME"
        assert kwargs["limit"] == 20
        return [fake_result]

    monkeypatch.setattr(hf_search, "_request_json", fake_request_json)
    monkeypatch.setattr(hf_search, "list_datasets", fake_list_datasets)

    result = hf_search.search_datasets("AIME", limit=20)

    assert "Found 1 datasets for 'AIME':" in result
    assert "test-org/aime-dataset" in result
    assert "Downloads: 123" in result


def test_search_datasets_returns_error_string_when_rest_and_hub_fallback_both_fail(monkeypatch) -> None:
    def fake_request_json(*args, **kwargs):
        raise requests.ConnectionError("[Errno 111] Connection refused")

    def fake_list_datasets(**kwargs):
        raise RuntimeError("hub fallback failed")

    monkeypatch.setattr(hf_search, "_request_json", fake_request_json)
    monkeypatch.setattr(hf_search, "list_datasets", fake_list_datasets)

    result = hf_search.search_datasets("AIME", limit=20)

    assert result.startswith("Error searching datasets:")
    assert "Connection refused" in result
    assert "hub fallback failed" in result



def test_build_session_does_not_respect_retry_after_by_default(monkeypatch) -> None:
    monkeypatch.delenv("HF_RESPECT_RETRY_AFTER", raising=False)
    session = hf_search._build_session()
    retries = session.adapters["https://"].max_retries
    assert retries.respect_retry_after_header is False


def test_build_session_can_enable_retry_after_via_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_RESPECT_RETRY_AFTER", "1")
    session = hf_search._build_session()
    retries = session.adapters["https://"].max_retries
    assert retries.respect_retry_after_header is True
