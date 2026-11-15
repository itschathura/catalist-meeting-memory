
# Unit tests for LLM Client, caching, and token metrics


import pytest
import tempfile
import json
from pathlib import Path
from src.llm import LLMClient


def test_cache_key_deterministic():
    client = LLMClient()
    key1 = client._compute_cache_key("test prompt", {"type": "OBJECT"}, "system prompt")
    key2 = client._compute_cache_key("test prompt", {"type": "OBJECT"}, "system prompt")
    key3 = client._compute_cache_key("different prompt", {"type": "OBJECT"}, "system prompt")

    assert key1 == key2
    assert key1 != key3


def test_metrics_tracking():
    client = LLMClient()
    initial_calls = client.metrics["total_calls"]
    initial_tokens = client.metrics["total_tokens"]

    # Record synthetic usage
    client._record_usage(prompt_tokens=100, completion_tokens=50, from_cache=False)
    assert client.metrics["total_calls"] == initial_calls + 1
    assert client.metrics["total_tokens"] == initial_tokens + 150
    assert client.metrics["cache_misses"] >= 1

    # Record cache hit
    client._record_usage(prompt_tokens=100, completion_tokens=50, from_cache=True)
    assert client.metrics["cache_hits"] >= 1


def test_structured_output_generation():
    client = LLMClient()
    schema = {
        "type": "OBJECT",
        "properties": {
            "fruit": {"type": "STRING"},
            "color": {"type": "STRING"}
        },
        "required": ["fruit", "color"]
    }
    
    result = client.generate(
        "Return an apple and its primary color.",
        response_schema=schema,
        use_cache=True
    )
    
    assert "parsed" in result
    assert result["parsed"] is not None
    assert "fruit" in result["parsed"]
    assert "color" in result["parsed"]
