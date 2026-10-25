"""Tests for Tier-2 AI Skeleton Healer and Drift Alert Notifier."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from payload_heal.ai_healer import (
    build_user_prompt,
    extract_json_from_response,
    heal_with_llm,
)
from payload_heal.config import HealConfig
from payload_heal.notifier import send_drift_notification

# ==============================================================================
# HealConfig Tests
# ==============================================================================

def test_heal_config_defaults_and_providers():
    # OpenAI default
    cfg_openai = HealConfig()
    assert cfg_openai.llm_provider == "openai"
    assert cfg_openai.model == "gpt-4o-mini"
    assert "api.openai.com" in cfg_openai.api_base

    # Groq provider auto-detection
    cfg_groq = HealConfig(llm_provider="groq")
    assert cfg_groq.model == "llama-3.3-70b-versatile"
    assert "api.groq.com" in cfg_groq.api_base

    # Ollama provider auto-detection
    cfg_ollama = HealConfig(llm_provider="ollama")
    assert cfg_ollama.model == "llama3"
    assert "11434" in cfg_ollama.api_base

    # Gemini provider auto-detection
    cfg_gemini = HealConfig(llm_provider="gemini")
    assert cfg_gemini.model == "gemini-1.5-flash"
    assert "generativelanguage.googleapis.com" in cfg_gemini.api_base

    # Custom base URL override
    cfg_custom = HealConfig(api_base="https://my-proxy.company.com/v1")
    assert cfg_custom.api_base == "https://my-proxy.company.com/v1"


# ==============================================================================
# JSON Extraction Tests
# ==============================================================================

def test_extract_json_from_response():
    # Pure JSON
    pure = '{"mappings": {"price": "payload.cost"}}'
    assert extract_json_from_response(pure) == {"mappings": {"price": "payload.cost"}}

    # Markdown fenced JSON (Ollama style)
    fenced = "Here is the mapping:\n```json\n{\n  \"mappings\": {\"price\": \"payload.cost\"}\n}\n```"
    assert extract_json_from_response(fenced) == {"mappings": {"price": "payload.cost"}}

    # Fenced without language tag
    fenced_no_tag = "```\n{\"mappings\": {\"title\": \"data.name\"}}\n```"
    assert extract_json_from_response(fenced_no_tag) == {"mappings": {"title": "data.name"}}

    # Text before and after
    messy = "I have mapped your fields:\n{\"mappings\": {\"id\": \"order.num\"}}\nHope this helps!"
    assert extract_json_from_response(messy) == {"mappings": {"id": "order.num"}}

    # Invalid input
    assert extract_json_from_response("Not a JSON") is None
    assert extract_json_from_response("") is None


# ==============================================================================
# AI Healer with Mocked HTTP Responses
# ==============================================================================

def test_heal_with_llm_openai_success():
    """Simulates a standard OpenAI/Groq response with valid skeleton mappings."""
    skeleton = {
        "payload.pricing.priceUsd": {"type": "float", "sample": "49.99"},
        "payload.product.sku": {"type": "str", "sample": "SKU-123"},
    }
    missing_fields = {
        "price_usd": {"type": "float", "description": "Price in US Dollars"},
        "sku": {"type": "str", "description": "Product SKU code"},
    }

    mock_response = httpx.Response(
        status_code=200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "mappings": {
                                "price_usd": "payload.pricing.priceUsd",
                                "sku": "payload.product.sku",
                            }
                        })
                    }
                }
            ]
        },
    )

    config = HealConfig(api_key="test-key")

    with patch.object(httpx.Client, "post", return_value=mock_response):
        mappings = heal_with_llm("ProductModel", missing_fields, skeleton, config)

    assert mappings == {
        "price_usd": "payload.pricing.priceUsd",
        "sku": "payload.product.sku",
    }


def test_heal_with_llm_markdown_fenced_response():
    """Simulates an Ollama model returning markdown code blocks."""
    skeleton = {
        "user.contact.emailAddress": {"type": "str", "sample": "[MASKED_EMAIL]"},
    }
    missing_fields = {
        "email": {"type": "str", "description": "User email address"},
    }

    fenced_content = """```json
{
  "mappings": {
    "email": "user.contact.emailAddress"
  }
}
```"""

    mock_response = httpx.Response(
        status_code=200,
        request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
        json={
            "choices": [
                {
                    "message": {
                        "content": fenced_content,
                    }
                }
            ]
        },
    )

    config = HealConfig(llm_provider="ollama")

    with patch.object(httpx.Client, "post", return_value=mock_response):
        mappings = heal_with_llm("UserModel", missing_fields, skeleton, config)

    assert mappings == {"email": "user.contact.emailAddress"}


def test_heal_with_llm_filters_hallucinated_paths():
    """Requirement: Anti-hallucination check filters out paths that do not exist in the skeleton."""
    skeleton = {
        "data.real_path": {"type": "str", "sample": "Valid"},
    }
    missing_fields = {
        "real_field": {"type": "str"},
        "hallucinated_field": {"type": "str"},
    }

    mock_response = httpx.Response(
        status_code=200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "mappings": {
                                "real_field": "data.real_path",
                                "hallucinated_field": "data.invented_path_that_does_not_exist",
                            }
                        })
                    }
                }
            ]
        },
    )

    config = HealConfig()

    with patch.object(httpx.Client, "post", return_value=mock_response):
        mappings = heal_with_llm("TestModel", missing_fields, skeleton, config)

    # Valid path is kept, hallucinated path is discarded
    assert "real_field" in mappings
    assert mappings["real_field"] == "data.real_path"
    assert "hallucinated_field" not in mappings


def test_heal_with_llm_network_timeout_resilience():
    """Requirement: Never raise unhandled exceptions on network failure, return empty dict."""
    skeleton = {"data.val": {"type": "int", "sample": "10"}}
    missing_fields = {"val": {"type": "int"}}

    config = HealConfig(timeout_seconds=0.1)

    with patch.object(httpx.Client, "post", side_effect=httpx.TimeoutException("Connection timed out")):
        mappings = heal_with_llm("TestModel", missing_fields, skeleton, config)

    assert mappings == {}


def test_heal_with_llm_http_error_resilience():
    """Handles HTTP 500 or 401 gracefully without crashing."""
    skeleton = {"data.val": {"type": "int", "sample": "10"}}
    missing_fields = {"val": {"type": "int"}}

    error_response = httpx.Response(
        status_code=500,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        text="Internal Server Error",
    )

    config = HealConfig()

    with patch.object(httpx.Client, "post", return_value=error_response):
        mappings = heal_with_llm("TestModel", missing_fields, skeleton, config)

    assert mappings == {}


def test_heal_with_llm_disabled_config():
    """Respects config.enabled = False or config.enable_ai = False."""
    skeleton = {"data.val": {"type": "int", "sample": "10"}}
    missing_fields = {"val": {"type": "int"}}

    cfg_disabled = HealConfig(enable_ai=False)
    assert heal_with_llm("TestModel", missing_fields, skeleton, cfg_disabled) == {}


# ==============================================================================
# Notifier Webhook Tests
# ==============================================================================

def test_send_drift_notification_success():
    config = HealConfig(webhook_url="https://hooks.slack.com/services/T00/B00/X00")
    healed_fields = {
        "price": ("data.pricing.priceUsd", "heuristic"),
        "user_id": ("data.user.id", "ai"),
    }

    mock_resp = httpx.Response(
        status_code=200,
        request=httpx.Request("POST", config.webhook_url),
        text="ok",
    )

    with patch.object(httpx.Client, "post", return_value=mock_resp) as mock_post:
        success = send_drift_notification("OrderModel", healed_fields, config)
        assert success is True

        # Verify POST arguments
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert "OrderModel" in payload["text"]
        assert "data.pricing.priceUsd" in payload["text"]
        assert payload["drift_count"] == 2
        assert payload["healed_fields"]["price"]["healed_by"] == "heuristic"


def test_send_drift_notification_no_webhook():
    config = HealConfig(webhook_url=None)
    healed = {"title": ("data.name", "heuristic")}
    # Should cleanly return False without attempting network request
    assert send_drift_notification("Model", healed, config) is False


def test_send_drift_notification_handles_failure():
    config = HealConfig(webhook_url="https://broken-webhook.example.com")
    healed = {"title": ("data.name", "heuristic")}

    with patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("Network unreachable")):
        # Must catch exception and return False without propagating
        assert send_drift_notification("Model", healed, config) is False
