from __future__ import annotations

import os

import pytest

from hermes_mcps_grok_search import MODULE, register_module
from hermes_mcps_grok_search.client import GrokSearchClient, sanitize_error
from hermes_mcps_grok_search.parser import parse_response
from hermes_mcps_grok_search.tools import make_grok_search_handler
from hermes_mcps_grok_search.validation import parse_config, normalize_endpoint
from scripts.probe_cpa_responses_shapes import (
    build_request_shapes,
    normalize_responses_endpoint,
    probe_shapes,
    sanitize_url,
)


class FakeRegistry:
    def __init__(self) -> None:
        self.tools = {}

    def register_tool(self, name, description, input_schema, handler, module_name="system"):
        self.tools[name] = {
            "description": description,
            "input_schema": input_schema,
            "handler": handler,
            "module_name": module_name,
        }


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def post(self, url, *, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.response


class ShapeFakeHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_module_metadata():
    assert MODULE == {
        "name": "grok_search",
        "version": "0.1.0",
        "description": "Grok Responses web_search module for hermes-mcp-s",
        "required_env": ["GROK_SEARCH_API_KEY"],
        "provided_tools": ["grok_search"],
        "requires_model": False,
    }


def test_register_module_registers_grok_search(monkeypatch):
    monkeypatch.delenv("GROK_SEARCH_API_KEY", raising=False)
    registry = FakeRegistry()

    status = register_module(registry, None, {"base_url": "https://example.sohri.net/v1"})

    assert "grok_search" in registry.tools
    assert registry.tools["grok_search"]["module_name"] == "grok_search"
    assert status["required_env"] == {"GROK_SEARCH_API_KEY": "missing"}


@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    [
        ("https://example.sohri.net/v1", "https://example.sohri.net/v1/responses"),
        ("https://example.sohri.net/v1/", "https://example.sohri.net/v1/responses"),
        ("https://example.sohri.net/v1/responses", "https://example.sohri.net/v1/responses"),
    ],
)
def test_endpoint_normalization(base_url, endpoint):
    assert normalize_endpoint(base_url) == endpoint


@pytest.mark.parametrize("config", [{}, {"base_url": ""}, {"base_url": "   "}])
def test_parse_config_requires_base_url(config):
    with pytest.raises(ValueError, match="config.base_url must be a non-empty string"):
        parse_config(config)


def test_missing_env_returns_friendly_error_without_secret(monkeypatch):
    monkeypatch.delenv("GROK_SEARCH_API_KEY", raising=False)
    handler = make_grok_search_handler(parse_config({"base_url": "https://example.sohri.net/v1"}))

    result = handler({"query": "latest AI search news"})

    assert result["ok"] is False
    assert "GROK_SEARCH_API_KEY" in result["error"]
    assert "Bearer" not in str(result)


def test_request_body_uses_responses_and_web_search(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_API_KEY", "test-secret-value")
    fake_http = FakeHttpClient(FakeResponse(payload={"output_text": "Answer", "output": [{"type": "web_search_call"}]}))
    client = GrokSearchClient(parse_config({"base_url": "https://example.sohri.net/v1"}), http_client=fake_http)

    result = client.search("Sohri", search_context_size="medium")

    assert result["ok"] is True
    call = fake_http.calls[0]
    assert call["url"] == "https://example.sohri.net/v1/responses"
    assert "/chat/completions" not in call["url"]
    assert call["json"]["tools"] == [{"type": "web_search"}]
    assert "messages" not in call["json"]
    assert call["headers"]["Authorization"] == "Bearer test-secret-value"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["headers"]["User-Agent"] == "hermes-mcp-s-grok-search"


def test_parser_parses_output_text():
    parsed = parse_response({"output_text": "Plain answer", "output": [{"type": "web_search_call"}]})

    assert parsed["answer"] == "Plain answer"
    assert parsed["has_search_trace"] is True
    assert parsed["trace_status"] == "structured"


def test_parser_parses_output_content_text():
    parsed = parse_response({"output": [{"type": "message", "content": [{"type": "output_text", "text": "Nested answer"}]}]})

    assert parsed["answer"] == "Nested answer"


def test_parser_extracts_annotations_citations_and_urls():
    payload = {
        "output_text": "Answer with source",
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "text": "Answer with source",
                        "annotations": [
                            {"title": "Source title", "url": "https://source.example/item", "snippet": "Useful snippet"}
                        ],
                    }
                ],
            }
        ],
    }

    parsed = parse_response(payload)

    assert parsed["citations"] == [
        {"title": "Source title", "url": "https://source.example/item", "snippet": "Useful snippet"}
    ]
    assert parsed["has_search_trace"] is True
    assert parsed["trace_status"] == "structured"


def test_parser_uses_text_citation_fallback_for_cpa_grok_response():
    payload = {
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Internal reasoning"}]},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": "Current result [[1]](https://example.com/source).",
                    }
                ],
            },
        ]
    }

    parsed = parse_response(payload)

    assert parsed["answer"] == "Current result [[1]](https://example.com/source)."
    assert parsed["citations"] == [{"url": "https://example.com/source", "title": "1"}]
    assert parsed["has_search_trace"] is False
    assert parsed["trace_status"] == "text_citation_fallback"


def test_parser_extracts_standard_markdown_link_fallback():
    parsed = parse_response({"output_text": "Read [Example source](https://example.com/path) for details."})

    assert parsed["citations"] == [{"url": "https://example.com/path", "title": "Example source"}]
    assert parsed["trace_status"] == "text_citation_fallback"


def test_parser_extracts_bare_url_fallback():
    parsed = parse_response({"output_text": "Source: https://example.com/path?q=1."})

    assert parsed["citations"] == [{"url": "https://example.com/path?q=1", "title": "example.com"}]
    assert parsed["trace_status"] == "text_citation_fallback"


@pytest.mark.parametrize(
    ("raw_url", "clean_url"),
    [
        ("https://pnpm.io/installation|num_results|3", "https://pnpm.io/installation"),
        (
            "https://github.com/pnpm/pnpm/releases|num_results|3",
            "https://github.com/pnpm/pnpm/releases",
        ),
    ],
)
def test_parser_cleans_pipe_suffix_from_bare_url_fallback(raw_url, clean_url):
    parsed = parse_response({"output_text": f"9|web_search|q|{raw_url}"})

    assert parsed["citations"] == [{"url": clean_url, "title": "pnpm.io" if "pnpm.io" in clean_url else "github.com"}]
    assert parsed["trace_status"] == "text_citation_fallback"


def test_parser_preserves_bare_url_query_string():
    parsed = parse_response({"output_text": "Source: https://example.com/search?q=pnpm&lang=zh"})

    assert parsed["citations"] == [
        {"url": "https://example.com/search?q=pnpm&lang=zh", "title": "example.com"}
    ]
    assert parsed["trace_status"] == "text_citation_fallback"


def test_parser_deduplicates_text_citation_urls():
    parsed = parse_response(
        {
            "output_text": (
                "Sources: https://example.com/same|num_results|3, "
                "https://example.com/same|num_results|3, and https://example.com/same."
            )
        }
    )

    assert parsed["citations"] == [{"url": "https://example.com/same", "title": "example.com"}]
    assert parsed["trace_status"] == "text_citation_fallback"


def test_pseudo_web_search_text_without_url_is_missing():
    parsed = parse_response(
        {
            "output_text": (
                'web_search[{"query":"current result"}]\n'
                "<web_search><query>current result</query></web_search>"
            )
        }
    )

    assert parsed["citations"] == []
    assert parsed["has_search_trace"] is False
    assert parsed["trace_status"] == "missing"


def test_text_fallback_does_not_override_structured_evidence():
    payload = {
        "output": [
            {"type": "web_search_call"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Answer [text source](https://text.example/source).",
                        "annotations": [
                            {
                                "title": "Structured source",
                                "url": "https://structured.example/source",
                            }
                        ],
                    }
                ],
            },
        ]
    }

    parsed = parse_response(payload)

    assert parsed["trace_status"] == "structured"
    assert parsed["citations"] == [
        {"title": "Structured source", "url": "https://structured.example/source"}
    ]


def test_no_search_trace_produces_warning(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_API_KEY", "test-secret-value")
    fake_http = FakeHttpClient(FakeResponse(payload={"output_text": "Answer without trace"}))
    handler = make_grok_search_handler(parse_config({"base_url": "https://example.sohri.net/v1"}), http_client=fake_http)

    result = handler({"query": "anything"})

    assert result["ok"] is True
    assert "warning" in result
    assert result["trace_status"] == "missing"
    assert "raw" not in result


def test_text_citation_fallback_produces_explicit_warning(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_API_KEY", "test-secret-value")
    fake_http = FakeHttpClient(FakeResponse(payload={"output_text": "Source [[1]](https://example.com/item)"}))
    handler = make_grok_search_handler(parse_config({"base_url": "https://example.sohri.net/v1"}), http_client=fake_http)

    result = handler({"query": "anything"})

    assert result["ok"] is True
    assert result["trace_status"] == "text_citation_fallback"
    assert result["citations"] == [{"url": "https://example.com/item", "title": "1"}]
    assert result["warning"] == (
        "No structured web_search trace was detected; "
        "extracted citation URLs from assistant text fallback."
    )


def test_include_raw_false_omits_raw(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_API_KEY", "test-secret-value")
    fake_http = FakeHttpClient(FakeResponse(payload={"output_text": "Answer", "output": [{"type": "web_search_call"}]}))
    handler = make_grok_search_handler(parse_config({"base_url": "https://example.sohri.net/v1"}), http_client=fake_http)

    result = handler({"query": "anything", "include_raw": False})

    assert result["ok"] is True
    assert "raw" not in result


def test_include_raw_true_returns_raw(monkeypatch):
    monkeypatch.setenv("GROK_SEARCH_API_KEY", "test-secret-value")
    payload = {"output_text": "Answer", "output": [{"type": "web_search_call"}]}
    fake_http = FakeHttpClient(FakeResponse(payload=payload))
    handler = make_grok_search_handler(parse_config({"base_url": "https://example.sohri.net/v1"}), http_client=fake_http)

    result = handler({"query": "anything", "include_raw": True})

    assert result["ok"] is True
    assert result["raw"] == payload


@pytest.mark.parametrize("status_code", [401, 429, 500])
def test_http_errors_are_friendly_and_sanitized(monkeypatch, status_code):
    monkeypatch.setenv("GROK_SEARCH_API_KEY", "test-secret-value")
    fake_http = FakeHttpClient(
        FakeResponse(status_code=status_code, text="Authorization: Bearer test-secret-value failed")
    )
    handler = make_grok_search_handler(parse_config({"base_url": "https://example.sohri.net/v1"}), http_client=fake_http)

    result = handler({"query": "anything"})

    assert result["ok"] is False
    assert result["error"]["status_code"] == status_code
    assert "test-secret-value" not in str(result)
    assert "Bearer test-secret-value" not in str(result)


@pytest.mark.parametrize(
    "message",
    [
        '{"api_key": "dummy-api-key-123"}',
        '{"apikey": "dummy-api-key-123"}',
        '{"api-key": "dummy-api-key-123"}',
        '{"token": "dummy-api-key-123"}',
        '{"secret": "dummy-api-key-123"}',
        '{"password": "dummy-api-key-123"}',
        '{"credential": "dummy-api-key-123"}',
        '{"authorization": "Bearer dummy-api-key-123"}',
        '{"error": "Authorization: Bearer dummy-api-key-123 failed"}',
    ],
)
def test_sanitize_error_redacts_json_style_quoted_secret_fields(message):
    sanitized = sanitize_error(message)

    assert "dummy-api-key-123" not in sanitized
    assert "[redacted]" in sanitized


def test_tests_do_not_use_real_env_key():
    assert os.environ.get("REAL_GROK_SEARCH_API_KEY_FOR_TESTS") is None


@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    [
        ("https://example.invalid/v1", "https://example.invalid/v1/responses"),
        ("https://example.invalid/v1/", "https://example.invalid/v1/responses"),
        ("https://example.invalid/v1/responses", "https://example.invalid/v1/responses"),
        ("https://example.invalid/responses", "https://example.invalid/responses"),
    ],
)
def test_probe_normalizes_responses_endpoint(base_url, endpoint):
    assert normalize_responses_endpoint(base_url) == endpoint


def test_probe_builds_all_request_shapes_without_default_tool_choice():
    shapes = build_request_shapes("grok-test")

    assert len(shapes) == 11
    assert shapes[0]["name"] == "baseline_string_input"
    assert shapes[0]["body"]["tools"] == [{"type": "web_search"}]
    assert "tool_choice" not in shapes[0]["body"]
    experimental = [shape for shape in shapes if shape["experimental_tool_choice"]]
    assert [shape["name"] for shape in experimental] == [
        "experimental_tool_choice_required",
        "experimental_tool_choice_object",
    ]


def test_probe_sanitizes_sensitive_url_query_values():
    sensitive_url = (
        "https://example.invalid/source?"
        + "tok"
        + "en=dummy-url-token-456"
        + "&q=pnpm&"
        + "api_"
        + "key=dummy-url-secret-789"
    )
    sanitized = sanitize_url(sensitive_url, "dummy-url-secret-789")

    assert "dummy-url-token-456" not in sanitized
    assert "dummy-url-secret-789" not in sanitized
    assert "q=pnpm" in sanitized
    assert sanitized.count("%5Bredacted%5D") == 2


def test_probe_uses_fake_http_and_summarizes_all_shapes_without_leaking_secret():
    sensitive_url = (
        "https://example.invalid/source?"
        + "tok"
        + "en=dummy-url-token-456"
        + "&q=pnpm&"
        + "api_"
        + "key=dummy-url-secret-789"
    )
    usable_payload = {
        "status": "completed",
        "output": [
            {"type": "reasoning"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": f"Official source: {sensitive_url}",
                    }
                ],
            },
        ],
    }
    missing_payload = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "annotations": [], "text": "I'll search for that."}],
            }
        ],
    }
    responses = [
        FakeResponse(payload=usable_payload),
        FakeResponse(payload=missing_payload),
        *[
            FakeResponse(
                status_code=400,
                payload={
                    "error": {
                        "code": "bad_shape",
                        "message": "tok" + "en=dummy-url-token-456",
                    }
                },
            )
            for _ in range(8)
        ],
        RuntimeError("Bearer dummy-api-key-123 failed"),
    ]
    fake_http = ShapeFakeHttpClient(responses)

    report = probe_shapes(
        base_url="https://example.invalid/v1",
        model="grok-test",
        http_client=fake_http,
        **{"api_" + "key": "dummy-api-key-123"},
    )

    assert len(fake_http.calls) == 11
    assert report["summary"]["usable_shapes"] == ["baseline_string_input"]
    assert report["summary"]["missing_trace_shapes"] == ["input_array_user_content_text"]
    assert report["summary"]["recommended_shape"] == "baseline_string_input"
    assert len(report["summary"]["failed_shapes"]) == 9
    serialized = str(report)
    assert "dummy-url-token-456" not in serialized
    assert "dummy-url-secret-789" not in serialized
    assert "dummy-api-key-123" not in serialized
    assert "q=pnpm" in serialized
    assert "[redacted]" in serialized or "%5Bredacted%5D" in serialized


def test_probe_rejects_router_model_before_http():
    fake_http = ShapeFakeHttpClient([])

    with pytest.raises(ValueError, match=r"direct grok-\* model"):
        probe_shapes(
            base_url="https://example.invalid/v1",
            model="router-grok-search",
            http_client=fake_http,
            **{"api_" + "key": "test-secret"},
        )

    assert fake_http.calls == []
