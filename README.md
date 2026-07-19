# hermes-mcp-s-grok-search

Grok Responses `web_search` external module for `hermes-mcp-s`.

This project is a Python external module loaded by Sohri's `hermes-mcp-s`. It is not a standalone MCP server and does not implement stdio, HTTP, or SSE transports.

## Scope

Phase 1 MVP provides one tool:

- `grok_search`: sends a search query to a Grok/OpenAI Responses-style endpoint using `tools: [{"type": "web_search"}]` and returns answer text, citations, timing, and a warning when no search trace is detectable.

This module intentionally does not implement multi-dimensional search, a web UI, an HTTP server, SSE, or real CPA debugging flows.

## Why Responses + web_search

Grok Search is treated as a search channel, not as the host's primary chat model. The module calls:

```text
{base_url}/responses
```

with a Responses-style request body containing:

```json
{
  "tools": [{"type": "web_search"}]
}
```

It does not use `/v1/chat/completions` or a chat `messages` payload as the search path. If a response contains answer text but no detectable `web_search_call`, `search`, `citations`, `annotations`, `url`, or `source` trace, the tool returns `ok: true` with a warning instead of pretending that a verified web search occurred.

Some CPA/Grok Responses payloads omit structured `web_search_call` and annotations while returning source links in assistant text. In that case, the module extracts Markdown or bare HTTP(S) URLs into `citations` and reports `trace_status: text_citation_fallback`. This is weaker evidence than a structured search trace and is always surfaced with a warning. Responses with structured evidence report `trace_status: structured`; responses with neither structured evidence nor citation URLs report `trace_status: missing`.

## Configuration

Example `hermes-mcp-s` module config:

```yaml
modules:
  grok_search:
    enabled: true
    package: hermes_mcps_grok_search
    config:
      base_url: https://example.sohri.net/v1
      api_key_env: GROK_SEARCH_API_KEY
      model: grok-4.5-low
      timeout: 90
      search_context_size: medium
```

The same sample is available in `examples/hermes_config.yaml`.

Supported config keys:

- `base_url`: required API base URL for the Sohri CPA-compatible endpoint. There is no built-in provider default.
- `api_key_env`: environment variable name for the API key. Defaults to `GROK_SEARCH_API_KEY`.
- `model`: defaults to `grok-4.5-low`. Adjust to the CPA model list if needed, for example `grok-4.3-low` or `grok-4.2-no-reasoning`.
- `timeout`: request timeout in seconds. Defaults to `90`.
- `search_context_size`: `low`, `medium`, or `high`. Defaults to `medium`.
- `max_query_length`: query character limit. Defaults to `10000`.

Secrets belong only in local environment files or process environment. Example local `.env` content should use only the environment variable name:

```text
GROK_SEARCH_API_KEY=...
```

Do not commit `.env`, API keys, CPA tokens, cookies, account data, or Authorization headers.

## Tool Input

```json
{
  "query": "required string",
  "search_context_size": "medium",
  "include_raw": false,
  "max_output_tokens": 1200
}
```

`timeout` is configured at module level rather than accepted as tool input.

Use `include_raw=true` only for debugging. It may expose the upstream raw payload returned by the configured endpoint.

## Tool Output

Success shape:

```json
{
  "ok": true,
  "query": "...",
  "answer": "...",
  "citations": [],
  "model": "grok-4.5-low",
  "duration_ms": 123,
  "trace_status": "structured",
  "warning": "optional warning",
  "raw": "only when include_raw=true"
}
```

Failure shape:

```json
{
  "ok": false,
  "query": "...",
  "answer": "",
  "citations": [],
  "model": "grok-4.5-low",
  "duration_ms": 0,
  "error": "sanitized error"
}
```

Missing environment variables are reported as friendly tool errors and module status values. The module reports only `present` or `missing`; it does not output secret values.

## Development

Create a virtual environment, install the project in editable mode, and run tests:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m pytest
```

When testing against a local `hermes-mcp-s` checkout, use `PYTHONPATH` or that project's development workflow to make both packages importable. Do not hardcode a local `D:\Project\hermes-mcp-s\src` path in `pyproject.toml`.

All tests mock HTTP behavior. The test suite must not send real network requests.

## External Module Contract

The package top level exposes:

- `MODULE`
- `register_module(registry, context, config)`

`MODULE` metadata:

```python
{
    "name": "grok_search",
    "version": "0.1.0",
    "description": "Grok Responses web_search module for hermes-mcp-s",
    "required_env": ["GROK_SEARCH_API_KEY"],
    "provided_tools": ["grok_search"],
    "requires_model": False,
}
```

The implementation follows the current `hermes-mcp-s` external module style observed in `examples/hello_module`: `register_module(registry, context, config)` calls `registry.register_tool(..., module_name="grok_search")`.
