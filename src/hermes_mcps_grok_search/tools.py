from __future__ import annotations

from typing import Any

from .client import GrokSearchClient
from .validation import (
    GrokSearchConfig,
    normalize_search_context_size,
    optional_positive_int,
    validate_query,
)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "search_context_size": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Web search context size.",
        },
        "include_raw": {"type": "boolean", "description": "Include raw Responses payload."},
        "max_output_tokens": {"type": "integer", "minimum": 1, "description": "Optional response token cap."},
    },
    "required": ["query"],
    "additionalProperties": False,
}


def make_grok_search_handler(config: GrokSearchConfig, http_client: Any | None = None):
    client = GrokSearchClient(config, http_client=http_client)

    def grok_search(arguments: dict[str, Any] | None) -> dict[str, Any]:
        args = dict(arguments or {})
        try:
            query = validate_query(args.get("query"), config.max_query_length)
            search_context_size = normalize_search_context_size(args.get("search_context_size", config.search_context_size))
            include_raw = bool(args.get("include_raw", False))
            max_output_tokens = optional_positive_int(args.get("max_output_tokens"), "max_output_tokens")
        except ValueError as exc:
            return {
                "ok": False,
                "query": str(args.get("query") or ""),
                "answer": "",
                "citations": [],
                "model": config.model,
                "duration_ms": 0,
                "error": str(exc),
            }

        return client.search(
            query,
            search_context_size=search_context_size,
            include_raw=include_raw,
            max_output_tokens=max_output_tokens,
        )

    return grok_search
