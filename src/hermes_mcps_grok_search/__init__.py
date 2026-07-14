from __future__ import annotations

import os

from .tools import TOOL_SCHEMA, make_grok_search_handler
from .validation import parse_config


MODULE = {
    "name": "grok_search",
    "version": "0.1.0",
    "description": "Grok Responses web_search module for hermes-mcp-s",
    "required_env": ["GROK_SEARCH_API_KEY"],
    "provided_tools": ["grok_search"],
    "requires_model": False,
}


def register_module(registry, context, config):
    parsed_config = parse_config(config)
    registry.register_tool(
        "grok_search",
        "Search the web through Grok Responses web_search and return answer text with citations.",
        TOOL_SCHEMA,
        make_grok_search_handler(parsed_config),
        module_name=MODULE["name"],
    )
    env_status = "present" if os.environ.get(parsed_config.api_key_env) else "missing"
    return {
        "status": "loaded" if env_status == "present" else "loaded_missing_env",
        "required_env": {parsed_config.api_key_env: env_status},
        "tools": ["grok_search"],
        "diagnostics": {
            "endpoint": "configured",
            "model": parsed_config.model,
            "search_context_size": parsed_config.search_context_size,
        },
    }


__all__ = ["MODULE", "register_module"]
