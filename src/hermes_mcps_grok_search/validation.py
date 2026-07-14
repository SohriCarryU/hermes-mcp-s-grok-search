from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_API_KEY_ENV = "GROK_SEARCH_API_KEY"
DEFAULT_MODEL = "grok-4.5-low"
DEFAULT_TIMEOUT = 90.0
DEFAULT_SEARCH_CONTEXT_SIZE = "medium"
DEFAULT_MAX_QUERY_LENGTH = 10000
ALLOWED_CONTEXT_SIZES = {"low", "medium", "high"}


@dataclass(frozen=True)
class GrokSearchConfig:
    base_url: str
    api_key_env: str = DEFAULT_API_KEY_ENV
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    search_context_size: str = DEFAULT_SEARCH_CONTEXT_SIZE
    max_query_length: int = DEFAULT_MAX_QUERY_LENGTH


def parse_config(config: dict[str, Any] | None) -> GrokSearchConfig:
    data = dict(config or {})
    base_url = str(data.get("base_url") or "").strip()
    if not base_url:
        raise ValueError("config.base_url must be a non-empty string")

    api_key_env = str(data.get("api_key_env") or DEFAULT_API_KEY_ENV).strip()
    if not api_key_env:
        raise ValueError("config.api_key_env must be a non-empty string")

    model = str(data.get("model") or DEFAULT_MODEL).strip()
    if not model:
        raise ValueError("config.model must be a non-empty string")

    timeout = _positive_float(data.get("timeout", DEFAULT_TIMEOUT), "config.timeout")
    search_context_size = normalize_search_context_size(data.get("search_context_size", DEFAULT_SEARCH_CONTEXT_SIZE))
    max_query_length = _positive_int(data.get("max_query_length", DEFAULT_MAX_QUERY_LENGTH), "config.max_query_length")

    return GrokSearchConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        model=model,
        timeout=timeout,
        search_context_size=search_context_size,
        max_query_length=max_query_length,
    )


def normalize_endpoint(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/responses"):
        return cleaned
    return f"{cleaned}/responses"


def validate_query(query: Any, max_length: int) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"query is too long; maximum length is {max_length}")
    return normalized


def normalize_search_context_size(value: Any) -> str:
    normalized = str(value or DEFAULT_SEARCH_CONTEXT_SIZE).strip().lower()
    if normalized not in ALLOWED_CONTEXT_SIZES:
        raise ValueError("search_context_size must be one of: low, medium, high")
    return normalized


def optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return parsed


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed
