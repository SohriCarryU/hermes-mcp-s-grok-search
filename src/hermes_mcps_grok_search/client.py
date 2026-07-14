from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import re
import time

import httpx

from .parser import parse_response
from .validation import GrokSearchConfig, normalize_endpoint


USER_AGENT = "hermes-mcp-s-grok-search"
SENSITIVE_PATTERN = re.compile(
    r"(authorization\s*[:=]\s*bearer\s+\S+|bearer\s+\S+|(?:api[_-]?key|apikey|token|secret|password|credential)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)
JSON_SECRET_FIELD_PATTERN = re.compile(
    r'("(?:api[_-]?key|apikey|token|secret|password|credential|authorization)"\s*:\s*)"(?:[^"\\]|\\.)*"',
    re.IGNORECASE,
)


class GrokSearchClient:
    def __init__(self, config: GrokSearchConfig, http_client: Any | None = None) -> None:
        self.config = config
        self.endpoint = normalize_endpoint(config.base_url)
        self._http_client = http_client

    def search(
        self,
        query: str,
        *,
        search_context_size: str,
        include_raw: bool = False,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            return {
                "ok": False,
                "query": query,
                "answer": "",
                "citations": [],
                "model": self.config.model,
                "duration_ms": 0,
                "error": f"Missing required environment variable: {self.config.api_key_env}",
            }

        body = self._build_body(query, search_context_size=search_context_size, max_output_tokens=max_output_tokens)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        started = time.perf_counter()
        try:
            response = self._post(body, headers)
            duration_ms = int((time.perf_counter() - started) * 1000)
            if getattr(response, "status_code", 0) >= 400:
                return self._error_result(query, duration_ms, response)
            payload = response.json()
        except httpx.HTTPError as exc:
            return self._exception_result(query, started, exc)
        except ValueError as exc:
            return self._exception_result(query, started, exc, message="Invalid JSON response")

        parsed = parse_response(payload)
        if not parsed["answer"]:
            return {
                "ok": False,
                "query": query,
                "answer": "",
                "citations": parsed["citations"],
                "model": self.config.model,
                "duration_ms": duration_ms,
                "error": "Grok Responses returned no answer text.",
            }

        result: dict[str, Any] = {
            "ok": True,
            "query": query,
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "model": self.config.model,
            "duration_ms": duration_ms,
        }
        if not parsed["has_search_trace"]:
            result["warning"] = "Response contained answer text but no detectable web_search trace. Treat it as unverified."
        if include_raw:
            result["raw"] = payload
        return result

    def _build_body(self, query: str, *, search_context_size: str, max_output_tokens: int | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        body: dict[str, Any] = {
            "model": self.config.model,
            "input": f"Current time: {now}\nSearch the web and answer with citations when available.\nQuery: {query}",
            "tools": [{"type": "web_search"}],
        }
        if search_context_size:
            body["search_context_size"] = search_context_size
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        return body

    def _post(self, body: dict[str, Any], headers: dict[str, str]) -> Any:
        if self._http_client is not None:
            return self._http_client.post(self.endpoint, json=body, headers=headers, timeout=self.config.timeout)
        with httpx.Client(timeout=self.config.timeout) as client:
            return client.post(self.endpoint, json=body, headers=headers)

    def _error_result(self, query: str, duration_ms: int, response: Any) -> dict[str, Any]:
        text = _safe_response_text(response)
        status_code = getattr(response, "status_code", None)
        return {
            "ok": False,
            "query": query,
            "answer": "",
            "citations": [],
            "model": self.config.model,
            "duration_ms": duration_ms,
            "error": {
                "status_code": status_code,
                "message": sanitize_error(text or "HTTP request failed"),
            },
        }

    def _exception_result(self, query: str, started: float, exc: Exception, message: str | None = None) -> dict[str, Any]:
        return {
            "ok": False,
            "query": query,
            "answer": "",
            "citations": [],
            "model": self.config.model,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "error": sanitize_error(message or str(exc) or exc.__class__.__name__),
        }


def sanitize_error(value: Any) -> str:
    text = str(value).replace("\n", " ")
    text = JSON_SECRET_FIELD_PATTERN.sub(r'\1"[redacted]"', text)
    return SENSITIVE_PATTERN.sub("[redacted]", text)[:500]


def _safe_response_text(response: Any) -> str:
    try:
        return str(response.text)
    except Exception:  # noqa: BLE001 - defensive error formatting only.
        return ""
