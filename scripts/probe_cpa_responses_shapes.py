from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from hermes_mcps_grok_search.client import sanitize_error
from hermes_mcps_grok_search.parser import parse_response


QUERY = "What is the latest stable pnpm version? Cite official sources."
BASELINE_INPUT = f"Search the web and answer with official source citations. Query: {QUERY}"
INSTRUCTIONS = "You must use web search when available. Return concise answer and cite official URLs."
DEFAULT_TIMEOUT = 30.0
SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "api-key",
    "token",
    "secret",
    "password",
    "credential",
    "authorization",
)


def normalize_responses_endpoint(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("GROK_SEARCH_BASE_URL must be a non-empty string")
    if cleaned.endswith("/responses"):
        return cleaned
    return f"{cleaned}/responses"


def build_request_shapes(model: str) -> list[dict[str, Any]]:
    baseline = {
        "model": model,
        "input": BASELINE_INPUT,
        "tools": [{"type": "web_search"}],
        "max_output_tokens": 1200,
    }
    return [
        _shape("baseline_string_input", baseline),
        _shape(
            "input_array_user_content_text",
            {
                **baseline,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": BASELINE_INPUT}]}],
            },
        ),
        _shape(
            "input_array_user_content_string",
            {**baseline, "input": [{"role": "user", "content": BASELINE_INPUT}]},
        ),
        _shape(
            "instructions_plus_input",
            {**baseline, "instructions": INSTRUCTIONS, "input": QUERY},
        ),
        _shape(
            "no_max_output_tokens",
            {key: value for key, value in baseline.items() if key != "max_output_tokens"},
        ),
        _shape("higher_output_tokens", {**baseline, "max_output_tokens": 2500}),
        _shape("search_context_size_low", {**baseline, "search_context_size": "low"}),
        _shape("search_context_size_medium", {**baseline, "search_context_size": "medium"}),
        _shape("search_context_size_high", {**baseline, "search_context_size": "high"}),
        _shape(
            "experimental_tool_choice_required",
            {**baseline, "tool_choice": "required"},
            experimental=True,
        ),
        _shape(
            "experimental_tool_choice_object",
            {**baseline, "tool_choice": {"type": "web_search"}},
            experimental=True,
        ),
    ]


def probe_shapes(
    *,
    base_url: str,
    api_key: str,
    model: str,
    http_client: Any,
    timeout: float = DEFAULT_TIMEOUT,
    shape_names: list[str] | None = None,
    skip_experimental_tool_choice: bool = False,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not model.lower().startswith("grok-"):
        raise ValueError("GROK_SEARCH_MODEL must be a direct grok-* model")

    endpoint = normalize_responses_endpoint(base_url)
    shapes = select_request_shapes(
        build_request_shapes(model),
        shape_names=shape_names,
        skip_experimental_tool_choice=skip_experimental_tool_choice,
    )
    results = []
    for shape in shapes:
        if event_sink is not None:
            event_sink(
                {
                    "event": "start",
                    "shape": shape["name"],
                    "experimental_tool_choice": shape["experimental_tool_choice"],
                }
            )
        result = probe_shape(
            shape,
            endpoint=endpoint,
            api_key=api_key,
            http_client=http_client,
            timeout=timeout,
        )
        results.append(result)
        if event_sink is not None:
            event_sink(result)
    return {"results": results, "summary": summarize_results(results)}


def select_request_shapes(
    shapes: list[dict[str, Any]],
    *,
    shape_names: list[str] | None = None,
    skip_experimental_tool_choice: bool = False,
) -> list[dict[str, Any]]:
    requested = set(shape_names or [])
    known = {shape["name"] for shape in shapes}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown shape name(s): {', '.join(unknown)}")

    selected = [shape for shape in shapes if not requested or shape["name"] in requested]
    if skip_experimental_tool_choice:
        selected = [shape for shape in selected if not shape["experimental_tool_choice"]]
    return selected


def probe_shape(
    shape: dict[str, Any],
    *,
    endpoint: str,
    api_key: str,
    http_client: Any,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = http_client.post(
            endpoint,
            json=shape["body"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "hermes-mcp-s-grok-search-cpa-probe",
            },
            timeout=timeout,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        payload, invalid_json_message = _response_payload(response, api_key)
        status_code = getattr(response, "status_code", None)
        result = _response_summary(shape, payload, status_code, duration_ms, api_key)
        if invalid_json_message:
            result["error"] = {"code": None, "message": invalid_json_message}
        return result
    except Exception as exc:  # noqa: BLE001 - each diagnostic shape must be isolated.
        return {
            "shape": shape["name"],
            "experimental_tool_choice": shape["experimental_tool_choice"],
            "http_status": None,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "response_status": None,
            "output_item_types": [],
            "web_search_call": False,
            "annotations_count": 0,
            "citations_count": 0,
            "trace_status": "missing",
            "text_url_count": 0,
            "text_urls": [],
            "answer_preview": "",
            "error": {"code": None, "message": redact_text(str(exc) or exc.__class__.__name__, api_key)},
        }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    usable_shapes: list[str] = []
    missing_trace_shapes: list[str] = []
    failed_shapes: list[str] = []
    recommended_shape = "none"
    for result in results:
        status_code = result.get("http_status")
        successful = isinstance(status_code, int) and 200 <= status_code < 300
        usable = successful and (
            result.get("trace_status") == "structured" or bool(result.get("text_url_count"))
        )
        if usable:
            usable_shapes.append(result["shape"])
            if recommended_shape == "none" and not result.get("experimental_tool_choice", False):
                recommended_shape = result["shape"]
        elif successful:
            missing_trace_shapes.append(result["shape"])
        else:
            failed_shapes.append(result["shape"])
    return {
        "usable_shapes": usable_shapes,
        "missing_trace_shapes": missing_trace_shapes,
        "failed_shapes": failed_shapes,
        "recommended_shape": recommended_shape,
    }


def redact_text(value: Any, api_key: str = "") -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return sanitize_error(text)


def sanitize_url(url: str, api_key: str = "") -> str:
    sanitized = url.replace(api_key, "[redacted]") if api_key else url
    try:
        parts = urlsplit(sanitized)
        query = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            safe_value = "[redacted]" if _is_sensitive_key(key) else redact_text(value, api_key)
            query.append((key, safe_value))
        return urlunsplit(
            (
                parts.scheme,
                _safe_netloc(parts),
                redact_text(parts.path, api_key),
                urlencode(query),
                redact_text(parts.fragment, api_key),
            )
        )
    except ValueError:
        return redact_text(sanitized, api_key)


def _shape(name: str, body: dict[str, Any], *, experimental: bool = False) -> dict[str, Any]:
    return {"name": name, "body": body, "experimental_tool_choice": experimental}


def _response_payload(response: Any, api_key: str) -> tuple[dict[str, Any], str | None]:
    try:
        payload = response.json()
    except (TypeError, ValueError):
        text = getattr(response, "text", "")
        return {}, redact_text(text or "Response was not valid JSON", api_key)[:500]
    if not isinstance(payload, dict):
        return {}, "Response JSON was not an object"
    return payload, None


def _response_summary(
    shape: dict[str, Any],
    payload: dict[str, Any],
    status_code: Any,
    duration_ms: int,
    api_key: str,
) -> dict[str, Any]:
    parsed = parse_response(payload)
    text_only = parse_response({"output_text": parsed["answer"]})
    text_urls = [
        sanitize_url(citation["url"], api_key)
        for citation in text_only["citations"]
        if isinstance(citation.get("url"), str)
    ]
    error = payload.get("error")
    return {
        "shape": shape["name"],
        "experimental_tool_choice": shape["experimental_tool_choice"],
        "http_status": status_code,
        "duration_ms": duration_ms,
        "response_status": redact_text(payload.get("status"), api_key) if payload.get("status") is not None else None,
        "output_item_types": _output_item_types(payload, api_key),
        "web_search_call": _contains_web_search_call(payload),
        "annotations_count": _count_list_items(payload, "annotations"),
        "citations_count": _count_list_items(payload, "citations"),
        "trace_status": parsed["trace_status"],
        "text_url_count": len(text_urls),
        "text_urls": text_urls[:5],
        "answer_preview": redact_text(parsed["answer"], api_key)[:500],
        "error": _error_summary(error, api_key),
    }


def _output_item_types(payload: dict[str, Any], api_key: str) -> list[str]:
    output = payload.get("output")
    if not isinstance(output, list):
        return []
    return [
        redact_text(item.get("type"), api_key)
        for item in output
        if isinstance(item, dict) and item.get("type") is not None
    ]


def _contains_web_search_call(value: Any) -> bool:
    if isinstance(value, dict):
        if str(value.get("type", "")).lower() == "web_search_call":
            return True
        if "web_search_call" in {str(key).lower() for key in value} and value.get("web_search_call"):
            return True
        return any(
            _contains_web_search_call(child)
            for child in value.values()
            if isinstance(child, (dict, list))
        )
    if isinstance(value, list):
        return any(_contains_web_search_call(item) for item in value)
    return False


def _count_list_items(value: Any, target_key: str) -> int:
    if isinstance(value, dict):
        count = 0
        for key, child in value.items():
            if str(key).lower() == target_key and isinstance(child, list):
                count += len(child)
            if isinstance(child, (dict, list)):
                count += _count_list_items(child, target_key)
        return count
    if isinstance(value, list):
        return sum(_count_list_items(item, target_key) for item in value)
    return 0


def _error_summary(error: Any, api_key: str) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, dict):
        return {
            "code": redact_text(error.get("code"), api_key) if error.get("code") is not None else None,
            "message": redact_text(error.get("message") or error.get("detail") or "Request failed", api_key)[:500],
        }
    return {"code": None, "message": redact_text(error, api_key)[:500]}


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS)


def _safe_netloc(parts: Any) -> str:
    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    return f"{hostname}:{parts.port}" if parts.port is not None else hostname


def _required_environment(environ: dict[str, str] | None = None) -> tuple[str, str, str]:
    source = os.environ if environ is None else environ
    names = ("GROK_SEARCH_BASE_URL", "GROK_SEARCH_API_KEY", "GROK_SEARCH_MODEL")
    values = tuple(source.get(name, "").strip() for name in names)
    missing = [name for name, value in zip(names, values) if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return values


def parse_shape_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        names.extend(name.strip() for name in value.split(",") if name.strip())
    return names


def emit_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe CPA Responses request-shape compatibility.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-shape HTTP timeout in seconds.")
    parser.add_argument(
        "--shape",
        action="append",
        help="Shape name to run. Repeat the option or use comma-separated names.",
    )
    parser.add_argument(
        "--skip-experimental-tool-choice",
        action="store_true",
        help="Skip shapes that set experimental tool_choice values.",
    )
    parser.add_argument(
        "--list-shapes",
        action="store_true",
        help="List available shapes without reading environment variables or sending HTTP requests.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    environ: dict[str, str] | None = None,
    client_factory: Callable[[], Any] = httpx.Client,
    output: Callable[[dict[str, Any]], None] = emit_json,
) -> int:
    args = build_argument_parser().parse_args(argv)
    shape_names = parse_shape_names(args.shape)
    available_shapes = build_request_shapes("grok-probe-placeholder")

    if args.list_shapes:
        for shape in available_shapes:
            output(
                {
                    "shape": shape["name"],
                    "experimental_tool_choice": shape["experimental_tool_choice"],
                }
            )
        return 0

    try:
        select_request_shapes(
            available_shapes,
            shape_names=shape_names,
            skip_experimental_tool_choice=args.skip_experimental_tool_choice,
        )
        base_url, api_key, model = _required_environment(environ)
        with client_factory() as client:
            report = probe_shapes(
                base_url=base_url,
                api_key=api_key,
                model=model,
                http_client=client,
                timeout=args.timeout,
                shape_names=shape_names,
                skip_experimental_tool_choice=args.skip_experimental_tool_choice,
                event_sink=output,
            )
    except ValueError as exc:
        output({"error": redact_text(exc)})
        return 2

    output({"summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
