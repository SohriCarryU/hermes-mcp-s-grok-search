from __future__ import annotations

from typing import Any


SEARCH_TRACE_KEYS = {"web_search_call", "search", "citations", "annotations", "url", "source"}


def parse_response(payload: dict[str, Any]) -> dict[str, Any]:
    answer = _extract_answer(payload)
    citations: list[dict[str, Any]] = []
    _collect_citations(payload, citations)
    has_search_trace = _has_search_trace(payload) or bool(citations)
    raw_summary = {
        "id": payload.get("id"),
        "object": payload.get("object"),
        "status": payload.get("status"),
        "output_count": len(payload.get("output") or []) if isinstance(payload.get("output"), list) else 0,
    }
    return {
        "answer": answer,
        "citations": _dedupe_citations(citations),
        "has_search_trace": has_search_trace,
        "raw_summary": {key: value for key, value in raw_summary.items() if value is not None},
    }


def _extract_answer(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type not in {None, "message"} and "content" not in item:
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    text = _content_text(content_item)
                    if text:
                        texts.append(text)
            else:
                text = _content_text(content)
                if text:
                    texts.append(text)
    return "\n\n".join(texts).strip()


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, dict):
        return ""
    for key in ("text", "output_text"):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _collect_citations(value: Any, citations: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        maybe = _citation_from_mapping(value)
        if maybe:
            citations.append(maybe)
        for key in ("annotations", "citations", "sources"):
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    _collect_citations(item, citations)
        for child in value.values():
            if isinstance(child, (dict, list)):
                _collect_citations(child, citations)
    elif isinstance(value, list):
        for item in value:
            _collect_citations(item, citations)


def _citation_from_mapping(value: dict[str, Any]) -> dict[str, Any]:
    url = _string_or_none(value.get("url")) or _string_or_none(value.get("uri"))
    source = _string_or_none(value.get("source"))
    title = _string_or_none(value.get("title"))
    snippet = _string_or_none(value.get("snippet")) or _string_or_none(value.get("text"))
    if not any((url, source, title)):
        return {}
    citation = {"title": title, "url": url, "snippet": snippet, "source": source}
    return {key: item for key, item in citation.items() if item}


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any]] = set()
    result: list[dict[str, Any]] = []
    for citation in citations:
        marker = (citation.get("url"), citation.get("title"), citation.get("source"))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(citation)
    return result


def _has_search_trace(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in SEARCH_TRACE_KEYS:
                return True
            if isinstance(child, str) and str(key).lower() == "type" and child in SEARCH_TRACE_KEYS:
                return True
            if _has_search_trace(child):
                return True
    elif isinstance(value, list):
        return any(_has_search_trace(item) for item in value)
    return False


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
