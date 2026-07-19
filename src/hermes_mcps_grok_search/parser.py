from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit


DOUBLE_BRACKET_LINK_PATTERN = re.compile(r"\[\[([^\]\r\n]+)\]\]\((https?://[^\s<>)]+)\)", re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"(?<!\[)\[([^\]\r\n]+)\]\((https?://[^\s<>)]+)\)", re.IGNORECASE)
BARE_URL_PATTERN = re.compile(r"https?://[^\s<>\]]+", re.IGNORECASE)


def parse_response(payload: dict[str, Any]) -> dict[str, Any]:
    answer = _extract_answer(payload)
    structured_citations: list[dict[str, Any]] = []
    _collect_citations(payload, structured_citations)
    structured_citations = _dedupe_citations(structured_citations)
    has_structured_trace = _has_structured_search_trace(payload) or bool(structured_citations)

    if has_structured_trace:
        citations = structured_citations
        trace_status = "structured"
    else:
        citations = _extract_text_citations(answer)
        trace_status = "text_citation_fallback" if citations else "missing"

    raw_summary = {
        "id": payload.get("id"),
        "object": payload.get("object"),
        "status": payload.get("status"),
        "output_count": len(payload.get("output") or []) if isinstance(payload.get("output"), list) else 0,
    }
    return {
        "answer": answer,
        "citations": citations,
        "has_search_trace": has_structured_trace,
        "trace_status": trace_status,
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
            if item_type not in {None, "message"}:
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
    seen_urls: set[str] = set()
    seen_other: set[tuple[Any, Any, Any]] = set()
    result: list[dict[str, Any]] = []
    for citation in citations:
        url = citation.get("url")
        if isinstance(url, str) and url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
        else:
            marker = (url, citation.get("title"), citation.get("source"))
            if marker in seen_other:
                continue
            seen_other.add(marker)
        result.append(citation)
    return result


def _extract_text_citations(answer: str) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for pattern in (DOUBLE_BRACKET_LINK_PATTERN, MARKDOWN_LINK_PATTERN):
        for match in pattern.finditer(answer):
            title = match.group(1).strip()
            _append_text_citation(citations, seen_urls, match.group(2), title)

    for match in BARE_URL_PATTERN.finditer(answer):
        url = _clean_url(match.group(0))
        host = urlsplit(url).hostname or ""
        _append_text_citation(citations, seen_urls, url, host)

    return citations


def _append_text_citation(
    citations: list[dict[str, Any]],
    seen_urls: set[str],
    url: str,
    title: str,
) -> None:
    cleaned_url = _clean_url(url)
    if not cleaned_url or cleaned_url in seen_urls:
        return
    seen_urls.add(cleaned_url)
    citation = {"url": cleaned_url}
    if title:
        citation["title"] = title
    citations.append(citation)


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?)]}'\"")


def _has_structured_search_trace(value: Any) -> bool:
    if isinstance(value, dict):
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type.lower() in {"web_search_call", "search"}:
            return True
        for key, child in value.items():
            lowered_key = str(key).lower()
            if lowered_key in {"web_search_call", "search"} and _is_non_empty(child):
                return True
            if lowered_key in {"annotations", "citations", "sources"} and _is_non_empty(child):
                return True
            if lowered_key in {"url", "uri"} and _is_http_url(child):
                return True
            if lowered_key == "source" and _is_non_empty(child):
                return True
            if isinstance(child, (dict, list)) and _has_structured_search_trace(child):
                return True
    elif isinstance(value, list):
        return any(_has_structured_search_trace(item) for item in value)
    return False


def _is_non_empty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _is_http_url(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith(("http://", "https://"))


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
