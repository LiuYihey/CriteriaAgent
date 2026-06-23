"""Shared LLM clients for CriteriaAgent and baselines.

Supports:
- Anthropic-compatible (MiniMax, etc.) via `new_client` / `call_text`
- OpenAI-compatible (DeepSeek, etc.) via `new_openai_client` / `call_text_openai`
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Pattern to match <think>...</think> blocks (MiniMax / DeepSeek inline thinking).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Pattern for unclosed <think> tag (no </think> before end of string).
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> tuple[str, str | None]:
    """Remove <think>…</think> blocks from *text*.

    Also handles unclosed <think> tags (model cut off before emitting </think>).
    Returns ``(cleaned_text, thinking_trace)`` where *thinking_trace* is the
    concatenated thinking content (or ``None`` if no block was found).
    """
    # First handle properly closed tags
    matches = _THINK_RE.findall(text)
    thinking_parts: list[str] = []
    if matches:
        for m in matches:
            inner_match = re.match(r"<think>(.*?)</think>", m, re.DOTALL | re.IGNORECASE)
            if inner_match:
                thinking_parts.append(inner_match.group(1).strip())
        text = _THINK_RE.sub("", text)

    # Then handle unclosed <think> (no closing tag — typically truncated response)
    unclosed = _THINK_UNCLOSED_RE.search(text)
    if unclosed:
        inner = unclosed.group(0)
        # Extract content after <think>
        inner_content = re.sub(r"^<think>", "", inner, flags=re.IGNORECASE).strip()
        if inner_content:
            thinking_parts.append(inner_content)
        text = _THINK_UNCLOSED_RE.sub("", text)

    cleaned = text.strip()
    trace = "\n".join(thinking_parts) if thinking_parts else None
    if not matches and not unclosed:
        return cleaned, None
    return cleaned, trace


def load_dotenv_simple() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def parse_api_key_file(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    base = re.search(r"ANTHROPIC_BASE_URL=(\S+)", text)
    key_m = re.search(r"API KEY:\s*(\S+)", text, re.I)
    model_m = re.search(r"MODEL:\s*(\S+)", text, re.I)
    if not key_m or not base or not model_m:
        raise RuntimeError(f"Could not parse BASE_URL, API KEY, MODEL from {path}")
    return base.group(1).rstrip("/"), key_m.group(1).strip(), model_m.group(1).strip()


def resolve_reviewer_llm_config() -> tuple[str, str, str]:
    """Parse reviewer_api.md → (base_url_openai, api_key, model_name)."""
    path = ROOT / "reviewer_api.md"
    if not path.is_file():
        raise RuntimeError("reviewer_api.md not found at project root")
    text = path.read_text(encoding="utf-8", errors="replace")
    base_m = re.search(r"base_url\s*\(OpenAI\)\s*=\s*(\S+)", text)
    key_m = re.search(r"api\s*key\s*=\s*(\S+)", text, re.I)
    model_m = re.search(r"model_name\s*=\s*(\S+)", text, re.I)
    if not base_m or not key_m or not model_m:
        raise RuntimeError(f"Could not parse base_url/api key/model_name from {path}")
    return base_m.group(1).rstrip("/"), key_m.group(1).strip(), model_m.group(1).strip()


def resolve_scorer_llm_config() -> tuple[str, str, str]:
    """Parse scorer_api.md → (base_url_openai, api_key, model_name)."""
    path = ROOT / "scorer_api.md"
    if not path.is_file():
        raise RuntimeError("scorer_api.md not found at project root")
    text = path.read_text(encoding="utf-8", errors="replace")
    base_m = re.search(r"base_url\s*\(OpenAI\)\s*=\s*(\S+)", text)
    key_m = re.search(r"api\s*key\s*=\s*(\S+)", text, re.I)
    model_m = re.search(r"model_name\s*=\s*(\S+)", text, re.I)
    if not base_m or not key_m or not model_m:
        raise RuntimeError(f"Could not parse base_url/api key/model_name from {path}")
    return base_m.group(1).rstrip("/"), key_m.group(1).strip(), model_m.group(1).strip()


def resolve_llm_config() -> tuple[str, str, str]:
    load_dotenv_simple()
    base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    model = os.environ.get("GRAPH_MODEL", "").strip() or os.environ.get("AGENT_MODEL", "").strip()
    if base and key and model:
        return base.rstrip("/"), key, model
    api_file = ROOT / "CriteriaBench" / "api_key.md"
    if api_file.is_file():
        return parse_api_key_file(api_file)
    raise RuntimeError("Set ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, AGENT_MODEL in .env or CriteriaBench/api_key.md")


def new_client() -> Any:
    import anthropic

    base_url, api_key, _ = resolve_llm_config()
    return anthropic.Anthropic(base_url=base_url, api_key=api_key)


def new_openai_client() -> Any:
    """Return an OpenAI client configured from reviewer_api.md."""
    from openai import OpenAI

    base_url, api_key, _ = resolve_reviewer_llm_config()
    return OpenAI(base_url=base_url, api_key=api_key)


def new_scorer_client() -> Any:
    """Create an OpenAI-compatible client for the scorer model."""
    from openai import OpenAI

    base_url, api_key, _ = resolve_scorer_llm_config()
    return OpenAI(base_url=base_url, api_key=api_key)


def call_text(
    client: Any,
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    import time as _time

    if model is None:
        _, _, model = resolve_llm_config()
    max_out = max_tokens or int(os.environ.get("AGENT_MAX_OUTPUT_TOKENS", "8192"))

    base_kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=max_out,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    max_retries = 3
    msg = None
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(**base_kwargs, temperature=temperature)
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                _time.sleep(2 * (attempt + 1))
            else:
                raise last_err from e

    def _text_from_msg(m: Any) -> str:
        return "\n".join(
            getattr(block, "text", "")
            for block in m.content
            if getattr(block, "type", None) == "text"
        ).strip()

    def _thinking_from_msg(m: Any) -> str | None:
        """Extract thinking block text if present."""
        parts = []
        for block in m.content:
            if getattr(block, "type", None) == "thinking":
                t = getattr(block, "thinking", None) or getattr(block, "text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts).strip() if parts else None

    text = _text_from_msg(msg)

    # If only thinking blocks were returned (no text), the model ran out of
    # output tokens on reasoning.  Retry with progressively larger budgets.
    if not text:
        thinking_trace = _thinking_from_msg(msg)
        for boost in (2, 4):
            boosted_kwargs = {**base_kwargs, "max_tokens": max_out * boost, "temperature": temperature}
            try:
                msg = client.messages.create(**boosted_kwargs)
            except Exception:
                break
            text = _text_from_msg(msg)
            if text:
                break
        # Last resort: if still no text but we captured thinking, return it
        # (caller can still parse useful content from the thinking trace).
        if not text and thinking_trace:
            return thinking_trace

    if not text:
        stop = getattr(msg, "stop_reason", "?")
        types = [getattr(b, "type", "?") for b in msg.content]
        raise RuntimeError(f"LLM returned no text (stop_reason={stop}, block_types={types})")
    return text


def call_text_openai(
    client: Any,
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    thinking: bool = False,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint.

    When *thinking* is ``True``, extended thinking is requested via
    ``extra_body`` using ``AGENT_THINKING_BUDGET``.  If the endpoint rejects
    it, the call is retried once without thinking (silent fallback).
    """
    if model is None:
        _, _, model = resolve_reviewer_llm_config()
    max_out = max_tokens or int(os.environ.get("AGENT_MAX_OUTPUT_TOKENS", "8192"))
    thinking_budget = int(os.environ.get("AGENT_THINKING_BUDGET", "0")) if thinking else 0

    kw: dict[str, Any] = dict(
        model=model,
        max_tokens=max_out,
        temperature=temperature if thinking_budget == 0 else 1.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if thinking_budget > 0:
        kw["extra_body"] = {
            "thinking": {"type": "adaptive", "budget_tokens": thinking_budget},
        }

    try:
        resp = client.chat.completions.create(**kw)
    except Exception:
        # Thinking not supported or transient error — retry without thinking.
        import time as _time
        if thinking_budget > 0:
            kw.pop("extra_body", None)
            kw["temperature"] = temperature
        _time.sleep(2)
        try:
            resp = client.chat.completions.create(**kw)
        except Exception:
            _time.sleep(4)
            resp = client.chat.completions.create(**kw)

    raw_text = (resp.choices[0].message.content or "").strip()
    if not raw_text:
        raise RuntimeError(f"OpenAI LLM returned no text (finish_reason={resp.choices[0].finish_reason})")
    text, _thinking = _strip_think(raw_text)
    if not text:
        raise RuntimeError(f"OpenAI LLM returned only thinking, no text content")
    return text


def _extract_json_from_text(text: str) -> str | None:
    """Try to extract a JSON object from text that may contain Markdown prose.

    Strategies:
    1. Find a ```json ... ``` or ``` ... ``` code fence containing valid JSON.
    2. Find the outermost { ... } via brace-matching.
    Returns the extracted JSON string, or None if nothing looks like JSON.
    """
    import json as _j
    # Strategy 1: code fences anywhere in text
    for m in re.finditer(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.DOTALL):
        candidate = m.group(1).strip()
        if candidate.startswith("{"):
            try:
                _j.loads(candidate)
                return candidate
            except _j.JSONDecodeError:
                pass
    # Strategy 2: outermost brace pair
    first = text.find("{")
    if first >= 0:
        depth = 0
        for i in range(first, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[first : i + 1]
                    try:
                        _j.loads(candidate)
                        return candidate
                    except _j.JSONDecodeError:
                        continue
    return None


def call_json_openai(
    client: Any,
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    thinking: bool = False,
) -> dict[str, Any]:
    """Call OpenAI endpoint with JSON output mode and optional thinking.

    Code-level format control via ``response_format={"type": "json_object"}``
    is always enforced — no plain-text fallback.
    When *thinking* is ``True``, thinking is enabled using ``AGENT_THINKING_BUDGET``;
    if the endpoint rejects it, silently retries without thinking.

    Returns the parsed JSON dict.
    """
    import json as _json

    if model is None:
        _, _, model = resolve_reviewer_llm_config()
    max_out = max_tokens or int(os.environ.get("AGENT_MAX_OUTPUT_TOKENS", "8192"))
    thinking_budget = int(os.environ.get("AGENT_THINKING_BUDGET", "0")) if thinking else 0

    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    kw: dict[str, Any] = dict(
        model=model,
        max_tokens=max_out,
        temperature=temperature,
        messages=msgs,
        response_format={"type": "json_object"},
    )
    if thinking_budget > 0:
        kw["extra_body"] = {
            "thinking": {"type": "adaptive", "budget_tokens": thinking_budget},
        }

    try:
        resp = client.chat.completions.create(**kw)
    except Exception:
        # Thinking not supported or transient error — retry.
        import time as _time
        if thinking_budget > 0:
            kw.pop("extra_body", None)
        _time.sleep(2)
        try:
            resp = client.chat.completions.create(**kw)
        except Exception:
            _time.sleep(4)
            resp = client.chat.completions.create(**kw)

    msg = resp.choices[0].message
    raw_text = (msg.content or "").strip()
    if not raw_text:
        raise RuntimeError(
            f"call_json_openai: empty response (finish_reason={resp.choices[0].finish_reason})"
        )
    # Strip inline <think> blocks that MiniMax injects into the content.
    text, think_trace = _strip_think(raw_text)
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    _fence = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
    m_fence = _fence.match(text)
    if m_fence:
        text = m_fence.group(1).strip()
    if not text:
        raise RuntimeError(
            f"call_json_openai: response contained only thinking, no JSON content. "
            f"raw_text[:200]={raw_text[:200]!r}"
        )
    try:
        result = _json.loads(text)
    except _json.JSONDecodeError:
        # Direct parse failed — model may have returned Markdown with embedded JSON.
        extracted = _extract_json_from_text(text)
        if extracted is not None:
            result = _json.loads(extracted)
        else:
            # Last resort: retry the API call once (model may behave differently).
            import time as _time
            _time.sleep(3)
            try:
                resp2 = client.chat.completions.create(**kw)
            except Exception:
                raise RuntimeError(
                    f"call_json_openai: JSON parse failed and retry also failed. "
                    f"cleaned_text[:300]={text[:300]!r}"
                )
            raw2 = (resp2.choices[0].message.content or "").strip()
            text2, _ = _strip_think(raw2)
            m_fence2 = _fence.match(text2)
            if m_fence2:
                text2 = m_fence2.group(1).strip()
            try:
                result = _json.loads(text2)
            except _json.JSONDecodeError:
                extracted2 = _extract_json_from_text(text2)
                if extracted2 is not None:
                    result = _json.loads(extracted2)
                else:
                    raise RuntimeError(
                        f"call_json_openai: JSON parse failed even after retry. "
                        f"cleaned_text[:300]={text[:300]!r}"
                    )
    # Capture thinking trace: prefer separate reasoning_content, then inline.
    reasoning = getattr(msg, "reasoning_content", None) or (
        getattr(msg, "model_extra", {}) or {}
    ).get("reasoning_content")
    if reasoning:
        result["_thinking"] = reasoning
    elif think_trace:
        result["_thinking"] = think_trace
    return result
