"""Independent scorer: scores criteria drafts using the formal rubric.

Returns structured ``dict`` via code-level JSON output control
(``response_format={"type": "json_object"}``).  Thinking mode is enabled
through ``AGENT_THINKING_BUDGET`` env var when the endpoint supports it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from criteria_agent.prompts import SCORER, SCORER_PAIRWISE
from criteria_agent.rubric import DIMENSION_KEYS
from shared.llm_client import call_json_openai
from shared.trial_config import TrialConfig

_SCORER_RUBRIC_PATH = Path(__file__).resolve().parent.parent / "Scorer rubric.md"


def _load_scorer_rubric() -> str:
    if not _SCORER_RUBRIC_PATH.is_file():
        raise FileNotFoundError(f"Scorer rubric not found: {_SCORER_RUBRIC_PATH}")
    return _SCORER_RUBRIC_PATH.read_text(encoding="utf-8")


def score_draft(
    client: Any,
    config: TrialConfig,
    draft: str,
    *,
    model: str | None = None,
    rubric_text: str | None = None,
    extra_context: str = "",
    temperature: float = 0.0,
    thinking: bool = False,
) -> dict[str, Any]:
    """Score a single draft.

    Returns ``{"scores": {dim: float, ...}, "raw_json": "..."}``.
    JSON output is enforced at the API level (``response_format``).
    """
    rubric = rubric_text or _load_scorer_rubric()
    ctx = config.context_block()
    if extra_context:
        ctx = f"{ctx}\n\n{extra_context}"

    parts = [
        f"Scoring rubric:\n---\n{rubric}\n---",
        f"\nTrial configuration:\n{ctx}",
    ]
    parts.append(f"\nEligibility criteria to score:\n---\n{draft}\n---")

    user = "\n".join(parts)
    raw = call_json_openai(
        client,
        system=SCORER,
        user=user,
        model=model,
        temperature=temperature,
        thinking=thinking,
    )

    # Extract thinking trace (injected by call_json_openai when available).
    thinking = raw.pop("_thinking", None)

    scores: dict[str, float] = {}
    raw_keys = set(raw.keys()) - {"_thinking"}
    expected_keys = {k for _, k in DIMENSION_KEYS}
    missing = expected_keys - raw_keys
    if missing:
        raise ValueError(
            f"Scorer response missing keys: {missing}. "
            f"Got keys: {raw_keys}. raw={json.dumps(raw, default=str)[:500]}"
        )

    for label, key in DIMENSION_KEYS:
        val = raw[key]
        # Unwrap dict-wrapped scores: {"safety": {"score": 7.5}} → 7.5
        if isinstance(val, dict):
            val = val.get("score", val.get("value"))
        scores[key] = float(val)  # int / float / str("7.5") all coerce cleanly

    result: dict[str, Any] = {"scores": scores, "raw_json": json.dumps(raw)}
    if thinking:
        result["thinking"] = thinking
    return result


def _parse_nested_scores(raw: dict[str, Any], prefix: str) -> dict[str, float]:
    """Extract dimension scores from ``{prefix: {dim: val, ...}}`` or flat keys."""
    block = raw.get(prefix)
    if block is None:
        # Fallback: flat keys like safety_a / safety_b
        scores: dict[str, float] = {}
        for _, key in DIMENSION_KEYS:
            val = raw.get(f"{key}_{prefix[-1]}")  # draft_a -> _a
            if val is not None:
                if isinstance(val, dict):
                    val = val.get("score", val.get("value"))
                scores[key] = float(val)
        if scores:
            return scores
        raise ValueError(f"Missing '{prefix}' block in pairwise response. keys={set(raw.keys())}")

    if not isinstance(block, dict):
        raise ValueError(f"Expected dict for '{prefix}', got {type(block).__name__}")

    scores = {}
    for _, key in DIMENSION_KEYS:
        val = block.get(key)
        if val is None:
            raise ValueError(f"Pairwise response missing {prefix}.{key}")
        if isinstance(val, dict):
            val = val.get("score", val.get("value"))
        scores[key] = float(val)
    return scores


def score_pairwise(
    client: Any,
    config: TrialConfig,
    draft_a: str,
    draft_b: str,
    *,
    model: str | None = None,
    rubric_text: str | None = None,
    expert_criteria: str | None = None,
    label_a: str = "Draft A",
    label_b: str = "Draft B",
    temperature: float = 0.0,
    thinking: bool = False,
) -> dict[str, Any]:
    """Score two drafts in one call for direct head-to-head comparison.

    Returns ``{"scores_a": {...}, "scores_b": {...}, "raw_json": "..."}``.
    """
    rubric = rubric_text or _load_scorer_rubric()
    ctx = config.context_block()

    parts = [
        f"Scoring rubric:\n---\n{rubric}\n---",
        f"\nTrial configuration:\n{ctx}",
    ]
    if expert_criteria:
        parts.append(
            f"\nExpert (human registry) criteria — calibration anchor only:\n---\n{expert_criteria}\n---"
        )
    parts.extend([
        f"\n{label_a}:\n---\n{draft_a}\n---",
        f"\n{label_b}:\n---\n{draft_b}\n---",
    ])

    user = "\n".join(parts)
    raw = call_json_openai(
        client,
        system=SCORER_PAIRWISE,
        user=user,
        model=model,
        temperature=temperature,
        thinking=thinking,
    )

    thinking_trace = raw.pop("_thinking", None)

    scores_a = _parse_nested_scores(raw, "draft_a")
    scores_b = _parse_nested_scores(raw, "draft_b")

    result: dict[str, Any] = {
        "scores_a": scores_a,
        "scores_b": scores_b,
        "raw_json": json.dumps(raw),
    }
    if thinking_trace:
        result["thinking"] = thinking_trace
    return result
