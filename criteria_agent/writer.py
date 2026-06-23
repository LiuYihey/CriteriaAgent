"""Criteria writer: integrate expert answers into eligibility criteria."""

from __future__ import annotations

import logging
from typing import Any

from criteria_agent.expert import ExpertResult
from criteria_agent.prompts import WRITER
from shared.llm_client import call_text
from shared.trial_config import TrialConfig

logger = logging.getLogger(__name__)

_WRITER_MAX_RETRIES = 2

_REQUIRED_SECTIONS = ("### Inclusion Criteria", "### Exclusion Criteria")

ROLE_LABEL = {
    "safety": "Safety Expert",
    "efficacy": "Efficacy Expert",
    "recruitment": "Recruitment Expert",
}


def _expert_block(results: list[ExpertResult]) -> str:
    parts = []
    for r in results:
        label = ROLE_LABEL.get(r.role, f"Subtask {r.subtask.index + 1}")
        parts.append(f"### {label}\n{r.l1l2_response}")
    return "\n\n".join(parts)


def _validate_criteria_structure(text: str) -> bool:
    """Return True iff text contains both required section headings."""
    return all(heading in text for heading in _REQUIRED_SECTIONS)


def write_initial_draft(
    client: Any,
    config: TrialConfig,
    expert_results: list[ExpertResult],
    *,
    model: str | None = None,
) -> str:
    user = (
        f"Trial configuration:\n{config.context_block()}\n\n"
        f"Expert answers:\n{_expert_block(expert_results)}\n\n"
        "Draft the inclusion and exclusion criteria now."
    )
    draft = call_text(client, system=WRITER, user=user, model=model)

    for attempt in range(1, _WRITER_MAX_RETRIES + 1):
        if _validate_criteria_structure(draft):
            return draft
        logger.warning(
            "Writer output missing required sections (attempt %d/%d) — retrying",
            attempt, _WRITER_MAX_RETRIES,
        )
        draft = call_text(client, system=WRITER, user=user, model=model)

    # Return last attempt even if structure is still incomplete
    if not _validate_criteria_structure(draft):
        logger.error("Writer output still missing sections after %d retries", _WRITER_MAX_RETRIES)
    return draft
