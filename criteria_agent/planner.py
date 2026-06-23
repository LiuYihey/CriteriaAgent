"""Dynamic task planner for eligibility-criteria design."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from criteria_agent.prompts import PLANNER, SUBTASK_ROLES
from shared.llm_client import call_text
from shared.trial_config import TrialConfig


@dataclass
class Subtask:
    index: int
    question: str
    role: str = "safety"  # one of SUBTASK_ROLES


# ---------------------------------------------------------------------------
# Parser: numbered-list format (1. / 2. / 3.)
# ---------------------------------------------------------------------------

# Matches lines starting with a digit followed by a period/parenthesis/colon
_NUMBERED_RE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.+)$")


def parse_subtasks(planner_text: str) -> list[Subtask]:
    """Parse planner output into a list of Subtask objects.

    Expected format is three numbered items (1. / 2. / 3.), each containing
    a question paragraph.  The parser is tolerant of blank lines, leading
    markers (``-``, ``*``), and trailing whitespace, but raises ValueError
    if it cannot extract at least one numbered item — no silent fallback.
    """
    text = planner_text.strip()
    if not text:
        raise ValueError("Planner returned empty text; cannot parse subtasks.")

    lines = text.splitlines()

    # Collect (number, body_lines) sections
    sections: list[tuple[int, list[str]]] = []
    current_num: int | None = None
    current_body: list[str] = []

    for ln in lines:
        stripped = ln.strip()
        m = _NUMBERED_RE.match(stripped)
        if m:
            # Save previous section
            if current_num is not None:
                sections.append((current_num, current_body))
            current_num = int(m.group(1))
            current_body = [m.group(2).strip()]
        elif current_num is not None and stripped:
            # Continuation line for the current numbered item
            current_body.append(stripped)

    # Save last section
    if current_num is not None:
        sections.append((current_num, current_body))

    if not sections:
        raise ValueError(
            f"Planner output contains no numbered items (1./2./3.). "
            f"Raw output:\n{text[:500]}"
        )

    result: list[Subtask] = []
    for idx, (_, body_lines) in enumerate(sections):
        question = " ".join(bl for bl in body_lines if bl).strip()
        if question:
            role = SUBTASK_ROLES[idx] if idx < len(SUBTASK_ROLES) else SUBTASK_ROLES[-1]
            result.append(Subtask(index=idx, question=question, role=role))

    if not result:
        raise ValueError(
            f"Planner output had numbered headers but all bodies were empty. "
            f"Raw output:\n{text[:500]}"
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan_subtasks(
    client: Any, config: TrialConfig, *, model: str | None = None,
) -> tuple[str, list[Subtask]]:
    user = (
        "Decompose eligibility-criteria design for this trial.\n\n"
        f"{config.context_block()}"
    )
    raw = call_text(client, system=PLANNER, user=user, model=model)
    return raw, parse_subtasks(raw)
