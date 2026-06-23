"""Prompt templates for Vanilla RAG baseline (comparison experiment)."""

from __future__ import annotations

VANILLA_RAG_SYSTEM = """You are an experienced clinical trialist. Draft eligibility criteria (Inclusion and Exclusion) for this study in English.

Use the trial configuration and the retrieved evidence profile below. Do not invent trial-specific numerical thresholds unless clearly supported by the evidence or standard practice for this context.

Output:
### Inclusion Criteria
(bullets)

### Exclusion Criteria
(bullets)
"""


def build_profile_block(profile: dict) -> str:
    parts = []
    for key, label in (
        ("disease_info", "Disease information"),
        ("drug_info", "Drug information"),
        ("relevant_papers", "Relevant papers"),
        ("similar_trials", "Similar trials"),
    ):
        val = profile.get(key)
        if val and str(val).strip().lower() != "none":
            parts.append(f"## {label}\n{val}")
    return "\n\n".join(parts) if parts else "(empty profile)"


def build_user_prompt(title: str, arms: str, primary: str, profile: dict) -> str:
    return (
        f"**Clinical trial title:** {title}\n\n"
        f"**Arms and interventions:**\n{arms}\n\n"
        f"**Primary outcome (first listed):**\n{primary}\n\n"
        f"**Retrieved evidence profile:**\n{build_profile_block(profile)}"
    )
