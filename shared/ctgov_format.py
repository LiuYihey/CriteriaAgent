"""Format ClinicalTrials.gov protocolSection fields for generation prompts."""

from __future__ import annotations

from typing import Any


def format_primary_outcomes(outcomes_module: dict[str, Any] | None) -> str:
    """Only the FIRST primary outcome (some trials list dozens of near-duplicate measures)."""
    if not outcomes_module:
        return "Primary outcomes not present in protocolSection.outcomesModule."
    rows = outcomes_module.get("primaryOutcomes") or []
    if not rows:
        return "Primary outcomes not present in protocolSection.outcomesModule."
    o = next((r for r in rows if isinstance(r, dict)), None)
    if o is None:
        return "Primary outcomes not present in protocolSection.outcomesModule."
    measure = str(o.get("measure", "")).strip()
    desc = str(o.get("description", "")).strip()
    timeframe = str(o.get("timeFrame", "")).strip()
    parts: list[str] = []
    if measure:
        parts.append(measure)
    if desc:
        parts.append(desc)
    if timeframe:
        parts.append(f"Time frame: {timeframe}")
    return "1. " + "\n   ".join(parts) if parts else "Primary outcomes not present in protocolSection.outcomesModule."


def format_arms_interventions(arms_mod: dict[str, Any] | None) -> str:
    """Arms and interventions from armsInterventionsModule (no truncation)."""
    if not arms_mod:
        return "Arms and interventions not present in protocolSection.armsInterventionsModule."
    chunks: list[str] = []
    arm_groups = arms_mod.get("armGroups") or []
    if arm_groups:
        ag_lines: list[str] = []
        for g in arm_groups:
            if not isinstance(g, dict):
                continue
            label = str(g.get("label") or g.get("armGroupLabel") or "").strip()
            desc = str(g.get("description", "")).strip()
            raw_names = g.get("interventionNames")
            if isinstance(raw_names, list):
                names = ", ".join(str(x).strip() for x in raw_names if x)
            else:
                names = str(raw_names or "").strip()
            bits: list[str] = []
            if label:
                bits.append(label)
            if names:
                bits.append(f"Intervention name(s): {names}")
            if desc:
                bits.append(desc)
            if bits:
                ag_lines.append(" — ".join(bits))
        if ag_lines:
            chunks.append("Arm groups:\n" + "\n".join(f"- {line}" for line in ag_lines))
    interventions = arms_mod.get("interventions") or []
    if interventions:
        iv_lines: list[str] = []
        for x in interventions:
            if isinstance(x, dict):
                t = str(x.get("type", "") or "?").strip()
                name = str(x.get("name", "")).strip()
                d = str(x.get("description", "")).strip()
                core = f"{t}: {name}" if name else t
                if d:
                    core = f"{core}\n  {d}"
                iv_lines.append(core)
            else:
                iv_lines.append(str(x))
        chunks.append("Interventions:\n" + "\n".join(f"- {line}" for line in iv_lines))
    if not chunks:
        return "Arms and interventions not present in protocolSection.armsInterventionsModule."
    return "\n\n".join(chunks)
