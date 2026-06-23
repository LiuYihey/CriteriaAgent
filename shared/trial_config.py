"""Load and normalize trial configuration for generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.ctgov_format import format_arms_interventions, format_primary_outcomes


@dataclass
class TrialConfig:
    title: str
    arms_text: str
    primary_text: str
    nct_id: str | None = None
    trial_type: str | None = None
    phase: str | None = None

    def context_block(self) -> str:
        result = [f"**Clinical trial title:** {self.title}"]
        if self.trial_type:
            result.append(f"**Trial type (registry, designModule.studyType):** {self.trial_type}")
        if self.phase:
            result.append(f"**Trial phase (registry, designModule.phases):** {self.phase}")
        result.extend([
            f"**Arms and interventions:**\n{self.arms_text}",
            f"**Primary outcome (first listed):**\n{self.primary_text}"
        ])
        return "\n\n".join(result)


def trial_config_from_study(study: dict[str, Any]) -> TrialConfig:
    """Build TrialConfig from a ClinicalTrials.gov study record."""
    return _from_ctgov_study(study)


def _from_simple(obj: dict[str, Any]) -> TrialConfig:
    title = str(obj.get("title", "")).strip()
    arms = obj.get("arms") or obj.get("arms_text") or ""
    primary = obj.get("primary_outcome") or obj.get("primary_text") or ""
    trial_type = str(obj.get("trial_type") or obj.get("studyType") or "").strip() or None
    phase = str(obj.get("phase") or "").strip() or None
    if isinstance(arms, list):
        arms = "\n".join(str(x) for x in arms)
    if isinstance(primary, list):
        primary = "\n".join(str(x) for x in primary)
    if not title:
        raise ValueError("trial config must include 'title'")
    return TrialConfig(
        title=title,
        arms_text=str(arms).strip() or "(not provided)",
        primary_text=str(primary).strip() or "(not provided)",
        nct_id=str(obj.get("nct_id") or obj.get("nctId") or "").strip() or None,
        trial_type=trial_type,
        phase=phase,
    )


def _from_ctgov_study(study: dict[str, Any]) -> TrialConfig:
    ps = study.get("protocolSection") or study
    ident = ps.get("identificationModule") or {}
    title = str(ident.get("officialTitle") or ident.get("briefTitle") or "").strip()
    nct = str(ident.get("nctId") or "").strip() or None
    arms_mod = ps.get("armsInterventionsModule")
    outcomes_mod = ps.get("outcomesModule")
    design_mod = ps.get("designModule") or {}
    
    # Extract studyType
    trial_type = str(design_mod.get("studyType") or "").strip() or None
    # Extract phase(s)
    phases = design_mod.get("phases") or []
    if isinstance(phases, list):
        phase = ", ".join(str(p) for p in phases) if phases else None
    else:
        phase = str(phases).strip() or None
        
    return TrialConfig(
        title=title or "(untitled)",
        arms_text=format_arms_interventions(arms_mod),
        primary_text=format_primary_outcomes(outcomes_mod),
        nct_id=nct,
        trial_type=trial_type,
        phase=phase,
    )


def load_trial_config(path: str | Path) -> TrialConfig:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a JSON object")
    if "protocolSection" in obj:
        return _from_ctgov_study(obj)
    if "title" in obj:
        return _from_simple(obj)
    raise ValueError(
        f"{path}: expected {{title, arms, primary_outcome}} or ClinicalTrials.gov protocolSection JSON"
    )
