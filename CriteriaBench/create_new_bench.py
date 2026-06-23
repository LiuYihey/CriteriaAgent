"""Create new bench dataset from bench超集.json with strict filters.

Filters:
- overallStatus = COMPLETED
- studyType = INTERVENTIONAL
- startDate >= 2025-08
- Has at least one DRUG or BIOLOGICAL intervention
- Condition is a real disease (not healthy volunteers / education / etc.)
- Has non-empty eligibilityCriteria (ground truth)
- Phase is not NA or EARLY_PHASE1
- Criteria count is between 7 and 35 (inclusive)
"""
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPERSET = ROOT / "bench超集.json"
CRITERIA_BENCH = ROOT / "CriteriaBench"

# --- disease filter helpers ---
SKIP_CONDITIONS_EXACT = {
    "healthy", "healthy volunteers", "health", "healthy subjects",
    "healthy participants", "healthy participant", "healthy volunteer",
    "healthy adult male", "healthy adults", "healthy adult participants",
    "healthy adult", "healthy male volunteers", "healthy female volunteers",
    "healthy male adults", "healthy adult male subjects",
    "healthy male adult volunteers", "healthy adult female participants",
    "healthy volunteer study",
}
SKIP_CONDITIONS_CONTAINS = [
    "educational video", "persuasive video", "tailored video",
    "food purchasing", "grocery shopping",
    "pk in healthy",
]
# Conditions that look like disease keywords but are actually Phase I PK/bioequivalence in healthy people
SKIP_CONDITIONS_KEYWORDS = [
    "bioequivalence", "bioequivalance", "pharmacokinetic",
    "healthy participant", "healthy recruit",
    "fed condition", "food effect",
]

def is_disease_condition(conditions: list[str]) -> bool:
    """Return True if conditions describe a real disease/disorder.

    Rejects Phase I healthy-volunteer studies even when paired with
    procedure-like terms (topical anesthesia, sedation, etc.).
    A trial passes only if at least one condition is a clearly named
    disease, disorder, or impairment.
    """
    if not conditions:
        return False
    text = " ".join(conditions).strip().lower()
    if text in SKIP_CONDITIONS_EXACT:
        return False
    for skip in SKIP_CONDITIONS_CONTAINS:
        if skip in text:
            return False
    for kw in SKIP_CONDITIONS_KEYWORDS:
        if kw in text:
            return False

    # Identify which conditions are "non-disease" placeholders
    non_disease_words = {
        "healthy", "healthy volunteers", "healthy volunteer", "healthy adult",
        "patients", "subjects", "participants", "fed conditions",
        "healthy male adults", "healthy adult male subjects",
        "healthy male adult volunteers", "healthy adult female participants",
        "healthy volunteer study",
    }
    # Also treat procedure/situation terms as non-disease when alone with "healthy"
    procedure_terms = {
        "topical anesthesia", "sedation", "spinal anesthesia", "cryotherapy",
        "pediatric dentistry", "effect of drug", "dental plaque",
    }

    real_disease = []
    for c in conditions:
        cl = c.strip().lower()
        if cl in non_disease_words or cl in procedure_terms:
            continue
        # Check if it looks like a real disease/disorder/impairment
        real_disease.append(c)

    return len(real_disease) > 0


def has_drug_intervention(interventions: list[dict]) -> bool:
    return any(i.get("type") in ("DRUG", "BIOLOGICAL") for i in interventions)


def _count_criteria(text: str) -> tuple[int, int]:
    """Return (n_inclusion, n_exclusion) by splitting on numbered/bulleted lines."""
    if not text:
        return 0, 0
    parts = re.split(r"(?i)\bExclusion\b", text, maxsplit=1)
    inc_block = parts[0]
    exc_block = parts[1] if len(parts) > 1 else ""
    inc = [x for x in re.split(r"\n\s*(?:\d+\.|[-*])\s*", inc_block) if x.strip()]
    exc = [x for x in re.split(r"\n\s*(?:\d+\.|[-*])\s*", exc_block) if x.strip()]
    return len(inc), len(exc)


def is_valid_phase(phases: list[str]) -> bool:
    """Reject trials whose only phase is NA or that include EARLY_PHASE1."""
    if not phases:
        return False
    if "EARLY_PHASE1" in phases:
        return False
    if phases == ["NA"]:
        return False
    return True


def is_valid_criteria_count(eligibility_criteria: str) -> bool:
    """Keep trials with total criteria count in [7, 35]."""
    ni, ne = _count_criteria(eligibility_criteria)
    total = ni + ne
    return 7 <= total <= 35


def main():
    data = json.loads(SUPERSET.read_text(encoding="utf-8"))
    print(f"Superset size: {len(data)}")

    # --- Clean old bench outputs ---
    final_dir = CRITERIA_BENCH / "final_bench"
    for old_file in (CRITERIA_BENCH / "filtered_drug_trials.json",
                     CRITERIA_BENCH / "filtered_all_trials.json"):
        if old_file.exists():
            old_file.unlink()
            print(f"Removed old: {old_file.relative_to(ROOT)}")
    if final_dir.exists():
        shutil.rmtree(final_dir)
        print(f"Removed old: {final_dir.relative_to(ROOT)}/")

    filtered = []
    skip_reasons = {
        "status": 0, "study_type": 0, "date": 0, "drug": 0,
        "disease": 0, "criteria_empty": 0, "phase": 0, "criteria_count": 0,
    }

    for t in data:
        ps = t.get("protocolSection", {})
        sm = ps.get("statusModule", {})
        dm = ps.get("designModule", {})
        cm = ps.get("conditionsModule", {})
        aim = ps.get("armsInterventionsModule", {})
        em = ps.get("eligibilityModule", {})

        # Filter 1: COMPLETED
        if sm.get("overallStatus") != "COMPLETED":
            skip_reasons["status"] += 1
            continue

        # Filter 2: INTERVENTIONAL
        if dm.get("studyType") != "INTERVENTIONAL":
            skip_reasons["study_type"] += 1
            continue

        # Filter 3: startDate >= 2025-08
        start_date = sm.get("startDateStruct", {}).get("date", "")
        if not start_date or start_date < "2025-08":
            skip_reasons["date"] += 1
            continue

        # Filter 4: DRUG/BIOLOGICAL intervention
        interventions = aim.get("interventions", [])
        if not has_drug_intervention(interventions):
            skip_reasons["drug"] += 1
            continue

        # Filter 5: Disease condition
        conditions = cm.get("conditions", [])
        if not is_disease_condition(conditions):
            skip_reasons["disease"] += 1
            continue

        # Filter 6: Has eligibility criteria (ground truth)
        ec = em.get("eligibilityCriteria", "").strip()
        if not ec or len(ec) < 50:
            skip_reasons["criteria_empty"] += 1
            continue

        # Filter 7: Phase not NA / EARLY_PHASE1
        phases = dm.get("phases", [])
        if not is_valid_phase(phases):
            skip_reasons["phase"] += 1
            continue

        # Filter 8: Criteria count in [7, 35]
        if not is_valid_criteria_count(ec):
            skip_reasons["criteria_count"] += 1
            continue

        filtered.append(t)

    print(f"\nSkip reasons:")
    for reason, count in skip_reasons.items():
        print(f"  {reason}: {count}")
    print(f"\nFiltered: {len(filtered)} trials")

    if not filtered:
        print("ERROR: No trials passed filters!")
        return

    # Sort by NCT ID
    filtered.sort(key=lambda t: t["protocolSection"]["identificationModule"]["nctId"])

    # Create final_bench structure (trials/ directory with individual JSONs)
    final_dir = CRITERIA_BENCH / "final_bench"
    trials_dir = final_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    for t in filtered:
        nct = t["protocolSection"]["identificationModule"]["nctId"]
        (trials_dir / f"{nct}.json").write_text(
            json.dumps(t, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Save combined final_bench.json
    (final_dir / "final_bench.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Summary
    # Phase distribution
    from collections import Counter
    phase_c = Counter()
    cond_c = Counter()
    drug_c = Counter()
    for t in filtered:
        ps = t["protocolSection"]
        for p in ps.get("designModule", {}).get("phases", []):
            phase_c[p] += 1
        for c in ps.get("conditionsModule", {}).get("conditions", []):
            cond_c[c] += 1
        for i in ps.get("armsInterventionsModule", {}).get("interventions", []):
            if i.get("type") in ("DRUG", "BIOLOGICAL"):
                drug_c[i.get("name", "?")] += 1

    summary = {
        "n_trials": len(filtered),
        "filters": {
            "status": "COMPLETED",
            "study_type": "INTERVENTIONAL",
            "start_date_after": "2025-08",
            "intervention_type": "DRUG or BIOLOGICAL",
            "condition": "disease/disorder (not healthy/education)",
            "has_eligibility_criteria": True,
            "phase": "not NA, not EARLY_PHASE1",
            "criteria_count": "7 <= total <= 35",
        },
        "phase_distribution": dict(phase_c.most_common()),
        "nct_ids": [t["protocolSection"]["identificationModule"]["nctId"] for t in filtered],
    }
    (final_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created {final_dir.relative_to(ROOT)} (trials/, final_bench.json, summary.json)")

    print(f"\nPhase distribution:")
    for p, n in phase_c.most_common():
        print(f"  {p}: {n}")
    print(f"\nTop 20 conditions:")
    for c, n in cond_c.most_common(20):
        print(f"  {c}: {n}")
    print(f"\nTop 20 drugs:")
    for d, n in drug_c.most_common(20):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    main()
