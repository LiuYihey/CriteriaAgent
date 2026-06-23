"""Set-matching consistency evaluator for eligibility criteria.

Replaces LLM-based agreement scoring and BERTScore with embedding-based
set matching.  Uses BiomedNLI-BioBERT sentence embeddings to compute
greedy bipartite matching between expert and AI-generated criteria bullets,
then reports Precision / Recall / F1.

All bullets (inclusion + exclusion) are pooled together for a single
unified matching pass.  Section labels are tracked so that negation flips
are handled correctly:
  - Same-section match + negation flip  → match FAILURE  (not counted)
  - Cross-section match + negation flip → match SUCCESS  (valid inversion)

Usage:
    python scripts/run_consistency_eval.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── fixed paths ──────────────────────────────────────────────────────────────
FILTERED_TRIALS_DIR = ROOT / "CriteriaBench" / "final_bench_filtered" / "trials"
PIPELINE_OUT = ROOT / "outputs" / "bench_criteria"
DIRECT_GEN_JSONL = ROOT / "CriteriaBench" / "final_bench_filtered" / "generated_criteria.jsonl"
CONSISTENCY_OUT = ROOT / "outputs" / "bench_consistency"
CONSISTENCY_JSONL = CONSISTENCY_OUT / "consistency_scores.jsonl"
SUMMARY_PATH = CONSISTENCY_OUT / "summary.json"

# Embedding model — BiomedNLI-BioBERT fine-tuned on SNLI, best for clinical text
# Local path (downloaded from Gitee AI mirror); falls back to HuggingFace hub
_LOCAL_MODEL_PATH = ROOT / "models" / "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
EMBEDDING_MODEL = str(_LOCAL_MODEL_PATH) if _LOCAL_MODEL_PATH.is_dir() else "pritamdeka/BiomedNLI-BioBERT-SNLI"

# Cosine similarity threshold for a valid match
MATCH_THRESHOLD = 0.5

# ── negation keywords for flip detection ─────────────────────────────────────
_NEG_WORDS = frozenset({
    "no", "not", "without", "never", "neither", "nor", "none",
    "exclude", "excluded", "excluding", "absence", "deny", "denies",
    "denied", "refuse", "refused", "discontinue", "discontinued",
})
_POS_WORDS = frozenset({
    "currently", "active", "present", "history of", "has", "have",
    "receiving", "taking", "using", "diagnosed with", "confirmed",
})


# ── helpers ──────────────────────────────────────────────────────────────────
def parse_criteria_bullets(text: str) -> list[tuple[str, str]]:
    """Parse eligibility criteria into (bullet_text, section) pairs.

    Returns list of (bullet, "inclusion"|"exclusion") tuples.
    If no section headers found, all bullets are labelled "inclusion".
    """
    if not text:
        return []

    low = text.lower()

    # Normalize section headers: "Key Inclusion Criteria:" / "Key Exclusion Criteria:"
    # and "### Inclusion Criteria" / "### Exclusion Criteria"
    # Insert "### inclusion" / "### exclusion" markers so downstream logic works.
    import re as _re
    text = _re.sub(r"(?m)^Key\s+Inclusion\s+Criteria\s*:?\s*$", "### Inclusion Criteria", text, flags=_re.IGNORECASE)
    text = _re.sub(r"(?m)^Key\s+Exclusion\s+Criteria\s*:?\s*$", "### Exclusion Criteria", text, flags=_re.IGNORECASE)
    text = _re.sub(r"(?m)^Inclusion\s+Criteria\s*:?\s*$", "### Inclusion Criteria", text, flags=_re.IGNORECASE)
    text = _re.sub(r"(?m)^Exclusion\s+Criteria\s*:?\s*$", "### Exclusion Criteria", text, flags=_re.IGNORECASE)

    low = text.lower()
    inc_start = low.find("### inclusion")
    exc_start = low.find("### exclusion")

    if inc_start < 0 and exc_start < 0:
        return [(b, "inclusion") for b in _extract_bullets(text)]

    result: list[tuple[str, str]] = []

    if inc_start >= 0:
        inc_end = exc_start if exc_start > inc_start else len(text)
        for b in _extract_bullets(text[inc_start:inc_end]):
            result.append((b, "inclusion"))

    if exc_start >= 0:
        for b in _extract_bullets(text[exc_start:]):
            result.append((b, "exclusion"))

    return result


def _extract_bullets(text: str) -> list[str]:
    """Extract numbered or bulleted items from text."""
    items = re.findall(r"^\s*(?:\d+\.|[-*])\s*(.+?)$", text, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


def load_direct_gen() -> dict[str, str]:
    """Load direct-generation criteria keyed by nct_id."""
    result: dict[str, str] = {}
    if not DIRECT_GEN_JSONL.is_file():
        return result
    with open(DIRECT_GEN_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                nid = row.get("nct_id", "")
                gen = row.get("generated_criteria", "")
                if nid and gen:
                    result[nid] = gen
    return result


def load_expert_criteria() -> dict[str, str]:
    """Load expert eligibility criteria from trial JSONs."""
    result: dict[str, str] = {}
    for cfg_path in sorted(FILTERED_TRIALS_DIR.glob("*.json")):
        nct = cfg_path.stem
        try:
            obj = json.loads(cfg_path.read_text(encoding="utf-8"))
            ps = obj.get("protocolSection") or obj
            em = ps.get("eligibilityModule") or {}
            ec = (em.get("eligibilityCriteria") or "").strip()
            if ec:
                result[nct] = ec
        except Exception:
            continue
    return result


def load_jsonl(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not path.is_file():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                nid = row.get("nct_id", "")
                if nid:
                    result[nid] = row
    return result


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── negation polarity detection ─────────────────────────────────────────────
def _negation_count(text: str) -> int:
    """Count negation keywords in text (case-insensitive word boundaries)."""
    low = text.lower()
    tokens = set(re.findall(r"[a-z]+", low))
    # Also check multi-word positive indicators
    n_neg = sum(1 for w in _NEG_WORDS if w in tokens)
    n_pos = sum(1 for w in _POS_WORDS if w in low)  # substring match for phrases
    return n_neg - n_pos


def _check_negation_flip(expert_bullet: str, ai_bullet: str) -> bool:
    """Detect potential negation polarity flip between matched pair."""
    e_pol = _negation_count(expert_bullet)
    a_pol = _negation_count(ai_bullet)
    # Flip if polarities have opposite signs
    return (e_pol > 0 and a_pol < 0) or (e_pol < 0 and a_pol > 0)


# ── set matching core ────────────────────────────────────────────────────────
def compute_set_matching(
    expert_bullets: list[tuple[str, str]],
    ai_bullets: list[tuple[str, str]],
    encoder: Any,
    threshold: float = MATCH_THRESHOLD,
) -> dict[str, Any]:
    """Greedy set matching between expert and AI bullets (section-aware).

    All bullets are pooled regardless of section (inclusion/exclusion).
    After matching, negation flips are checked:
      - Same-section + negation flip  → match FAILS (excluded from matched set)
      - Cross-section + negation flip → match SUCCEEDS (valid inversion)

    Args:
        expert_bullets: [(text, section), ...] from parse_criteria_bullets
        ai_bullets:     [(text, section), ...]

    Returns:
        {
            "recall": float,     # fraction of expert bullets covered
            "precision": float,  # fraction of AI bullets that are matched
            "f1": float,
            "soft_recall": float,    # Σ(sim_i) / n_expert
            "soft_precision": float, # Σ(sim_i) / n_ai
            "soft_f1": float,
            "n_expert": int,
            "n_ai": int,
            "n_matched": int,
            "match_scores": list[float],  # cosine sim of each matched pair
        }
    """
    empty = {
        "recall": 0.0, "precision": 0.0, "f1": 0.0,
        "soft_recall": 0.0, "soft_precision": 0.0, "soft_f1": 0.0,
        "n_expert": len(expert_bullets), "n_ai": len(ai_bullets),
        "n_matched": 0, "match_scores": [],
    }
    if not expert_bullets or not ai_bullets:
        return empty

    e_texts = [b for b, _ in expert_bullets]
    a_texts = [b for b, _ in ai_bullets]

    # Encode all bullets
    expert_embs = encoder.encode(e_texts, convert_to_numpy=True, show_progress_bar=False)
    ai_embs = encoder.encode(a_texts, convert_to_numpy=True, show_progress_bar=False)

    # Normalize for cosine similarity
    expert_embs = expert_embs / (np.linalg.norm(expert_embs, axis=1, keepdims=True) + 1e-10)
    ai_embs = ai_embs / (np.linalg.norm(ai_embs, axis=1, keepdims=True) + 1e-10)

    # Cosine similarity matrix: (n_expert, n_ai)
    sim_matrix = expert_embs @ ai_embs.T

    # Greedy matching: process expert bullets by best-match score (highest first)
    matched_ai_indices: set[int] = set()
    match_scores: list[float] = []

    best_ai_per_expert = sim_matrix.argmax(axis=1)
    best_score_per_expert = sim_matrix[np.arange(len(expert_bullets)), best_ai_per_expert]
    expert_order = np.argsort(-best_score_per_expert)

    for ei in expert_order:
        scores_for_expert = sim_matrix[ei].copy()
        for used_j in matched_ai_indices:
            scores_for_expert[used_j] = -1.0

        best_j = int(scores_for_expert.argmax())
        best_s = float(scores_for_expert[best_j])

        if best_s >= threshold:
            # Negation flip check: same-section flip → reject this match
            e_section = expert_bullets[ei][1]
            a_section = ai_bullets[best_j][1]
            has_flip = _check_negation_flip(e_texts[ei], a_texts[best_j])

            if has_flip and e_section == a_section:
                # Same-section negation flip → match fails, skip
                continue

            matched_ai_indices.add(best_j)
            match_scores.append(best_s)

    n_matched = len(matched_ai_indices)
    n_expert = len(expert_bullets)
    n_ai = len(ai_bullets)

    recall = n_matched / n_expert if n_expert > 0 else 0.0
    precision = n_matched / n_ai if n_ai > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Soft metrics: weighted by actual similarity scores
    sum_sim = sum(match_scores)
    soft_recall = sum_sim / n_expert if n_expert > 0 else 0.0
    soft_precision = sum_sim / n_ai if n_ai > 0 else 0.0
    soft_f1 = (2 * soft_precision * soft_recall / (soft_precision + soft_recall)) if (soft_precision + soft_recall) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "soft_recall": round(soft_recall, 4),
        "soft_precision": round(soft_precision, 4),
        "soft_f1": round(soft_f1, 4),
        "n_expert": n_expert,
        "n_ai": n_ai,
        "n_matched": n_matched,
        "match_scores": [round(s, 4) for s in match_scores],
    }


# ── per-trial scoring ────────────────────────────────────────────────────────
def score_one_consistency(
    nct_id: str,
    pipeline_text: str,
    direct_text: str,
    expert_text: str,
    encoder: Any,
) -> dict[str, Any]:
    """Compute set-matching consistency for one trial.

    All bullets (inclusion + exclusion) are pooled for a single unified match.
    """
    # Parse: list of (bullet_text, section_label)
    expert_bullets = parse_criteria_bullets(expert_text)
    pipeline_bullets = parse_criteria_bullets(pipeline_text)
    direct_bullets = parse_criteria_bullets(direct_text)

    # Single unified matching
    pipeline_result = compute_set_matching(expert_bullets, pipeline_bullets, encoder)
    direct_result = compute_set_matching(expert_bullets, direct_bullets, encoder)

    # Bullet counts by section
    e_inc = sum(1 for _, s in expert_bullets if s == "inclusion")
    e_exc = sum(1 for _, s in expert_bullets if s == "exclusion")
    p_inc = sum(1 for _, s in pipeline_bullets if s == "inclusion")
    p_exc = sum(1 for _, s in pipeline_bullets if s == "exclusion")
    d_inc = sum(1 for _, s in direct_bullets if s == "inclusion")
    d_exc = sum(1 for _, s in direct_bullets if s == "exclusion")

    return {
        "nct_id": nct_id,
        "pipeline": pipeline_result,
        "direct": direct_result,
        "counts": {
            "expert_inclusion": e_inc, "expert_exclusion": e_exc,
            "pipeline_inclusion": p_inc, "pipeline_exclusion": p_exc,
            "direct_inclusion": d_inc, "direct_exclusion": d_exc,
        },
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Set-matching consistency evaluator")
    ap.add_argument("--model", default=EMBEDDING_MODEL, help="Sentence-transformer model")
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Cosine sim threshold")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Initialize encoder
    print(f"[init] Loading embedding model: {args.model}", flush=True)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.model)
    print("[init] Model loaded.", flush=True)

    # Load data
    direct_gen = load_direct_gen()
    expert_criteria = load_expert_criteria()
    print(f"[data] direct_gen={len(direct_gen)}, expert_criteria={len(expert_criteria)}", flush=True)

    # Build work items
    existing = load_jsonl(CONSISTENCY_JSONL)
    work: list[tuple[str, str, str, str]] = []  # (nct, pipeline, direct, expert)

    for nct in sorted(direct_gen):
        if nct in existing:
            continue
        pipeline_final = PIPELINE_OUT / nct / "criteria_final.md"
        if not pipeline_final.is_file():
            print(f"  [skip] {nct}: no pipeline output", flush=True)
            continue
        if nct not in expert_criteria:
            print(f"  [skip] {nct}: no expert criteria", flush=True)
            continue
        pipeline_text = pipeline_final.read_text(encoding="utf-8")
        work.append((nct, pipeline_text, direct_gen[nct], expert_criteria[nct]))

    if args.limit:
        work = work[:args.limit]

    print(f"[plan] {len(existing)} cached, {len(work)} pending", flush=True)

    CONSISTENCY_OUT.mkdir(parents=True, exist_ok=True)

    if not work:
        print("[done] nothing to score", flush=True)
        _build_summary()
        return

    for i, (nct, ptxt, dtxt, etxt) in enumerate(work, 1):
        print(f"  [{i:03d}/{len(work)}] {nct} …", end=" ", flush=True)
        try:
            row = score_one_consistency(nct, ptxt, dtxt, etxt, encoder)
            append_jsonl(CONSISTENCY_JSONL, row)
            p = row["pipeline"]
            d = row["direct"]
            print(f"pipeline(F1={p['f1']:.3f} sF1={p['soft_f1']:.3f} R={p['recall']:.3f} sR={p['soft_recall']:.3f})  "
                  f"direct(F1={d['f1']:.3f} sF1={d['soft_f1']:.3f} R={d['recall']:.3f} sR={d['soft_recall']:.3f})",
                  flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    print("\n[done] consistency scoring complete", flush=True)
    _build_summary()


def _build_summary() -> None:
    scores = load_jsonl(CONSISTENCY_JSONL)
    if not scores:
        print("[summary] no scores", flush=True)
        return

    n = len(scores)
    methods = ["pipeline", "direct"]
    metrics = ["recall", "precision", "f1", "soft_recall", "soft_precision", "soft_f1"]

    summary: dict[str, Any] = {
        "n_trials": n,
        "model": EMBEDDING_MODEL,
        "threshold": MATCH_THRESHOLD,
    }

    # Per-method aggregate R/P/F1
    for meth in methods:
        for metric in metrics:
            vals = [s.get(meth, {}).get(metric, 0.0) for s in scores.values()]
            summary[f"{meth}_{metric}_mean"] = round(float(np.mean(vals)), 4) if vals else 0.0
            summary[f"{meth}_{metric}_std"] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0

    # Delta (pipeline - direct) per metric
    for metric in metrics:
        p_vals = [s.get("pipeline", {}).get(metric, 0.0) for s in scores.values()]
        d_vals = [s.get("direct", {}).get(metric, 0.0) for s in scores.values()]
        deltas = [p - d for p, d in zip(p_vals, d_vals)]
        summary[f"delta_{metric}"] = round(float(np.mean(deltas)), 4) if deltas else 0.0

    CONSISTENCY_OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}", flush=True)
    print(f"[SUMMARY] {n} trials — Set Matching ({EMBEDDING_MODEL}, τ={MATCH_THRESHOLD})", flush=True)
    for meth in methods:
        r = summary.get(f"{meth}_recall_mean", 0)
        p = summary.get(f"{meth}_precision_mean", 0)
        f1 = summary.get(f"{meth}_f1_mean", 0)
        sr = summary.get(f"{meth}_soft_recall_mean", 0)
        sp = summary.get(f"{meth}_soft_precision_mean", 0)
        sf1 = summary.get(f"{meth}_soft_f1_mean", 0)
        print(f"  {meth:10s}  R={r:.3f}  P={p:.3f}  F1={f1:.3f}  sR={sr:.3f}  sP={sp:.3f}  sF1={sf1:.3f}", flush=True)
    for metric in metrics:
        d = summary.get(f"delta_{metric}", 0)
        print(f"  Δ({metric}) = {d:+.4f}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  -> {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
