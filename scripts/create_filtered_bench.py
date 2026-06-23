#!/usr/bin/env python3
"""Extract a filtered CriteriaBench containing only trials where expert criteria
count is in [5, 30].  Copies generated_criteria.jsonl, agreements.jsonl,
reviews.jsonl, final_bench.json, trial configs, summary, and scoring rule."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "CriteriaBench" / "final_bench"
DST = ROOT / "CriteriaBench" / "final_bench_filtered"

LO, HI = 5, 30  # expert total range to keep

V0_DIR = ROOT / "outputs" / "bench_criteria"


# ── count helpers (same as compare_criteria_counts.py) ──────────────────────

def count_criteria_items(text: str) -> tuple[int, int, int]:
    if not text or not text.strip():
        return 0, 0, 0
    inc_text, exc_text = "", ""
    for inc_pat, exc_pat in [
        (r"(?i)(?:###?\s*)?inclusion\s+criteria[:\s]*\n(.*?)(?=(?:###?\s*)?exclusion\s+criteria|$)",
         r"(?i)(?:###?\s*)?exclusion\s+criteria[:\s]*\n(.*)"),
    ]:
        m_i = re.search(inc_pat, text, re.S)
        m_e = re.search(exc_pat, text, re.S)
        if m_i:
            inc_text = m_i.group(1).strip()
        if m_e:
            exc_text = m_e.group(1).strip()
    if not inc_text and not exc_text:
        inc_text = text

    def _count(block: str) -> int:
        if not block.strip():
            return 0
        count = 0
        for line in block.strip().splitlines():
            s = line.strip()
            if not s:
                continue
            if re.match(r"^\d+[\.\)]\s", s):
                count += 1
            elif re.match(r"^[-*•]\s", s):
                count += 1
        if count == 0:
            for line in block.strip().splitlines():
                s = line.strip()
                if s and not re.match(r"^(?:inclusion|exclusion)\s+criteria", s, re.I):
                    count += 1
        return count

    return _count(inc_text), _count(exc_text), _count(inc_text) + _count(exc_text)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Source : {SRC}")
    print(f"Target : {DST}")
    print(f"Filter : expert_total in [{LO}, {HI}]\n")

    # 1. Read generated_criteria.jsonl and determine which NCT IDs to keep
    jsonl_path = SRC / "generated_criteria.jsonl"
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"Total rows in generated_criteria.jsonl: {len(rows)}")

    keep_ids: set[str] = set()
    skip_rows = []
    for r in rows:
        inc, exc, total = count_criteria_items(r.get("expert_eligibility_criteria", ""))
        nid = r["nct_id"]
        # Must be in expert range
        if not (LO <= total <= HI):
            skip_rows.append((nid, f"expert_total={total}"))
            continue
        # Must have non-empty direct generation
        if not r.get("generated_criteria", "").strip():
            skip_rows.append((nid, "no direct generation"))
            continue
        # Must have v0 output
        v0_file = V0_DIR / nid / "criteria_v0.md"
        if not v0_file.exists():
            skip_rows.append((nid, "no v0 output"))
            continue
        keep_ids.add(nid)

    print(f"Keeping {len(keep_ids)} trials, skipping {len(skip_rows)}\n")
    if skip_rows:
        print("Skipped trials:")
        for nid, reason in sorted(skip_rows, key=lambda x: x[0]):
            print(f"  {nid}: {reason}")

    # 2. Create target directory
    DST.mkdir(parents=True, exist_ok=True)

    # 3. Write filtered generated_criteria.jsonl
    kept_rows = [r for r in rows if r["nct_id"] in keep_ids]
    out_jsonl = DST / "generated_criteria.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(kept_rows)} rows -> {out_jsonl}")

    # 4. Copy trial JSON files
    src_trials = SRC / "trials"
    dst_trials = DST / "trials"
    dst_trials.mkdir(exist_ok=True)
    copied = 0
    for nid in keep_ids:
        src_f = src_trials / f"{nid}.json"
        if src_f.exists():
            shutil.copy2(src_f, dst_trials / f"{nid}.json")
            copied += 1
    print(f"Copied {copied} trial JSON files -> {dst_trials}")

    # 5. Filter final_bench.json
    fb_path = SRC / "final_bench.json"
    if fb_path.exists():
        with open(fb_path, encoding="utf-8") as f:
            fb_data = json.load(f)
        fb_filtered = []
        for item in fb_data:
            nct_id = item.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
            if nct_id in keep_ids:
                fb_filtered.append(item)
        out_fb = DST / "final_bench.json"
        with open(out_fb, "w", encoding="utf-8") as f:
            json.dump(fb_filtered, f, ensure_ascii=False, indent=2)
        print(f"Filtered final_bench.json: {len(fb_data)} -> {len(fb_filtered)} -> {out_fb}")

    # 6. Filter agreements.jsonl
    agr_path = SRC / "agreements.jsonl"
    if agr_path.exists():
        with open(agr_path, encoding="utf-8") as f:
            agr_rows = [json.loads(l) for l in f if l.strip()]
        agr_kept = [r for r in agr_rows if r.get("nct_id") in keep_ids]
        out_agr = DST / "agreements.jsonl"
        with open(out_agr, "w", encoding="utf-8") as f:
            for r in agr_kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Filtered agreements.jsonl: {len(agr_rows)} -> {len(agr_kept)} -> {out_agr}")

    # 7. Filter reviews.jsonl
    rev_path = SRC / "reviews.jsonl"
    if rev_path.exists():
        with open(rev_path, encoding="utf-8") as f:
            rev_rows = [json.loads(l) for l in f if l.strip()]
        rev_kept = [r for r in rev_rows if r.get("nct_id") in keep_ids]
        out_rev = DST / "reviews.jsonl"
        with open(out_rev, "w", encoding="utf-8") as f:
            for r in rev_kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Filtered reviews.jsonl: {len(rev_rows)} -> {len(rev_kept)} -> {out_rev}")

    # 8. Copy agreement_scoring_rule.md
    rule_src = SRC / "agreement_scoring_rule.md"
    if rule_src.exists():
        shutil.copy2(rule_src, DST / "agreement_scoring_rule.md")
        print(f"Copied agreement_scoring_rule.md")

    # 9. Copy figures/
    fig_src = SRC / "figures"
    if fig_src.exists():
        fig_dst = DST / "figures"
        if fig_dst.exists():
            shutil.rmtree(fig_dst)
        shutil.copytree(fig_src, fig_dst)
        print(f"Copied figures/")

    # 10. Write updated summary.json
    # Recompute mean rubric scores from reviews
    rubric_keys = ["signal_enrichment", "population_reach", "data_evaluability",
                   "clinical_feasibility", "risk_mitigation"]
    rubric_means = {}
    if rev_kept:
        for k in rubric_keys:
            vals = [r["scores"][k] for r in rev_kept if k in r.get("scores", {})]
            rubric_means[k] = sum(vals) / len(vals) if vals else 0.0

    agreement_keys = ["inclusion_coverage", "exclusion_coverage", "quantitative_alignment"]
    agr_means = {}
    if agr_kept:
        for k in agreement_keys:
            vals = [r[k]["score"] for r in agr_kept if k in r and isinstance(r[k], dict) and "score" in r[k]]
            agr_means[k] = sum(vals) / len(vals) if vals else 0.0

    summary = {
        "model": "MiniMax-M2.7",
        "n_trials": len(kept_rows),
        "expert_criteria_filter": f"[{LO}, {HI}]",
        "mean_rubric_scores_1_to_10": rubric_means,
        "mean_agreement_subscores_0_to_10": agr_means,
        "excluded_nct_ids": {nid: reason for nid, reason in sorted(skip_rows, key=lambda x: x[0])},
    }
    out_sum = DST / "summary.json"
    with open(out_sum, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nUpdated summary.json -> {out_sum}")
    print(f"  n_trials: {summary['n_trials']}")
    print(f"  rubric means: {rubric_means}")
    print(f"  agreement means: {agr_means}")

    print(f"\n{'='*60}")
    print(f"Done! Filtered bench at: {DST}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
