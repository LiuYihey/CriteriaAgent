#!/usr/bin/env python3
"""Pairwise LLM-as-judge: pipeline vs direct-gen in ONE call per trial.

Unlike ``run_llm_judge_with_phase.py`` (two independent scorer calls), this
script presents both drafts side-by-side under the same rubric and trial
config so the judge can compare directly.

Draft order is randomized (deterministic by nct_id hash) to mitigate
position bias; scores are mapped back to pipeline/direct before saving.

Usage:
    python scripts/run_llm_judge_pairwise_with_phase.py
    python scripts/run_llm_judge_pairwise_with_phase.py --force
    BENCH_LIMIT=3 python scripts/run_llm_judge_pairwise_with_phase.py

Outputs:
    outputs/bench_scoring_with_phase/pairwise_scores.jsonl
    outputs/bench_scoring_with_phase/pairwise_summary.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIRECT_JSONL = ROOT / "CriteriaBench" / "outputs" / "direct_gen_with_phase" / "generated_criteria.jsonl"
PIPELINE_DIR = ROOT / "outputs" / "bench_criteria_with_phase"
TRIALS_DIR = ROOT / "CriteriaBench" / "final_bench" / "trials"
SCORING_OUT = ROOT / "outputs" / "bench_scoring_with_phase"
SCORES_JSONL = SCORING_OUT / "pairwise_scores.jsonl"
SUMMARY_PATH = SCORING_OUT / "pairwise_summary.json"

RUBRIC_KEYS = ["safety", "efficacy", "recruitment"]
SCORING_WORKERS = max(1, int(os.environ.get("SCORING_WORKERS", "2")))
LIMIT = int(os.environ.get("BENCH_LIMIT", "0")) or None


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


def _mean(scores: dict[str, float]) -> float:
    vals = [float(scores.get(k, 0.0)) for k in RUBRIC_KEYS]
    return sum(vals) / len(vals) if vals else 0.0


def wilcoxon_p(p_vals: list[float], d_vals: list[float]) -> float:
    try:
        from scipy.stats import wilcoxon as wx

        diffs = np.array(p_vals) - np.array(d_vals)
        if np.allclose(diffs, 0):
            return float("nan")
        _, pval = wx(p_vals, d_vals, zero_method="wilcox", alternative="two-sided")
        return float(pval)
    except Exception:
        return float("nan")


def pipeline_is_draft_a(nct_id: str) -> bool:
    """Deterministic 50/50 assignment of pipeline to Draft A vs B."""
    h = int(hashlib.sha256(nct_id.encode()).hexdigest(), 16)
    return h % 2 == 0


def discover_work(force: bool) -> list[tuple[str, str, str, str, Path]]:
    """Return (nct_id, pipeline_text, direct_text, expert_text, trial_config_path)."""
    existing = load_jsonl(SCORES_JSONL) if not force else {}
    work: list[tuple[str, str, str, str, Path]] = []

    with open(DIRECT_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            nct = row["nct_id"]
            if nct in existing:
                continue
            direct = (row.get("generated_criteria") or "").strip()
            expert = (row.get("expert_eligibility_criteria") or "").strip()
            pfile = PIPELINE_DIR / nct / "criteria_final.md"
            cfg = TRIALS_DIR / f"{nct}.json"
            if not direct:
                print(f"  [skip] {nct}: no direct-gen text", flush=True)
                continue
            if not pfile.is_file():
                print(f"  [skip] {nct}: no pipeline criteria_final.md", flush=True)
                continue
            if not cfg.is_file():
                print(f"  [skip] {nct}: no trial config", flush=True)
                continue
            work.append((
                nct,
                pfile.read_text(encoding="utf-8").strip(),
                direct,
                expert,
                cfg,
            ))

    work.sort(key=lambda x: x[0])
    if LIMIT:
        work = work[:LIMIT]
    return work


def score_one_pairwise(
    nct_id: str,
    pipeline_text: str,
    direct_text: str,
    expert_text: str,
    trial_config_path: Path,
) -> dict[str, Any]:
    from criteria_agent.scorer import score_pairwise
    from shared.llm_client import new_scorer_client, resolve_scorer_llm_config
    from shared.trial_config import load_trial_config

    _, _, scorer_model = resolve_scorer_llm_config()
    client = new_scorer_client()
    config = load_trial_config(trial_config_path)

    pipe_first = pipeline_is_draft_a(nct_id)
    if pipe_first:
        draft_a, draft_b = pipeline_text, direct_text
        label_a, label_b = "Draft A (Pipeline)", "Draft B (Direct generation)"
    else:
        draft_a, draft_b = direct_text, pipeline_text
        label_a, label_b = "Draft A (Direct generation)", "Draft B (Pipeline)"

    result: dict[str, Any] = {
        "nct_id": nct_id,
        "mode": "pairwise",
        "scorer_model": scorer_model,
        "temperature": 0.0,
        "thinking": False,
        "pipeline_as_draft_a": pipe_first,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }

    pw = score_pairwise(
        client,
        config,
        draft_a,
        draft_b,
        model=scorer_model,
        expert_criteria=expert_text or None,
        label_a=label_a,
        label_b=label_b,
        temperature=0.0,
        thinking=False,
    )

    if pipe_first:
        p_scores, d_scores = pw["scores_a"], pw["scores_b"]
    else:
        d_scores, p_scores = pw["scores_a"], pw["scores_b"]

    result["pipeline_scores"] = p_scores
    result["direct_scores"] = d_scores
    result["pipeline_mean"] = round(_mean(p_scores), 3)
    result["direct_mean"] = round(_mean(d_scores), 3)
    result["delta"] = round(result["pipeline_mean"] - result["direct_mean"], 3)
    for k in RUBRIC_KEYS:
        result[f"delta_{k}"] = round(p_scores.get(k, 0.0) - d_scores.get(k, 0.0), 3)
        result[f"winner_{k}"] = (
            "pipeline" if p_scores.get(k, 0.0) > d_scores.get(k, 0.0)
            else "direct" if d_scores.get(k, 0.0) > p_scores.get(k, 0.0)
            else "tie"
        )
    result["winner_overall"] = (
        "pipeline" if result["delta"] > 0
        else "direct" if result["delta"] < 0
        else "tie"
    )
    return result


def build_summary() -> dict[str, Any]:
    scores = load_jsonl(SCORES_JSONL)
    valid = [
        s for s in scores.values()
        if s.get("pipeline_mean") is not None and s.get("direct_mean") is not None
    ]
    n = len(valid)
    if n == 0:
        return {"n_trials": 0, "mode": "pairwise"}

    pipeline_means = [s["pipeline_mean"] for s in valid]
    direct_means = [s["direct_mean"] for s in valid]
    deltas = [s["delta"] for s in valid]

    per_dim: dict[str, Any] = {}
    for k in RUBRIC_KEYS:
        pv = [s["pipeline_scores"].get(k, 0.0) for s in valid]
        dv = [s["direct_scores"].get(k, 0.0) for s in valid]
        dim_deltas = [p - d for p, d in zip(pv, dv)]
        wins_p = sum(1 for s in valid if s.get(f"winner_{k}") == "pipeline")
        wins_d = sum(1 for s in valid if s.get(f"winner_{k}") == "direct")
        ties = n - wins_p - wins_d
        per_dim[k] = {
            "pipeline_mean": round(float(np.mean(pv)), 3),
            "direct_mean": round(float(np.mean(dv)), 3),
            "delta": round(float(np.mean(dim_deltas)), 3),
            "wins_pipeline": wins_p,
            "wins_direct": wins_d,
            "ties": ties,
            "win_rate_pipeline": round(wins_p / n, 3),
            "p_wilcoxon": round(wilcoxon_p(pv, dv), 6),
        }

    wins = sum(1 for s in valid if s.get("winner_overall") == "pipeline")
    losses = sum(1 for s in valid if s.get("winner_overall") == "direct")
    ties = n - wins - losses
    pipe_first_n = sum(1 for s in valid if s.get("pipeline_as_draft_a"))

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "mode": "pairwise",
        "scorer_settings": {"temperature": 0.0, "thinking": False},
        "n_trials": n,
        "pipeline_overall_mean": round(float(np.mean(pipeline_means)), 3),
        "direct_overall_mean": round(float(np.mean(direct_means)), 3),
        "overall_delta": round(float(np.mean(deltas)), 3),
        "overall_p_wilcoxon": round(wilcoxon_p(pipeline_means, direct_means), 6),
        "wins_pipeline": wins,
        "wins_direct": losses,
        "ties": ties,
        "win_rate_pipeline": round(wins / n, 3),
        "pipeline_as_draft_a_count": pipe_first_n,
        "per_dimension": per_dim,
    }


def print_summary(summary: dict[str, Any]) -> None:
    if summary.get("n_trials", 0) == 0:
        print("[summary] no valid pairwise scores", flush=True)
        return

    n = summary["n_trials"]
    print(f"\n{'='*72}", flush=True)
    print(f"[PAIRWISE LLM-JUDGE]  {n} trials  (1 call/trial, temp=0)", flush=True)
    print(f"  Pipeline mean : {summary['pipeline_overall_mean']:.3f}", flush=True)
    print(f"  Direct   mean : {summary['direct_overall_mean']:.3f}", flush=True)
    print(f"  Delta         : {summary['overall_delta']:+.3f}", flush=True)
    print(f"  p (Wilcoxon)  : {summary['overall_p_wilcoxon']:.4f}", flush=True)
    print(
        f"  Win/Loss/Tie  : {summary['wins_pipeline']}/{summary['wins_direct']}/{summary['ties']}",
        flush=True,
    )
    print(
        f"  Position bias check: pipeline shown as Draft A in "
        f"{summary['pipeline_as_draft_a_count']}/{n} trials",
        flush=True,
    )
    print(f"\n  {'dimension':<14}{'pipeline':>10}{'direct':>10}{'delta':>10}{'win%':>8}{'p':>10}", flush=True)
    for k in RUBRIC_KEYS:
        d = summary["per_dimension"][k]
        print(
            f"  {k:<14}{d['pipeline_mean']:>10.3f}{d['direct_mean']:>10.3f}"
            f"{d['delta']:>+10.3f}{d['win_rate_pipeline']*100:>7.0f}%"
            f"{d['p_wilcoxon']:>10.4f}",
            flush=True,
        )
    print(f"{'='*72}", flush=True)
    print(f"  -> {SUMMARY_PATH}", flush=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pairwise LLM-as-judge for with-phase bench.")
    ap.add_argument("--force", action="store_true", help="Re-score all trials (ignore cache).")
    ap.add_argument("--summary-only", action="store_true", help="Rebuild summary from cached scores.")
    args = ap.parse_args()

    SCORING_OUT.mkdir(parents=True, exist_ok=True)

    if args.force and SCORES_JSONL.is_file():
        backup = SCORING_OUT / f"pairwise_scores_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        SCORES_JSONL.rename(backup)
        print(f"[force] backed up existing scores to {backup.name}", flush=True)

    if args.summary_only:
        summary = build_summary()
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(summary)
        return

    work = discover_work(force=args.force)
    existing_n = len(load_jsonl(SCORES_JSONL))
    print(f"[plan] pairwise: {existing_n} cached, {len(work)} pending, workers={SCORING_WORKERS}", flush=True)

    if not work:
        summary = build_summary()
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(summary)
        return

    t0 = time.time()
    if SCORING_WORKERS == 1:
        for i, item in enumerate(work, 1):
            nct, ptxt, dtxt, etxt, cfg = item
            print(f"  [{i:03d}/{len(work)}] {nct} …", end=" ", flush=True)
            try:
                row = score_one_pairwise(nct, ptxt, dtxt, etxt, cfg)
                append_jsonl(SCORES_JSONL, row)
                print(
                    f"winner={row.get('winner_overall')}  "
                    f"pipeline={row.get('pipeline_mean')}  direct={row.get('direct_mean')}  "
                    f"delta={row.get('delta')}",
                    flush=True,
                )
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                traceback.print_exc()
    else:
        with ThreadPoolExecutor(max_workers=SCORING_WORKERS) as ex:
            futs = {
                ex.submit(score_one_pairwise, nct, ptxt, dtxt, etxt, cfg): nct
                for nct, ptxt, dtxt, etxt, cfg in work
            }
            for fut in as_completed(futs):
                nct = futs[fut]
                try:
                    row = fut.result()
                    append_jsonl(SCORES_JSONL, row)
                    print(
                        f"  [done] {nct}  winner={row.get('winner_overall')}  "
                        f"pipeline={row.get('pipeline_mean')}  direct={row.get('direct_mean')}  "
                        f"delta={row.get('delta')}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [done] {nct}  ERROR: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[pairwise scoring] done in {elapsed:.0f}s", flush=True)

    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
