#!/usr/bin/env python3
"""LLM-as-judge: score pipeline vs direct-gen (with-phase bench).

Stable settings: temperature=0, thinking=False, same scorer model for both.

Usage:
    python scripts/run_llm_judge_with_phase.py
    python scripts/run_llm_judge_with_phase.py --force
    BENCH_LIMIT=3 python scripts/run_llm_judge_with_phase.py

Outputs:
    outputs/bench_scoring_with_phase/scores.jsonl
    outputs/bench_scoring_with_phase/summary.json
"""
from __future__ import annotations

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
SCORES_JSONL = SCORING_OUT / "scores.jsonl"
SUMMARY_PATH = SCORING_OUT / "summary.json"

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


def discover_work(force: bool) -> list[tuple[str, str, str, Path]]:
    """Return (nct_id, pipeline_text, direct_text, trial_config_path)."""
    existing = load_jsonl(SCORES_JSONL) if not force else {}
    work: list[tuple[str, str, str, Path]] = []

    with open(DIRECT_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            nct = row["nct_id"]
            if nct in existing:
                continue
            direct = (row.get("generated_criteria") or "").strip()
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
            work.append((nct, pfile.read_text(encoding="utf-8").strip(), direct, cfg))

    work.sort(key=lambda x: x[0])
    if LIMIT:
        work = work[:LIMIT]
    return work


def score_one_pair(
    nct_id: str,
    pipeline_text: str,
    direct_text: str,
    trial_config_path: Path,
) -> dict[str, Any]:
    from criteria_agent.scorer import score_draft
    from shared.llm_client import new_scorer_client, resolve_scorer_llm_config
    from shared.trial_config import load_trial_config

    _, _, scorer_model = resolve_scorer_llm_config()
    client = new_scorer_client()
    config = load_trial_config(trial_config_path)

    result: dict[str, Any] = {
        "nct_id": nct_id,
        "scorer_model": scorer_model,
        "temperature": 0.0,
        "thinking": False,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        p = score_draft(
            client, config, pipeline_text,
            model=scorer_model, temperature=0.0, thinking=False,
        )
        result["pipeline_scores"] = p["scores"]
        result["pipeline_mean"] = round(_mean(p["scores"]), 3)
    except Exception as e:
        result["pipeline_scores"] = {}
        result["pipeline_mean"] = None
        result["pipeline_error"] = str(e)[:500]

    try:
        d = score_draft(
            client, config, direct_text,
            model=scorer_model, temperature=0.0, thinking=False,
        )
        result["direct_scores"] = d["scores"]
        result["direct_mean"] = round(_mean(d["scores"]), 3)
    except Exception as e:
        result["direct_scores"] = {}
        result["direct_mean"] = None
        result["direct_error"] = str(e)[:300]

    if result["pipeline_mean"] is not None and result["direct_mean"] is not None:
        result["delta"] = round(result["pipeline_mean"] - result["direct_mean"], 3)
        for k in RUBRIC_KEYS:
            result[f"delta_{k}"] = round(
                result["pipeline_scores"].get(k, 0.0) - result["direct_scores"].get(k, 0.0), 3
            )
    else:
        result["delta"] = None

    return result


def build_summary() -> dict[str, Any]:
    scores = load_jsonl(SCORES_JSONL)
    valid = [
        s for s in scores.values()
        if s.get("pipeline_mean") is not None and s.get("direct_mean") is not None
    ]
    n = len(valid)
    if n == 0:
        return {"n_trials": 0}

    pipeline_means = [s["pipeline_mean"] for s in valid]
    direct_means = [s["direct_mean"] for s in valid]
    deltas = [s["delta"] for s in valid]

    per_dim: dict[str, Any] = {}
    for k in RUBRIC_KEYS:
        pv = [s["pipeline_scores"].get(k, 0.0) for s in valid]
        dv = [s["direct_scores"].get(k, 0.0) for s in valid]
        dim_deltas = [p - d for p, d in zip(pv, dv)]
        per_dim[k] = {
            "pipeline_mean": round(float(np.mean(pv)), 3),
            "direct_mean": round(float(np.mean(dv)), 3),
            "delta": round(float(np.mean(dim_deltas)), 3),
            "win_rate_pipeline": round(float(np.mean([d > 0 for d in dim_deltas])), 3),
            "p_wilcoxon": round(wilcoxon_p(pv, dv), 6),
        }

    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = n - wins - losses

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
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
        "per_dimension": per_dim,
    }


def print_summary(summary: dict[str, Any]) -> None:
    if summary.get("n_trials", 0) == 0:
        print("[summary] no valid scores", flush=True)
        return

    n = summary["n_trials"]
    print(f"\n{'='*72}", flush=True)
    print(f"[LLM-JUDGE SUMMARY]  {n} trials  (temp=0, thinking=False)", flush=True)
    print(f"  Pipeline mean : {summary['pipeline_overall_mean']:.3f}", flush=True)
    print(f"  Direct   mean : {summary['direct_overall_mean']:.3f}", flush=True)
    print(f"  Delta         : {summary['overall_delta']:+.3f}", flush=True)
    print(f"  p (Wilcoxon)  : {summary['overall_p_wilcoxon']:.4f}", flush=True)
    print(f"  Win/Loss/Tie  : {summary['wins_pipeline']}/{summary['wins_direct']}/{summary['ties']}", flush=True)
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

    ap = argparse.ArgumentParser(description="LLM-as-judge for with-phase bench.")
    ap.add_argument("--force", action="store_true", help="Re-score all trials (ignore cache).")
    ap.add_argument("--summary-only", action="store_true", help="Rebuild summary from cached scores.")
    args = ap.parse_args()

    SCORING_OUT.mkdir(parents=True, exist_ok=True)

    if args.force and SCORES_JSONL.is_file():
        backup = SCORING_OUT / f"scores_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        SCORES_JSONL.rename(backup)
        print(f"[force] backed up existing scores to {backup.name}", flush=True)

    if args.summary_only:
        summary = build_summary()
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(summary)
        return

    work = discover_work(force=args.force)
    existing_n = len(load_jsonl(SCORES_JSONL))
    print(f"[plan] {existing_n} cached, {len(work)} pending, workers={SCORING_WORKERS}", flush=True)

    if not work:
        summary = build_summary()
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print_summary(summary)
        return

    t0 = time.time()
    if SCORING_WORKERS == 1:
        for i, (nct, ptxt, dtxt, cfg) in enumerate(work, 1):
            print(f"  [{i:03d}/{len(work)}] {nct} …", end=" ", flush=True)
            try:
                row = score_one_pair(nct, ptxt, dtxt, cfg)
                append_jsonl(SCORES_JSONL, row)
                print(
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
                ex.submit(score_one_pair, nct, ptxt, dtxt, cfg): nct
                for nct, ptxt, dtxt, cfg in work
            }
            for fut in as_completed(futs):
                nct = futs[fut]
                try:
                    row = fut.result()
                    append_jsonl(SCORES_JSONL, row)
                    print(
                        f"  [done] {nct}  pipeline={row.get('pipeline_mean')}  "
                        f"direct={row.get('direct_mean')}  delta={row.get('delta')}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  [done] {nct}  ERROR: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[scoring] done in {elapsed:.0f}s", flush=True)

    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
