#!/usr/bin/env python3
"""Run full graph pipeline + unified scoring comparison for filtered bench trials.

Phase A  – Run the criteria-agent pipeline (planner → experts → writer)
           for every filtered trial.  Outputs ``criteria_final.md`` per trial.
Phase B  – Score **pipeline** vs **direct-generation** criteria using the
           SAME scorer model for both, ensuring fair comparison.

Usage:
    # Run both phases (default):
    python scripts/run_bench.py

    # Pipeline only:
    python scripts/run_bench.py --phase pipeline

    # Scoring only (assumes pipeline outputs already exist):
    python scripts/run_bench.py --phase score

Env knobs:
    PIPELINE_WORKERS   – concurrent pipeline workers  (default 1)
    SCORING_WORKERS    – concurrent scoring workers    (default 2)
    PIPELINE_FORCE     – 1/true to re-run even if output exists
    BENCH_LIMIT        – process only first N trials (smoke test)
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── fixed paths ──────────────────────────────────────────────────────────────
FILTERED_TRIALS_DIR = ROOT / "CriteriaBench" / "final_bench_filtered" / "trials"
GRAPHS_DIR          = ROOT / "outputs" / "bench_graphs" / "graphs"
PIPELINE_OUT        = ROOT / "outputs" / "bench_criteria"
SCORING_OUT         = ROOT / "outputs" / "bench_scoring"
DIRECT_GEN_JSONL    = ROOT / "CriteriaBench" / "final_bench_filtered" / "generated_criteria.jsonl"
SCORES_JSONL        = SCORING_OUT / "scores.jsonl"
SUMMARY_PATH        = SCORING_OUT / "summary.json"

PIPELINE_WORKERS = max(1, int(os.environ.get("PIPELINE_WORKERS", "1")))
SCORING_WORKERS  = max(1, int(os.environ.get("SCORING_WORKERS", "2")))
FORCE            = os.environ.get("PIPELINE_FORCE", "").lower() in ("1", "true", "yes")
LIMIT            = int(os.environ.get("BENCH_LIMIT", "0")) or None

RUBRIC_KEYS = [
    "safety",
    "efficacy",
    "recruitment",
]


# ── helpers ──────────────────────────────────────────────────────────────────
def discover_trials() -> list[tuple[str, Path, Path]]:
    """Return sorted (nct_id, config_path, graph_path) for all filtered trials."""
    pairs: list[tuple[str, Path, Path]] = []
    for cfg in sorted(FILTERED_TRIALS_DIR.glob("*.json")):
        nct = cfg.stem
        graph = GRAPHS_DIR / f"{nct}_graph.json"
        if graph.is_file():
            pairs.append((nct, cfg, graph))
    return pairs


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


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load JSONL keyed by nct_id."""
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
    vals = [scores.get(k, 0.0) for k in RUBRIC_KEYS]
    return sum(vals) / len(vals) if vals else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Phase A: Graph Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_one_pipeline(nct_id: str, cfg_path: Path, graph_path: Path) -> dict:
    """Run the full criteria-agent pipeline for one trial."""
    from criteria_agent.pipeline import run_pipeline

    out_dir = PIPELINE_OUT / nct_id
    final_path = out_dir / "criteria_final.md"

    if final_path.is_file() and not FORCE:
        return {"nct_id": nct_id, "status": "skipped_exists", "output_dir": str(out_dir)}

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        trace = run_pipeline(
            graph_path=graph_path,
            config_path=cfg_path,
            output_dir=out_dir,
        )
        elapsed = time.time() - t0
        return {
            "nct_id": nct_id,
            "status": "ok",
            "elapsed_s": round(elapsed, 1),
            "output_dir": str(out_dir),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        return {
            "nct_id": nct_id,
            "status": "error",
            "elapsed_s": round(elapsed, 1),
            "error": str(exc)[:500],
            "traceback": traceback.format_exc()[-1500:],
            "output_dir": str(out_dir),
        }


def phase_pipeline(trials: list[tuple[str, Path, Path]]) -> None:
    """Run the graph pipeline for all filtered trials."""
    PIPELINE_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[phase A] pipeline: {len(trials)} trials, workers={PIPELINE_WORKERS}, force={FORCE}", flush=True)

    t_start = time.time()
    rows: list[dict] = []

    if PIPELINE_WORKERS == 1:
        for i, (nct, cfg, grp) in enumerate(trials, 1):
            print(f"  [{i:03d}/{len(trials)}] {nct} …", end=" ", flush=True)
            row = run_one_pipeline(nct, cfg, grp)
            rows.append(row)
            print(f"{row['status']}  ({row.get('elapsed_s', 0):.0f}s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=PIPELINE_WORKERS) as ex:
            futs = {ex.submit(run_one_pipeline, n, c, g): n for n, c, g in trials}
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                print(f"  [done] {row['nct_id']}  {row['status']}  ({row.get('elapsed_s', 0):.0f}s)", flush=True)

    rows.sort(key=lambda r: r["nct_id"])
    ok = [r for r in rows if r["status"] == "ok"]
    skip = [r for r in rows if r["status"] == "skipped_exists"]
    err = [r for r in rows if r["status"] == "error"]
    elapsed_total = time.time() - t_start

    summary = {
        "phase": "pipeline",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(elapsed_total, 1),
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_skipped": len(skip),
        "n_errors": len(err),
        "error_nct_ids": [r["nct_id"] for r in err],
    }
    (PIPELINE_OUT / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[phase A] done: {len(ok)} ok, {len(skip)} skipped, {len(err)} errors  ({elapsed_total:.0f}s)", flush=True)
    if err:
        for r in err:
            print(f"  FAIL {r['nct_id']}: {r.get('error', '')[:120]}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase B: Unified Scoring Comparison
# ══════════════════════════════════════════════════════════════════════════════
def _load_pipeline_text(nct_id: str) -> str | None:
    """Read pipeline's criteria_final.md text, or None if missing."""
    final_path = PIPELINE_OUT / nct_id / "criteria_final.md"
    if not final_path.is_file():
        return None
    return final_path.read_text(encoding="utf-8").strip()


def score_one_pair(
    nct_id: str,
    pipeline_text: str,
    direct_text: str,
    trial_config_path: Path,
) -> dict[str, Any]:
    """Score both pipeline and direct criteria with the same scorer."""
    from criteria_agent.scorer import score_draft
    from shared.llm_client import new_scorer_client, resolve_scorer_llm_config
    from shared.trial_config import load_trial_config

    _, _, scorer_model = resolve_scorer_llm_config()
    client = new_scorer_client()
    config = load_trial_config(trial_config_path)

    result: dict[str, Any] = {"nct_id": nct_id}

    # Score pipeline criteria
    try:
        p = score_draft(client, config, pipeline_text, model=scorer_model)
        result["pipeline_scores"] = p["scores"]
        result["pipeline_mean"] = round(_mean(p["scores"]), 3)
    except Exception as e:
        result["pipeline_scores"] = {}
        result["pipeline_mean"] = None
        result["pipeline_error"] = str(e)[:300]

    # Score direct-gen criteria
    try:
        d = score_draft(client, config, direct_text, model=scorer_model)
        result["direct_scores"] = d["scores"]
        result["direct_mean"] = round(_mean(d["scores"]), 3)
    except Exception as e:
        result["direct_scores"] = {}
        result["direct_mean"] = None
        result["direct_error"] = str(e)[:300]

    # Delta
    if result["pipeline_mean"] is not None and result["direct_mean"] is not None:
        result["delta"] = round(result["pipeline_mean"] - result["direct_mean"], 3)
    else:
        result["delta"] = None

    return result


def phase_scoring(trials: list[tuple[str, Path, Path]]) -> None:
    """Score pipeline vs direct-gen for all filtered trials using the same scorer."""
    SCORING_OUT.mkdir(parents=True, exist_ok=True)

    direct_gen = load_direct_gen()
    print(f"[phase B] scoring: loaded {len(direct_gen)} direct-gen entries", flush=True)

    # Build work items — load pipeline text from criteria_final.md
    existing = load_jsonl(SCORES_JSONL)
    work: list[tuple[str, Path, str, str]] = []  # (nct, cfg, pipeline_text, direct_text)
    for nct, cfg, _ in trials:
        if nct in existing:
            continue
        pipeline_text = _load_pipeline_text(nct)
        if pipeline_text is None:
            print(f"  [skip] {nct}: no pipeline criteria_final.md", flush=True)
            continue
        if nct not in direct_gen:
            print(f"  [skip] {nct}: no direct-gen criteria", flush=True)
            continue
        work.append((nct, cfg, pipeline_text, direct_gen[nct]))

    print(f"[phase B] scoring: {len(existing)} cached, {len(work)} pending, workers={SCORING_WORKERS}", flush=True)

    if not work:
        print("[phase B] nothing to score", flush=True)
        _build_summary(trials)
        return

    t_start = time.time()

    if SCORING_WORKERS == 1:
        for i, (nct, cfg, ptxt, dtxt) in enumerate(work, 1):
            print(f"  [{i:03d}/{len(work)}] {nct} …", end=" ", flush=True)
            try:
                row = score_one_pair(nct, ptxt, dtxt, cfg)
                append_jsonl(SCORES_JSONL, row)
                d = row.get("delta", "?")
                print(f"pipeline={row.get('pipeline_mean')}  direct={row.get('direct_mean')}  delta={d}", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=SCORING_WORKERS) as ex:
            futs = {}
            for nct, cfg, ptxt, dtxt in work:
                futs[ex.submit(score_one_pair, nct, ptxt, dtxt, cfg)] = nct
            for fut in as_completed(futs):
                nct = futs[fut]
                try:
                    row = fut.result()
                    append_jsonl(SCORES_JSONL, row)
                    d = row.get("delta", "?")
                    print(f"  [done] {nct}  pipeline={row.get('pipeline_mean')}  direct={row.get('direct_mean')}  delta={d}", flush=True)
                except Exception as e:
                    print(f"  [done] {nct}  ERROR: {e}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n[phase B] scoring done ({elapsed:.0f}s)", flush=True)
    _build_summary(trials)


def _build_summary(trials: list[tuple[str, Path, Path]]) -> None:
    """Build summary.json from scores.jsonl."""
    scores = load_jsonl(SCORES_JSONL)
    if not scores:
        print("[summary] no scores to summarize", flush=True)
        return

    valid = [s for s in scores.values() if s.get("pipeline_mean") is not None and s.get("direct_mean") is not None]
    n = len(valid)

    # Per-dimension means
    dim_pipeline: dict[str, float] = {}
    dim_direct: dict[str, float] = {}
    dim_delta: dict[str, float] = {}
    for k in RUBRIC_KEYS:
        pv = [s["pipeline_scores"].get(k, 0.0) for s in valid]
        dv = [s["direct_scores"].get(k, 0.0) for s in valid]
        dim_pipeline[k] = round(sum(pv) / n, 3) if n else 0.0
        dim_direct[k] = round(sum(dv) / n, 3) if n else 0.0
        dim_delta[k] = round(dim_pipeline[k] - dim_direct[k], 3)

    # Overall means
    pipeline_means = [s["pipeline_mean"] for s in valid]
    direct_means = [s["direct_mean"] for s in valid]
    deltas = [s["delta"] for s in valid]

    # Win / loss / tie
    wins = sum(1 for d in deltas if d > 0.5)
    losses = sum(1 for d in deltas if d < -0.5)
    ties = n - wins - losses

    summary = {
        "scorer_model": "unified",
        "thinking_enabled": True,
        "n_trials": n,
        "pipeline_overall_mean": round(sum(pipeline_means) / n, 3) if n else 0.0,
        "direct_overall_mean": round(sum(direct_means) / n, 3) if n else 0.0,
        "overall_delta": round(sum(deltas) / n, 3) if n else 0.0,
        "per_dimension": {
            k: {"pipeline": dim_pipeline[k], "direct": dim_direct[k], "delta": dim_delta[k]}
            for k in RUBRIC_KEYS
        },
        "wins_pipeline": wins,
        "wins_direct": losses,
        "ties": ties,
        "win_rate_pipeline": round(wins / n, 3) if n else 0.0,
    }
    SCORING_OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*60}", flush=True)
    print(f"[SUMMARY]  {n} trials scored with unified scorer", flush=True)
    print(f"  Pipeline mean : {summary['pipeline_overall_mean']:.3f}", flush=True)
    print(f"  Direct   mean : {summary['direct_overall_mean']:.3f}", flush=True)
    print(f"  Delta         : {summary['overall_delta']:+.3f}", flush=True)
    print(f"  Win/Loss/Tie  : {wins}/{losses}/{ties}", flush=True)
    print(f"\n  Per-dimension deltas:", flush=True)
    for k in RUBRIC_KEYS:
        d = dim_delta[k]
        tag = "pipeline" if d > 0 else "direct" if d < 0 else "tie"
        print(f"    {k:25s}  pipeline={dim_pipeline[k]:.2f}  direct={dim_direct[k]:.2f}  delta={d:+.2f}  ({tag})", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  -> {SUMMARY_PATH}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Pipeline + unified scoring comparison for filtered bench.")
    ap.add_argument("--phase", choices=["pipeline", "score", "both"], default="both")
    ap.add_argument("--nct", type=str, default=None,
                    help="Comma-separated NCT IDs to process (e.g. NCT07158398,NCT07211425)")
    args = ap.parse_args()

    trials = discover_trials()
    if args.nct:
        wanted = {n.strip() for n in args.nct.split(",")}
        trials = [(n, c, g) for n, c, g in trials if n in wanted]
        print(f"[filter] --nct: kept {len(trials)} of {len(discover_trials())} trials", flush=True)
    if LIMIT:
        trials = trials[:LIMIT]
    print(f"[plan] {len(trials)} filtered trials  phase={args.phase}", flush=True)

    if args.phase in ("pipeline", "both"):
        phase_pipeline(trials)

    if args.phase in ("score", "both"):
        phase_scoring(trials)


if __name__ == "__main__":
    main()
