#!/usr/bin/env python3
"""Batch-run CriteriaAgent pipeline for all bench trials, WITH trial type + phase.

Usage:
    python scripts/run_bench_pipeline_with_phase.py

Env overrides:
    BENCH_PIPELINE_WORKERS   – concurrent workers (default: 1, LLM calls are heavy)
    BENCH_PIPELINE_LIMIT     – stop after N trials (smoke test)
    BENCH_PIPELINE_START     – skip first N trials (resume from offset)
    BENCH_PIPELINE_FORCE     – 1/true to re-generate even if criteria_final.md exists
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── fixed paths ──────────────────────────────────────────────────────────────
TRIALS_DIR   = ROOT / "CriteriaBench" / "final_bench" / "trials"
GRAPHS_DIR   = ROOT / "outputs"       / "bench_profiles_graphs" / "graphs"
OUTPUT_BASE  = ROOT / "outputs"       / "bench_criteria_with_phase"
SUMMARY_PATH = OUTPUT_BASE / "run_summary.json"

# ── env knobs ─────────────────────────────────────────────────────────────────
WORKERS   = max(1, int(os.environ.get("BENCH_PIPELINE_WORKERS", "1")))
LIMIT     = int(os.environ.get("BENCH_PIPELINE_LIMIT", "0")) or None
START     = int(os.environ.get("BENCH_PIPELINE_START", "0"))
FORCE     = os.environ.get("BENCH_PIPELINE_FORCE", "").lower() in ("1", "true", "yes")


def discover_trials() -> list[tuple[str, Path, Path]]:
    """Return sorted list of (nct_id, config_path, graph_path)."""
    pairs: list[tuple[str, Path, Path]] = []
    skipped: list[str] = []
    for cfg_path in sorted(TRIALS_DIR.glob("*.json")):
        nct_id = cfg_path.stem                       # e.g. NCT05225961
        graph_path = GRAPHS_DIR / f"{nct_id}_graph.json"
        if not graph_path.is_file():
            skipped.append(nct_id)
            continue
        pairs.append((nct_id, cfg_path, graph_path))
    if skipped:
        print(f"[warn] {len(skipped)} trial(s) have no graph and will be skipped:", flush=True)
        for nid in skipped:
            print(f"       {nid}", flush=True)
    return pairs


def run_one(nct_id: str, cfg_path: Path, graph_path: Path) -> dict:
    """Run the criteria-agent pipeline for one trial (v0 only)."""
    from criteria_agent.pipeline import run_pipeline

    out_dir = OUTPUT_BASE / nct_id
    final_path = out_dir / "criteria_final.md"

    if final_path.is_file() and not FORCE:
        # Resume: count as success without re-running.
        return {
            "nct_id": nct_id,
            "status": "skipped_exists",
            "output_dir": str(out_dir),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        trace = run_pipeline(
            graph_path=graph_path,
            config_path=cfg_path,
            output_dir=out_dir,
        )
        elapsed = time.time() - t0
        n_experts = len(trace.get("experts") or [])
        chars = sum(trace.get("expert_answer_chars") or [0])
        return {
            "nct_id": nct_id,
            "status": "ok",
            "elapsed_s": round(elapsed, 1),
            "n_experts": n_experts,
            "expert_answer_chars": chars,
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


def write_summary(rows: list[dict], t_start: float) -> None:
    ok      = [r for r in rows if r["status"] == "ok"]
    skipped = [r for r in rows if r["status"] == "skipped_exists"]
    errors  = [r for r in rows if r["status"] == "error"]
    elapsed_total = time.time() - t_start

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(elapsed_total, 1),
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_skipped_exists": len(skipped),
        "n_errors": len(errors),
        "error_nct_ids": [r["nct_id"] for r in errors],
        "ok_mean_elapsed_s": (
            round(sum(r["elapsed_s"] for r in ok) / len(ok), 1) if ok else None
        ),
        "results": rows,
    }
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\n[summary] total={len(rows)}  ok={len(ok)}  "
        f"skipped={len(skipped)}  errors={len(errors)}  "
        f"elapsed={elapsed_total:.0f}s",
        flush=True,
    )
    if errors:
        print("[errors]", flush=True)
        for r in errors:
            print(f"  {r['nct_id']}: {r['error'][:120]}", flush=True)


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    trials = discover_trials()
    print(
        f"[plan] trials={len(trials)}  workers={WORKERS}  "
        f"force={FORCE}  start_offset={START}  limit={LIMIT}",
        flush=True,
    )

    if START:
        trials = trials[START:]
    if LIMIT:
        trials = trials[:LIMIT]

    t_start = time.time()
    rows: list[dict] = []

    if WORKERS == 1:
        # Sequential – easier to follow in the terminal.
        for i, (nct_id, cfg_path, graph_path) in enumerate(trials, 1):
            print(
                f"[{i:03d}/{len(trials)}] {nct_id} …",
                end=" ",
                flush=True,
            )
            row = run_one(nct_id, cfg_path, graph_path)
            rows.append(row)
            tag = row["status"]
            elapsed = row.get("elapsed_s", 0)
            print(f"{tag}  ({elapsed:.0f}s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {
                ex.submit(run_one, nid, cfg, grp): nid
                for nid, cfg, grp in trials
            }
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                tag = row["status"]
                elapsed = row.get("elapsed_s", 0)
                print(f"[done] {row['nct_id']}  {tag}  ({elapsed:.0f}s)", flush=True)

    # Sort results by NCT ID for stable output.
    rows.sort(key=lambda r: r["nct_id"])
    write_summary(rows, t_start)
    print(f"[done] summary -> {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
