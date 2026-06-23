#!/usr/bin/env python3
"""CLI entry point for the CriteriaAgent pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from criteria_agent.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CriteriaAgent eligibility-criteria pipeline.")
    ap.add_argument("--graph", required=True, help="Path to trial_graph.json")
    ap.add_argument("--config", required=True, help="Path to trial config JSON")
    ap.add_argument("-o", "--output", default="outputs/criteria_agent_run", help="Output directory")
    args = ap.parse_args()

    trace = run_pipeline(
        args.graph,
        args.config,
        args.output,
    )
    n_experts = len(trace.get("experts") or [])
    print(f"Done. {n_experts} expert(s), output -> {Path(args.output).resolve()}")
    print(f"  criteria_final.md, trace.json")


if __name__ == "__main__":
    main()
