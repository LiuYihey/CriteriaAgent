#!/usr/bin/env python3
"""Vanilla RAG baseline: concatenate four-domain profile, single-pass generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.vanilla_rag.prompts import VANILLA_RAG_SYSTEM, build_user_prompt  # noqa: E402
from shared.llm_client import call_text, new_client, resolve_llm_config  # noqa: E402
from shared.trial_config import load_trial_config  # noqa: E402


def load_profile(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError("trial profile must be a JSON object")
    return obj


def generate(config_path: Path, profile_path: Path, output_dir: Path) -> str:
    config = load_trial_config(config_path)
    profile = load_profile(profile_path)
    client = new_client()
    _, _, model = resolve_llm_config()
    user = build_user_prompt(config.title, config.arms_text, config.primary_text, profile)
    text = call_text(client, system=VANILLA_RAG_SYSTEM, user=user, model=model)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "criteria.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Vanilla RAG baseline (single-pass, no optimizer).")
    ap.add_argument("--config", required=True, help="Trial config JSON")
    ap.add_argument("--profile", required=True, help="Four-domain trial_profile JSON")
    ap.add_argument("-o", "--output", default="baselines/vanilla_rag/outputs/run", help="Output directory")
    args = ap.parse_args()
    out = generate(Path(args.config), Path(args.profile), Path(args.output))
    print(f"Wrote criteria ({len(out)} chars) -> {Path(args.output).resolve() / 'criteria.md'}")


if __name__ == "__main__":
    main()
