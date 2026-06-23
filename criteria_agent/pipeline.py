"""End-to-end CriteriaAgent pipeline orchestration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from criteria_agent.expert import expert_result_to_dict, run_expert
from criteria_agent.planner import plan_subtasks
from criteria_agent.writer import write_initial_draft
from shared.llm_client import (
    new_client,
    resolve_llm_config,
)
from shared.trial_config import load_trial_config


def run_pipeline(
    graph_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()

    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_trial_config(config_path)
    client = new_client()
    _, _, model = resolve_llm_config()

    # 1. Planner: three investigation questions
    planner_raw, subtasks = plan_subtasks(client, config, model=model)

    # 2. Expert L0 + L1/L2 for each subtask
    expert_results = [run_expert(client, config, graph_path, st, model=model) for st in subtasks]

    # 3. Writer generates one criteria draft
    final = write_initial_draft(client, config, expert_results, model=model)

    # 4. Output
    (output_dir / "criteria_final.md").write_text(final, encoding="utf-8")

    # 5. Build trace
    trace: dict[str, Any] = {
        "schema": "criteria_agent_trace_v3",
        "started_at": started_at,
        "model": model,
        "graph_path": str(graph_path.resolve()),
        "config_path": str(Path(config_path).resolve()),
        "trial_title": config.title,
        "nct_id": config.nct_id,
        "planner": {
            "raw": planner_raw,
            "subtasks": [{"index": s.index, "question": s.question, "role": s.role} for s in subtasks],
        },
        "experts": [expert_result_to_dict(er) for er in expert_results],
        "expert_answer_chars": [len(er.l1l2_response) for er in expert_results],
        "criteria_final": final,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }

    (output_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return trace
