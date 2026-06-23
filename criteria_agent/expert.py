"""Expert subagent: L0 selection → L1/L2 disclosure → natural-language answer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from criteria_agent.planner import Subtask
from criteria_agent.prompts import EXPERT_L0, EXPERT_L1L2, ROLE_L0, ROLE_L1L2
from shared.llm_client import call_text
from shared.trial_config import TrialConfig
from trial_graph.disclosure_format import format_l0_catalog, format_l1_l2_prompt
from trial_graph.subgraph import l0_catalog, load_graph, package_disclosure


NODE_ID_RE = re.compile(r"\bn(\d{4})\b", re.I)


@dataclass
class ExpertResult:
    subtask: Subtask
    l0_response: str
    selected_ids: list[str]
    l1l2_response: str
    role: str = "safety"
    disclosure: dict[str, Any] | None = None


def extract_node_ids(text: str, valid: set[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for m in NODE_ID_RE.finditer(text):
        nid = f"n{m.group(1)}"
        if nid in valid and nid not in seen:
            seen.add(nid)
            ordered.append(nid)
    return ordered


def run_expert(
    client: Any,
    config: TrialConfig,
    graph_path: str | Path,
    subtask: Subtask,
    *,
    model: str | None = None,
    l0_system: str | None = None,
    l1l2_system: str | None = None,
) -> ExpertResult:
    # Resolve role-specific prompts
    role = subtask.role
    _l0 = l0_system or ROLE_L0.get(role, EXPERT_L0)
    _l1l2 = l1l2_system or ROLE_L1L2.get(role, EXPERT_L1L2)
    graph_path = Path(graph_path)
    graph = load_graph(graph_path)
    valid_ids = {n["id"] for n in graph["nodes"]}
    catalog = l0_catalog(graph)

    l0_user = (
        f"Trial configuration:\n{config.context_block()}\n\n"
        f"Your subtask:\n{subtask.question}\n\n"
        f"Concept catalog ({len(catalog)} nodes, id|name per line):\n"
        f"{format_l0_catalog(catalog)}"
    )
    l0_resp = call_text(client, system=_l0, user=l0_user, model=model)
    selected = extract_node_ids(l0_resp, valid_ids)
    if len(selected) < 3:
        for row in catalog[:30]:
            if row["id"] not in selected:
                selected.append(row["id"])
            if len(selected) >= 8:
                break

    bundle = package_disclosure(selected, graph_path)
    evidence = format_l1_l2_prompt(
        bundle["L1_subgraph"],
        bundle["L2_chunks_dedup"],
        selected_ids=set(selected),
    )
    l1l2_user = (
        f"Trial configuration:\n{config.context_block()}\n\n"
        f"Your subtask:\n{subtask.question}\n\n"
        f"Graph evidence:\n{evidence}"
    )
    l1l2_resp = call_text(client, system=_l1l2, user=l1l2_user, model=model)

    return ExpertResult(
        subtask=subtask,
        l0_response=l0_resp,
        selected_ids=selected,
        l1l2_response=l1l2_resp,
        role=role,
        disclosure={
            "selected_ids": selected,
            "L1_subgraph": bundle["L1_subgraph"],
            "L2_chunk_ids": list(bundle["L2_chunks_dedup"].keys()),
        },
    )


def expert_result_to_dict(result: ExpertResult) -> dict[str, Any]:
    return {
        "subtask_index": result.subtask.index,
        "subtask_question": result.subtask.question,
        "role": result.role,
        "l0_response": result.l0_response,
        "selected_ids": result.selected_ids,
        "l1l2_response": result.l1l2_response,
        "disclosure": result.disclosure,
    }
