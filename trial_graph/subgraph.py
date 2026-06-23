"""
Helpers for Agentic Subgraph Extraction (L0→L2) from trial_graph.json (schema v4).

  L0: catalog only (id, name).
  L1: induced subgraph on the selected nodes only (no 1-hop expansion);
      edges kept only when BOTH endpoints are selected.
  L2: deduplicated chunk texts for the selected nodes only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_graph(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def l0_catalog(graph: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in graph["nodes"]:
        row: dict[str, Any] = {"id": n["id"], "name": n["name"]}
        if n.get("category"):
            row["category"] = n["category"]
        out.append(row)
    return out


def selected_edges(selected_ids: list[str], graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Edges where BOTH endpoints are selected, deduplicated (no 1-hop expansion)."""
    sel = set(selected_ids)
    edges_out: list[dict[str, Any]] = []
    seen_e: set[tuple[str, str, str, str]] = set()
    for e in graph["edges"]:
        if e["src"] not in sel or e["dst"] not in sel:
            continue
        tup = (e["src"], e["dst"], e["relation"], e["kind"])
        if tup in seen_e:
            continue
        seen_e.add(tup)
        row: dict[str, Any] = {
            "src": e["src"],
            "dst": e["dst"],
            "relation": e["relation"],
            "kind": e["kind"],
        }
        freq = e.get("frequency", 1)
        if freq > 1:
            row["frequency"] = freq
        edges_out.append(row)
    return edges_out


def l1_subgraph(selected_ids: list[str], graph: dict[str, Any]) -> dict[str, Any]:
    """Induced subgraph on the selected nodes only (edges among selected nodes)."""
    sel = list(dict.fromkeys(selected_ids))
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    nodes_out = []
    for vid in sel:
        if vid not in node_by_id:
            continue
        n = node_by_id[vid]
        row: dict[str, Any] = {
            "id": n["id"],
            "name": n["name"],
            "chunk_ref": n.get("chunk_ref"),
        }
        if n.get("category"):
            row["category"] = n["category"]
        nodes_out.append(row)
    return {"nodes": nodes_out, "edges": selected_edges(sel, graph)}


def l2_chunks_for_selected(selected_ids: list[str], graph: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Passage appendix only for explicitly selected nodes (not 1-hop neighbors)."""
    chunks = graph.get("chunks", {})
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    out: dict[str, dict[str, str]] = {}
    seen_chunk: set[str] = set()
    for nid in dict.fromkeys(selected_ids):
        n = node_by_id.get(nid)
        if not n:
            continue
        cid = n.get("chunk_ref")
        if not cid or cid in seen_chunk or cid not in chunks:
            continue
        seen_chunk.add(cid)
        c = chunks[cid]
        if isinstance(c, str):
            out[cid] = {"text": c, "source": ""}
        else:
            row: dict[str, Any] = {"text": c.get("text", ""), "source": c.get("source", "")}
            if c.get("source_meta"):
                row["source_meta"] = c["source_meta"]
            out[cid] = row
    return out


def l2_raw_appendix(subgraph: dict[str, Any], graph: dict[str, Any]) -> dict[str, str]:
    chunks = graph.get("chunks", {})
    ids_ordered = []
    seen: set[str] = set()
    for n in subgraph["nodes"]:
        cid = n.get("chunk_ref")
        if cid and cid not in seen:
            seen.add(cid)
            ids_ordered.append(cid)
    out: dict[str, str] = {}
    for cid in ids_ordered:
        if cid not in chunks:
            continue
        c = chunks[cid]
        out[cid] = c if isinstance(c, str) else c.get("text", "")
    return out


def package_disclosure(selected_ids: list[str], graph_path: str | Path) -> dict[str, Any]:
    """Full L0 → L1 → L2 bundle for one expert turn."""
    g = load_graph(graph_path)
    sg = l1_subgraph(selected_ids, g)
    raw = l2_chunks_for_selected(selected_ids, g)
    return {"L0_catalog": l0_catalog(g), "L1_subgraph": sg, "L2_chunks_dedup": raw}


__all__ = [
    "load_graph",
    "l0_catalog",
    "l1_subgraph",
    "l2_chunks_for_selected",
    "l2_raw_appendix",
    "selected_edges",
    "package_disclosure",
]
