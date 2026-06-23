"""Compact, clinical-semantics-only formatting for Expert L0/L1/L2 prompts."""

from __future__ import annotations

from typing import Any


def _kind_tag(kind: str) -> str:
    k = (kind or "").strip().lower()
    if k == "extracted":
        return "E"
    if k == "inferred":
        return "I"
    return k[:1].upper() if k else "?"


def format_l0_catalog(catalog: list[dict[str, Any]]) -> str:
    """Full catalog, one line per node: id|name. Category only if it adds meaning."""
    lines: list[str] = []
    for row in catalog:
        nid = row["id"]
        name = str(row["name"]).strip()
        cat = str(row.get("category") or "").strip()
        if cat and cat.lower() not in name.lower():
            lines.append(f"{nid}|{name} ({cat})")
        else:
            lines.append(f"{nid}|{name}")
    return "\n".join(lines)


def format_l1_l2_prompt(
    subgraph: dict[str, Any],
    chunks: dict[str, Any],
    *,
    selected_ids: set[str] | None = None,
) -> str:
    """Node-centric evidence: each selected concept, with its relations and source
    passage attached underneath — both deduplicated across nodes.

    - Relations: each relation between two selected nodes is described exactly once,
      under its source node. The same A–B relation is never repeated on both ends.
    - Passages: a passage shared by several selected nodes is shown once, under the
      first node that references it.
    - A node with no relations omits the "Relations" line entirely; a node with no
      (or already-shown) passage omits the "Passage" line entirely.
    """
    nodes = subgraph.get("nodes") or []
    edges = subgraph.get("edges") or []
    id_to_name = {n["id"]: str(n["name"]).strip() for n in nodes}
    if selected_ids is None:
        selected_ids = set(id_to_name)

    # Group relations under their source node, deduplicated.
    rel_by_src: dict[str, list[str]] = {}
    seen_rel: set[tuple[str, str, str, str]] = set()
    for e in edges:
        src_id, dst_id = e["src"], e["dst"]
        if src_id not in selected_ids or dst_id not in selected_ids:
            continue
        src = id_to_name.get(src_id, src_id)
        dst = id_to_name.get(dst_id, dst_id)
        rel = str(e.get("relation") or "").strip()
        tag = _kind_tag(str(e.get("kind") or ""))
        key = (src, dst, rel, tag)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        freq = e.get("frequency")
        freq_s = f" ×{freq}" if isinstance(freq, int) and freq > 1 else ""
        rel_by_src.setdefault(src_id, []).append(f"{src} --{rel} [{tag}]-> {dst}{freq_s}")

    blocks: list[str] = []
    shown_chunks: set[str] = set()
    for n in nodes:
        nid = n["id"]
        if nid not in selected_ids:
            continue
        name = str(n["name"]).strip()
        cat = str(n.get("category") or "").strip()
        header = f"**{name}** ({nid})"
        if cat and cat.lower() not in name.lower():
            header += f" [{cat}]"

        lines = [header]

        rels = rel_by_src.get(nid)
        if rels:
            lines.append("Relations:")
            lines.extend(f"- {r}" for r in rels)

        cid = n.get("chunk_ref")
        if cid and cid not in shown_chunks and cid in chunks:
            chunk_data = chunks[cid]
            if isinstance(chunk_data, dict):
                passage = str(chunk_data.get("text", "")).strip()
                source = chunk_data.get("source", "")
            else:
                passage = str(chunk_data or "").strip()
                source = ""
            if passage:
                shown_chunks.add(cid)
                source_label = f" [{source}]" if source else ""
                lines.append(f"Passage{source_label}:")
                lines.append(passage)

        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks) if blocks else "(no selected concepts)"
