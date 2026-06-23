#!/usr/bin/env python3
"""Audit trial_graph.json edge quality and write a markdown report."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def audit_graph(graph_path: Path, out_path: Path) -> dict:
    g = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    edges = g["edges"]
    chunks = g.get("chunks", {})

    meta_rx = re.compile(r"\b(?:NCT|PMID)\b|similar trial|study record|paper id", re.I)
    bad_nodes = [n for n in g["nodes"] if meta_rx.search(n["name"])]

    lines: list[str] = [
        "# Trial graph edge audit",
        "",
        f"Source: `{graph_path.name}`",
        "",
        "## Summary",
        "",
        f"- Nodes: {len(g['nodes'])}",
        f"- Edges: {len(edges)}",
        f"- Chunks: {len(chunks)}",
        f"- Kind: {dict(Counter(e.get('kind') for e in edges))}",
        f"- Frequency > 1: {sum(e.get('frequency', 1) > 1 for e in edges)}",
        f"- Self-loops: {sum(e['src'] == e['dst'] for e in edges)}",
        f"- Missing endpoints: {sum(e['src'] not in nodes or e['dst'] not in nodes for e in edges)}",
        f"- Relation contains `*`: {sum('*' in e.get('relation', '') for e in edges)}",
        f"- Metadata-like nodes: {len(bad_nodes)}",
        "",
    ]

    if bad_nodes:
        lines.append("### Metadata-like nodes (should be 0)")
        for n in bad_nodes:
            lines.append(f"- {n['name']} ({n.get('category', '')})")
        lines.append("")

    inferred = [e for e in edges if e.get("kind") == "inferred"]
    lines.append(f"## Inferred edges ({len(inferred)})")
    lines.append("")
    for i, e in enumerate(inferred, 1):
        s, d = nodes[e["src"]], nodes[e["dst"]]
        lines.append(f"{i}. **{s['name']}** — {e['relation']} — **{d['name']}** (f={e.get('frequency', 1)})")
    lines.append("")

    freq_edges = [e for e in edges if e.get("frequency", 1) > 1]
    if freq_edges:
        lines.append(f"## High-frequency edges ({len(freq_edges)})")
        lines.append("")
        for e in sorted(freq_edges, key=lambda x: -x.get("frequency", 1)):
            s, d = nodes[e["src"]], nodes[e["dst"]]
            lines.append(
                f"- f={e['frequency']}: {s['name']} — {e['relation']} — {d['name']} [{e['kind']}]"
            )
        lines.append("")

    def _chunk_text(cid: str | None) -> str:
        if not cid:
            return ""
        raw = chunks.get(cid, "")
        return raw if isinstance(raw, str) else raw.get("text", "")

    def _name_in_chunk(name: str, text: str) -> bool:
        key = name.lower()
        if key in text.lower():
            return True
        # partial: first significant token
        parts = [p for p in re.split(r"[\s\-/,]+", key) if len(p) > 3]
        return any(p in text.lower() for p in parts[:3])

    weak: list[tuple[int, str, str]] = []
    for i, e in enumerate(edges, 1):
        s, d = nodes[e["src"]], nodes[e["dst"]]
        st, dt = _chunk_text(s.get("chunk_ref")), _chunk_text(d.get("chunk_ref"))
        combined = f"{st}\n{dt}"
        if not (_name_in_chunk(s["name"], combined) and _name_in_chunk(d["name"], combined)):
            weak.append((i, s["name"], d["name"]))

    lines.append(f"## Weak anchor check ({len(weak)} edges)")
    lines.append("")
    lines.append(
        "Endpoints whose Top-1 chunk text may not mention both entity names "
        "(heuristic; review manually)."
    )
    lines.append("")
    for i, sname, dname in weak[:40]:
        e = edges[i - 1]
        lines.append(
            f"- #{i}: {sname} — {e['relation']} — {dname} [{e['kind']}]"
        )
    if len(weak) > 40:
        lines.append(f"- ... and {len(weak) - 40} more")
    lines.append("")

    suspicious: list[tuple[int, dict]] = []
    for i, e in enumerate(edges, 1):
        rel = e.get("relation", "").lower()
        if any(x in rel for x in ("compared to", " more than ", "than ", " when combined")):
            suspicious.append((i, e))
    lines.append(f"## Suspicious relation phrasing ({len(suspicious)})")
    lines.append("")
    for i, e in suspicious[:25]:
        s, d = nodes[e["src"]], nodes[e["dst"]]
        lines.append(f"- #{i}: {s['name']} — {e['relation']} — {d['name']}")
    lines.append("")

    lines.append("## Top relations")
    lines.append("")
    for rel, c in Counter(e["relation"] for e in edges).most_common(15):
        lines.append(f"- {c}× `{rel}`")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "nodes": len(g["nodes"]),
        "edges": len(edges),
        "metadata_nodes": len(bad_nodes),
        "weak_anchors": len(weak),
        "inferred": len(inferred),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit trial_graph.json edges.")
    ap.add_argument("-i", "--input", default="trial_graph.json")
    ap.add_argument("-o", "--output", default="trial_graph_edge_audit.md")
    args = ap.parse_args()
    stats = audit_graph(Path(args.input), Path(args.output))
    print(f"Wrote {args.output}: {stats}")


if __name__ == "__main__":
    main()
