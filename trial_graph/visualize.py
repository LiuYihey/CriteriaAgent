"""
Publication-grade interactive visualization for trial_knowledge_graph_v4 JSON.

Single HTML output, white background, minimal chrome — designed to look clean enough
for a screenshot to drop into a top-venue paper figure.

Node:  size scales with degree; uniform soft-blue fill, navy border, dark label.
Edge:  extracted = solid charcoal; inferred = dashed slate.
Tip:   hovering shows the node's name and 1-hop relations (the node IS its edges).

Requires: pip install pyvis networkx
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from pyvis.network import Network
except ImportError as e:
    raise SystemExit("pip install pyvis networkx") from e


NODE_FILL = "#dbe6f3"
NODE_BORDER = "#1f3a5f"
NODE_LABEL = "#0f1c30"
NODE_HIGHLIGHT_BORDER = "#c0392b"

EDGE_EXTRACTED = "#2c3e50"
EDGE_INFERRED = "#9aa6b2"
EDGE_HIGHLIGHT = "#c0392b"


def load_graph(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_digraph(data: dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = data["nodes"]
    edges = data["edges"]
    chunks = data.get("chunks") or {}

    id_to_name = {n["id"]: n["name"] for n in nodes}
    neighbors_out: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    neighbors_in: dict[str, list[str]] = {n["id"]: [] for n in nodes}

    for e in edges:
        src, dst = e["src"], e["dst"]
        rel = e.get("relation", "")
        kind = e.get("kind", "")
        freq = e.get("frequency", 1)
        freq_suffix = f" ×{freq}" if int(freq) > 1 else ""
        marker = "·" if kind == "extracted" else "~"
        neighbors_out.setdefault(src, []).append(
            f"{marker} {rel}{freq_suffix} → {id_to_name.get(dst, dst)}"
        )
        neighbors_in.setdefault(dst, []).append(
            f"{id_to_name.get(src, src)} → {rel}{freq_suffix} {marker}"
        )

    for n in nodes:
        nid = n["id"]
        cref = n.get("chunk_ref", "")
        category = n.get("category", "")
        chunk_text = ""
        resource = ""
        if cref and cref in chunks:
            raw = chunks[cref]
            if isinstance(raw, str):
                chunk_text = raw
            else:
                chunk_text = raw.get("text", "")
                resource = raw.get("source", "") or ""
                if not resource and isinstance(raw.get("source_meta"), dict):
                    meta = raw["source_meta"]
                    t = meta.get("type", "")
                    ent = meta.get("entity", "")
                    sec = meta.get("section", "")
                    if t and ent:
                        resource = f"{ent}_info::{sec}" if sec else f"{ent}_info"
            chunk_text = chunk_text[:260].replace("\n", " ")
            if len(chunk_text) >= 260:
                chunk_text += "…"

        nb_lines = neighbors_out.get(nid, [])[:8] + neighbors_in.get(nid, [])[:8]
        nb_lines = nb_lines[:12]
        tip_lines = [n["name"]]
        if category:
            tip_lines.append(f"category: {category}")
        if cref:
            tip_lines.append(f"chunk_ref: {cref}")
            if resource:
                tip_lines.append(f"resource: {resource}")
            if chunk_text:
                tip_lines.append(f"  {chunk_text}")
        if nb_lines:
            tip_lines.append("— 1-hop —")
            tip_lines.extend(nb_lines)

        G.add_node(
            nid,
            label=n["name"][:46] + ("…" if len(n["name"]) > 46 else ""),
            title=html.escape("\n".join(tip_lines)),
        )

    for e in edges:
        rel = e.get("relation", "")
        kind = e.get("kind", "inferred")
        freq = e.get("frequency", 1)
        freq_suffix = f" ×{freq}" if freq and int(freq) > 1 else ""
        G.add_edge(
            e["src"],
            e["dst"],
            relation=rel,
            kind=kind,
            title=html.escape(f"{rel}  ({kind}){freq_suffix}"),
        )
    return G


def to_pyvis(G: nx.DiGraph, title: str) -> Network:
    # Construct with empty heading so pyvis won't render its own <h1>.
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color=NODE_LABEL,
        directed=True,
        notebook=False,
        heading="",
    )

    degrees = dict(G.degree())
    if degrees:
        d_max = max(degrees.values()) or 1
    else:
        d_max = 1

    for nid in G.nodes():
        attrs = G.nodes[nid]
        deg = degrees.get(nid, 0)
        size = 14 + 18 * (deg / d_max) ** 0.7
        net.add_node(
            nid,
            label=attrs["label"],
            title=attrs["title"],
            color={
                "background": NODE_FILL,
                "border": NODE_BORDER,
                "highlight": {"background": "#ffffff", "border": NODE_HIGHLIGHT_BORDER},
                "hover": {"background": "#eef3fa", "border": NODE_BORDER},
            },
            borderWidth=1.4,
            borderWidthSelected=2.2,
            size=size,
            font={
                "size": 13,
                "face": "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
                "color": NODE_LABEL,
                "strokeWidth": 0,
            },
            shape="dot",
        )

    for u, v, attrs in G.edges(data=True):
        kind = attrs.get("kind", "inferred")
        ec = EDGE_EXTRACTED if kind == "extracted" else EDGE_INFERRED
        net.add_edge(
            u,
            v,
            title=attrs["title"],
            label=attrs.get("relation", "") if False else "",  # labels hidden by default for clarity
            color={"color": ec, "highlight": EDGE_HIGHLIGHT, "hover": EDGE_HIGHLIGHT},
            width=1.2 if kind == "extracted" else 0.9,
            dashes=(kind != "extracted"),
            arrows={"to": {"enabled": True, "scaleFactor": 0.55}},
            smooth={"type": "continuous", "roundness": 0.22},
        )

    opts = {
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -58,
                "centralGravity": 0.012,
                "springLength": 110,
                "springConstant": 0.07,
                "avoidOverlap": 0.95,
            },
            "minVelocity": 0.6,
            "timestep": 0.32,
            "stabilization": {"iterations": 500},
        },
        "interaction": {
            "hover": True,
            "navigationButtons": False,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": False,
            "multiselect": True,
        },
        "edges": {"smooth": {"type": "continuous"}, "selectionWidth": 1.2},
        "nodes": {"shadow": False},
    }
    net.set_options(json.dumps(opts))
    return net


PAGE_CSS = """
:root { --ink:#0f1c30; --mute:#5b6b7f; --line:#dbe1ea; }
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background:#ffffff; color:var(--ink);
  font-family: Inter, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
.kg-banner {
  padding: 14px 22px 10px; background:#ffffff; border-bottom:1px solid var(--line);
}
.kg-banner h1 {
  font-size: 16px; font-weight: 600; margin:0 0 4px; letter-spacing:.2px;
}
.kg-banner .sub { font-size:12px; color:var(--mute); margin:0; }
.kg-banner .legend { font-size:12px; color:var(--ink); margin-top:6px; display:flex; gap:18px; flex-wrap:wrap; }
.kg-banner .legend .sw { display:inline-block; width:24px; height:0; vertical-align:middle; margin-right:6px; }
.kg-banner .sw.ext { border-top:2px solid #2c3e50; }
.kg-banner .sw.inf { border-top:2px dashed #9aa6b2; }
.kg-banner .dot {
  display:inline-block; width:10px; height:10px; border-radius:50%;
  background:#dbe6f3; border:1.4px solid #1f3a5f; margin-right:6px; vertical-align:middle;
}
#mynetwork { border: none !important; }
.vis-tooltip {
  font-family: Inter, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif !important;
  font-size: 12px !important; line-height: 1.45 !important;
  background:#ffffff !important; color:#0f1c30 !important;
  border:1px solid #cbd5e1 !important; border-radius:6px !important;
  padding:8px 10px !important; box-shadow:0 4px 16px rgba(15,28,48,.10) !important;
  white-space: pre-wrap !important; max-width: 460px !important;
}
"""


def build_banner_html(title: str, n_nodes: int, n_edges: int) -> str:
    return (
        f'<div class="kg-banner">'
        f'<h1>{html.escape(title)}</h1>'
        f'<p class="sub">{n_nodes} nodes · {n_edges} edges · concept graph for eligibility-criteria writing</p>'
        f'<div class="legend">'
        f'<span><span class="dot"></span>concept node (size ∝ degree)</span>'
        f'<span><span class="sw ext"></span>extracted relation</span>'
        f'<span><span class="sw inf"></span>inferred relation</span>'
        f'</div></div>'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render trial_graph.json as a publication-ready HTML graph.")
    ap.add_argument("-i", "--input", type=Path, default=Path("trial_graph.json"))
    ap.add_argument("-o", "--output", type=Path, default=Path("trial_graph_visualization.html"))
    args = ap.parse_args()

    data = load_graph(args.input)
    title = f"Trial Knowledge Graph — {data.get('source_profile', args.input.name)}"
    G = build_digraph(data)
    net = to_pyvis(G, title=title)

    raw_html = net.generate_html()
    style_inject = f"<style>{PAGE_CSS}</style>"
    banner_html = build_banner_html(title, G.number_of_nodes(), G.number_of_edges())

    if "</head>" in raw_html:
        raw_html = raw_html.replace("</head>", style_inject + "</head>", 1)
    else:
        raw_html = style_inject + raw_html
    raw_html = raw_html.replace("<body>", "<body>" + banner_html, 1)

    args.output.write_text(raw_html, encoding="utf-8")
    print(
        f"Wrote {args.output.resolve()}  (nodes={G.number_of_nodes()} edges={G.number_of_edges()})"
    )


if __name__ == "__main__":
    main()
