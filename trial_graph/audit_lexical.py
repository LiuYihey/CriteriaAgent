#!/usr/bin/env python3
"""Audit trial_graph edges against source chunks (lexical evidence check)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _tokens(name: str) -> list[str]:
    n = _norm(name)
    parts = re.split(r"[^a-z0-9]+", n)
    return [p for p in parts if len(p) >= 3]


def load_graph(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def audit(path: str) -> dict[str, Any]:
    g = load_graph(Path(path))
    nodes = {n["id"]: n for n in g["nodes"]}
    chunks = g.get("chunks", {})
    chunk_texts = [
        (cid, t if isinstance(t, str) else t.get("text", ""))
        for cid, t in chunks.items()
    ]
    all_text = "\n".join(t for _, t in chunk_texts)

    results: list[dict[str, Any]] = []
    stats = {
        "total_edges": 0,
        "both_entities_in_corpus": 0,
        "relation_near_pair": 0,
        "same_chunk_pair": 0,
        "extracted_kind": 0,
        "inferred_kind": 0,
        "high_freq": 0,
    }

    for e in g.get("edges", []):
        stats["total_edges"] += 1
        sid, did = e["src"], e["dst"]
        sn = nodes.get(sid, {}).get("name", sid)
        dn = nodes.get(did, {}).get("name", did)
        rel = e.get("relation", "")
        kind = e.get("kind", "")
        freq = e.get("frequency", 1)
        if kind == "extracted":
            stats["extracted_kind"] += 1
        else:
            stats["inferred_kind"] += 1
        if freq >= 2:
            stats["high_freq"] += 1

        stoks = _tokens(sn)
        dtoks = _tokens(dn)
        src_hit = any(t in all_text for t in (stoks or [_norm(sn)]))
        dst_hit = any(t in all_text for t in (dtoks or [_norm(dn)]))
        both = src_hit and dst_hit
        if both:
            stats["both_entities_in_corpus"] += 1

        rel_toks = [t for t in re.split(r"[^a-z0-9]+", _norm(rel)) if len(t) >= 4]
        near = False
        same_chunk = False
        best_chunk = ""
        for cid, text in chunk_texts:
            tl = _norm(text)
            if not (stoks or [_norm(sn)]):
                continue
            s_ok = any(t in tl for t in (stoks or [_norm(sn)]))
            d_ok = any(t in tl for t in (dtoks or [_norm(dn)]))
            if s_ok and d_ok:
                same_chunk = True
                best_chunk = cid
                if rel_toks and any(rt in tl for rt in rel_toks):
                    near = True
                elif rel and _norm(rel)[:12] in tl:
                    near = True
                break
        if not near and both and rel_toks:
            # cross-chunk: both exist somewhere and relation word in corpus
            near = any(rt in all_text for rt in rel_toks)
        if near:
            stats["relation_near_pair"] += 1
        if same_chunk:
            stats["same_chunk_pair"] += 1

        if both and (near or same_chunk):
            verdict = "supported"
        elif both:
            verdict = "weak" if kind == "inferred" else "check"
        elif src_hit or dst_hit:
            verdict = "partial"
        else:
            verdict = "unsupported"

        results.append({
            "src": sn,
            "relation": rel,
            "dst": dn,
            "kind": kind,
            "frequency": freq,
            "verdict": verdict,
            "same_chunk": same_chunk,
            "chunk": best_chunk,
        })

    by_verdict: dict[str, int] = {}
    for r in results:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1

    weak_or_bad = [r for r in results if r["verdict"] in ("unsupported", "partial", "check")]
    weak_or_bad.sort(key=lambda x: (x["verdict"], -x["frequency"]))

    return {
        "stats": stats,
        "by_verdict": by_verdict,
        "sample_issues": weak_or_bad[:25],
        "sample_good": [r for r in results if r["verdict"] == "supported"][:8],
    }


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "trial_graph.json"
    report = audit(path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
