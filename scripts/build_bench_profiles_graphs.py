#!/usr/bin/env python3
"""Build trial knowledge graphs + HTML visualizations for all bench_profiles.

This script processes all JSON files in the bench_profiles directory to
build knowledge graphs and interactive HTML visualizations.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trial_graph.build import run_pipeline


def load_graph_simple(path: Path) -> dict:
    """Simplified graph loading without visualization dependencies."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def render_html(graph_path: Path, html_path: Path) -> tuple[int, int]:
    """Render HTML visualization with deferred import of visualization libraries."""
    from trial_graph.visualize import (
        build_banner_html,
        build_digraph,
        load_graph,
        to_pyvis,
        PAGE_CSS,
    )
    data = load_graph(graph_path)
    title = f"Trial Knowledge Graph — {data.get('source_profile', graph_path.name)}"
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
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(raw_html, encoding="utf-8")
    return G.number_of_nodes(), G.number_of_edges()

PROFILE_DIR = ROOT / "bench_profiles"
OUT_DIR = ROOT / "outputs" / "bench_profiles_graphs"
GRAPH_DIR = OUT_DIR / "graphs"
VIZ_DIR = OUT_DIR / "viz"
MAP_PATH = OUT_DIR / "profile_nct_map.json"


def build_profile_nct_map() -> dict[str, str]:
    """Map profile filename -> NCT id (bijective)."""
    manifest_path = PROFILE_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapping = {v["file"]: nct for nct, v in manifest.items()}
        print(f"Loaded mapping from manifest: {len(mapping)} profiles")
        return mapping

    print("Manifest not found, building mapping from filenames...")
    profiles = {p.name for p in PROFILE_DIR.glob("NCT*.json")}
    mapping = {}
    for p in profiles:
        nct = p.replace(".json", "")
        mapping[p] = nct

    print(f"Mapped {len(mapping)} profiles")
    return mapping


def process_one(
    profile_name: str,
    nct: str,
    *,
    force: bool,
    skip_viz: bool,
    viz_only: bool = False,
) -> dict:
    profile_path = PROFILE_DIR / profile_name
    graph_path = GRAPH_DIR / f"{nct}_graph.json"
    html_path = VIZ_DIR / f"{nct}.html"
    result: dict = {"nct_id": nct, "profile": profile_name, "status": "ok"}

    try:
        if viz_only:
            if not graph_path.is_file():
                raise FileNotFoundError(f"missing graph: {graph_path}")
            result["graph"] = "existing"
        elif force or not graph_path.is_file():
            run_pipeline(str(profile_path), str(graph_path))
        else:
            result["graph"] = "cached"

        if not skip_viz:
            if force or not html_path.is_file() or (
                graph_path.is_file() and graph_path.stat().st_mtime > html_path.stat().st_mtime
            ):
                n_nodes, n_edges = render_html(graph_path, html_path)
                result["nodes"] = n_nodes
                result["edges"] = n_edges
            else:
                result["viz"] = "cached"
                data = load_graph_simple(graph_path)
                result["nodes"] = len(data.get("nodes") or [])
                result["edges"] = len(data.get("edges") or [])
        else:
            # For skip_viz, load and count nodes/edges from graph
            if graph_path.is_file():
                data = load_graph_simple(graph_path)
                result["nodes"] = len(data.get("nodes") or [])
                result["edges"] = len(data.get("edges") or [])

        result["graph_path"] = str(graph_path.relative_to(ROOT))
        result["viz_path"] = str(html_path.relative_to(ROOT))
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    return result


def write_index(results: list[dict], mapping: dict[str, str]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") != "ok"]
    rows = []
    for r in sorted(ok, key=lambda x: x["nct_id"]):
        n = r.get("nodes", "?")
        e = r.get("edges", "?")
        rows.append(
            f'<tr><td><a href="viz/{r["nct_id"]}.html">{r["nct_id"]}</a></td>'
            f'<td>{r["profile"]}</td><td>{n}</td><td>{e}</td></tr>'
        )
    err_rows = "".join(
        f'<tr><td>{r.get("nct_id","?")}</td><td colspan="3">{r.get("error","?")}</td></tr>'
        for r in err
    )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Bench Profiles Knowledge Graph Index</title>
<style>
body {{ font-family: Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dbe1ea; padding: 8px 10px; text-align: left; font-size: 13px; }}
th {{ background: #f4f7fb; }}
.summary {{ margin-bottom: 16px; color: #5b6b7f; }}
</style></head><body>
<h1>Bench Profiles Knowledge Graphs</h1>
<p class="summary">{len(ok)} graphs built · {len(err)} errors · {len(mapping)} profiles</p>
<table>
<thead><tr><th>NCT</th><th>Profile</th><th>Nodes</th><th>Edges</th></tr></thead>
<tbody>{''.join(rows)}{err_rows}</tbody>
</table>
</body></html>"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch build + visualize bench_profiles graphs.")
    ap.add_argument("--force", action="store_true", help="Rebuild even if outputs exist.")
    ap.add_argument("--skip-viz", action="store_true", help="Only build JSON graphs.")
    ap.add_argument(
        "--viz-only",
        action="store_true",
        help="Skip graph build; render HTML for existing graphs only.",
    )
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1).")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N profiles (0=all).")
    ap.add_argument("--nct", action="append", default=[], help="Process specific NCT id(s) only.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    mapping = build_profile_nct_map()
    MAP_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    items = sorted(mapping.items(), key=lambda kv: kv[1])
    if args.nct:
        allow = set(args.nct)
        items = [(p, n) for p, n in items if n in allow]
    if args.limit > 0:
        items = items[: args.limit]

    print(f"Processing {len(items)} profiles...")
    results: list[dict] = []
    if args.workers <= 1:
        for i, (profile_name, nct) in enumerate(items, 1):
            print(f"[{i}/{len(items)}] {nct} <- {profile_name}", flush=True)
            results.append(
                process_one(
                    profile_name,
                    nct,
                    force=args.force,
                    skip_viz=args.skip_viz and not args.viz_only,
                    viz_only=args.viz_only,
                )
            )
    else:
        skip_viz = args.skip_viz and not args.viz_only
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    process_one,
                    profile_name,
                    nct,
                    force=args.force,
                    skip_viz=skip_viz,
                    viz_only=args.viz_only,
                ): (profile_name, nct)
                for profile_name, nct in items
            }
            for i, fut in enumerate(as_completed(futs), 1):
                profile_name, nct = futs[fut]
                r = fut.result()
                print(f"[{i}/{len(items)}] {nct} -> {r.get('status')}", flush=True)
                results.append(r)

    summary_path = OUT_DIR / "build_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_viz:
        write_index(results, mapping)

    ok = sum(1 for r in results if r.get("status") == "ok")
    err = len(results) - ok
    print(f"Done: {ok} ok, {err} errors -> {OUT_DIR.relative_to(ROOT)}")
    if err:
        for r in results:
            if r.get("status") != "ok":
                print(f"  FAIL {r.get('nct_id')}: {r.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
