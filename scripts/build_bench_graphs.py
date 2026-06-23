#!/usr/bin/env python3
"""Build trial knowledge graphs + HTML visualizations for all CriteriaBench RAG profiles."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trial_graph.build import run_pipeline  # noqa: E402
from trial_graph.visualize import (  # noqa: E402
    build_banner_html,
    build_digraph,
    load_graph,
    to_pyvis,
    PAGE_CSS,
)

PROFILE_DIR = ROOT / "data" / "Bench_RAG_profiles"
TRIALS_DIR = ROOT / "CriteriaBench" / "final_bench" / "trials"
OUT_DIR = ROOT / "outputs" / "bench_graphs"
GRAPH_DIR = OUT_DIR / "graphs"
VIZ_DIR = OUT_DIR / "viz"
MAP_PATH = OUT_DIR / "profile_nct_map.json"


def _slug(title: str, max_len: int) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len].rstrip("_")


def _profile_candidates(title: str, nct: str) -> list[str]:
    names: list[str] = []
    for max_len in range(50, 19, -1):
        names.append(f"trial_profile_{_slug(title, max_len)}.json")
    for max_len in range(40, 19, -1):
        names.append(f"trial_profile_{_slug(title, max_len)}_{nct}.json")
    for max_len in range(35, 19, -1):
        names.append(f"trial_profile_{_slug(title, max_len)}_{nct[-8:]}.json")
    # observed pattern: truncated slug + _NCT + full id
    for max_len in range(40, 19, -1):
        names.append(f"trial_profile_{_slug(title, max_len)}_{nct}.json".replace(
            f"_{nct}.json", f"_{nct}.json"
        ))
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_profile_nct_map() -> dict[str, str]:
    """Map profile filename -> NCT id (bijective over bench set)."""
    profiles = {p.name: p for p in PROFILE_DIR.glob("*.json")}
    trials: list[tuple[str, str]] = []
    for p in sorted(TRIALS_DIR.glob("*.json")):
        t = json.loads(p.read_text(encoding="utf-8"))
        ps = t["protocolSection"]
        nct = ps["identificationModule"]["nctId"]
        title = ps["identificationModule"].get("officialTitle", "")
        trials.append((nct, title))

    mapping: dict[str, str] = {}
    used_profiles: set[str] = set()

    # Pass 1: NCT embedded in filename
    for nct, title in trials:
        hits = [name for name in profiles if nct in name and name not in used_profiles]
        if len(hits) == 1:
            mapping[hits[0]] = nct
            used_profiles.add(hits[0])

    # Pass 2: expected filename from title slug
    for nct, title in trials:
        if nct in mapping.values():
            continue
        for cand in _profile_candidates(title, nct):
            if cand in profiles and cand not in used_profiles:
                mapping[cand] = nct
                used_profiles.add(cand)
                break

    # Pass 3: unique longest common prefix between remaining slug and title
    remaining_trials = [(n, t) for n, t in trials if n not in mapping.values()]
    remaining_profiles = [n for n in profiles if n not in used_profiles]

    def prefix_score(profile_name: str, title: str) -> float:
        stem = profile_name.replace("trial_profile_", "").replace(".json", "")
        stem = re.sub(r"_NCT\d+$", "", stem)
        slug = _slug(title, 50)
        common = 0
        for a, b in zip(stem.lower(), slug.lower()):
            if a == b:
                common += 1
            else:
                break
        return common / max(len(stem), 1)

    for nct, title in remaining_trials:
        scored = sorted(
            ((prefix_score(pn, title), pn) for pn in remaining_profiles),
            reverse=True,
        )
        if scored and scored[0][0] >= 0.35:
            best = scored[0][1]
            mapping[best] = nct
            used_profiles.add(best)
            remaining_profiles.remove(best)

    if len(mapping) != len(profiles):
        unmapped_profiles = sorted(set(profiles) - set(mapping))
        unmapped_ncts = sorted({n for n, _ in trials} - set(mapping.values()))
        raise RuntimeError(
            f"Profile↔NCT mapping incomplete: {len(mapping)}/{len(profiles)} profiles, "
            f"unmapped_profiles={unmapped_profiles[:5]}, unmapped_ncts={unmapped_ncts[:5]}"
        )
    return mapping


def render_html(graph_path: Path, html_path: Path) -> tuple[int, int]:
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
            if force or not html_path.is_file() or graph_path.stat().st_mtime > html_path.stat().st_mtime:
                n_nodes, n_edges = render_html(graph_path, html_path)
                result["nodes"] = n_nodes
                result["edges"] = n_edges
            else:
                result["viz"] = "cached"
                data = load_graph(graph_path)
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
<html lang="en"><head><meta charset="utf-8"><title>CriteriaBench Graph Index</title>
<style>
body {{ font-family: Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #dbe1ea; padding: 8px 10px; text-align: left; font-size: 13px; }}
th {{ background: #f4f7fb; }}
.summary {{ margin-bottom: 16px; color: #5b6b7f; }}
</style></head><body>
<h1>CriteriaBench Knowledge Graphs</h1>
<p class="summary">{len(ok)} graphs built · {len(err)} errors · {len(mapping)} profiles</p>
<table>
<thead><tr><th>NCT</th><th>Profile</th><th>Nodes</th><th>Edges</th></tr></thead>
<tbody>{''.join(rows)}{err_rows}</tbody>
</table>
</body></html>"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch build + visualize CriteriaBench trial graphs.")
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
    print(f"Mapped {len(mapping)} profiles -> NCT ids ({MAP_PATH.relative_to(ROOT)})")

    items = sorted(mapping.items(), key=lambda kv: kv[1])
    if args.nct:
        allow = set(args.nct)
        items = [(p, n) for p, n in items if n in allow]
    if args.limit > 0:
        items = items[: args.limit]

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
