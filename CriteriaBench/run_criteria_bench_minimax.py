"""
CriteriaBench: generate trial eligibility criteria with MiniMax (Anthropic-compatible API),
score rubric dimensions, compare to expert criteria, plot radar summary.

Reads local api_key.md (not for redistribution). Resumes from outputs JSONL if interrupted.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# --- paths ---
ROOT = Path(__file__).resolve().parent
DATA_JSON = ROOT / "filtered_drug_trials.json"
API_KEY_FILE = ROOT / "api_key.md"
OUT_DIR = ROOT / "outputs" / "criteria_bench_minimax"
FIG_DIR = ROOT / "figures"
# Decoupled caches: generation is the expensive part and must be reusable.
GENERATED_JSONL = OUT_DIR / "generated_criteria.jsonl"
REVIEW_JSONL = OUT_DIR / "reviews.jsonl"
# Joined final view (rebuilt from the two caches; cheap to regenerate).
JSONL_PATH = OUT_DIR / "per_trial_results.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

# Pipeline stage selection (default: run whatever is missing for every trial).
# Override via env CRITERIA_BENCH_STAGES="generate,review" or any subset.
DEFAULT_STAGES = ("generate", "review")
STAGES = tuple(
    s.strip().lower()
    for s in os.environ.get("CRITERIA_BENCH_STAGES", ",".join(DEFAULT_STAGES)).split(",")
    if s.strip()
)

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "neutral": "#CFCECE",
}

MAX_WORKERS = int(os.environ.get("CRITERIA_BENCH_WORKERS", "3"))
TRIAL_LIMIT = os.environ.get("CRITERIA_BENCH_LIMIT")  # optional: e.g. "10" for smoke test
GEN_MODE = os.environ.get("CRITERIA_BENCH_GEN_MODE", "direct").strip().lower()
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.ctgov_format import format_arms_interventions, format_primary_outcomes
from shared.llm_client import new_openai_client, resolve_reviewer_llm_config
from shared.trial_config import trial_config_from_study
from criteria_agent.scorer import score_draft

PROFILE_DIR = REPO_ROOT / "data" / "Bench_RAG_profiles"
GRAPH_CACHE_DIR = OUT_DIR / "graphs"

# Review prompt embeds the full Criteria reviewer.md (no length cap).


# REVIEW_SYSTEM removed — scoring now delegated to criteria_agent.scorer




def parse_api_key_file(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    base = re.search(r"ANTHROPIC_BASE_URL=(\S+)", text)
    key_m = re.search(r"API KEY:\s*(\S+)", text, re.I)
    model_m = re.search(r"MODEL:\s*(\S+)", text, re.I)
    if not key_m or not base or not model_m:
        raise RuntimeError(f"Could not parse BASE_URL, API KEY, MODEL from {path}")
    return base.group(1).rstrip("/"), key_m.group(1).strip(), model_m.group(1).strip()


def load_trials() -> list[dict[str, Any]]:
    data = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("filtered_drug_trials.json must be a list")
    return data


def extract_last_json_object(text: str) -> dict[str, Any] | None:
    """Parse the outermost balanced {...} block as JSON.

    We scan left-to-right and try each '{' as the start of a candidate top-level
    object (matching braces with depth counting, ignoring braces inside JSON strings).
    The first candidate that yields a valid JSON object with at least one key is
    returned, so we prefer the OUTERMOST object over any inner nested ones.
    """
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(start, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : j + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and obj:
                        return obj
                    break
    return None


def anthropic_message(
    client: Any,
    model: str,
    user_text: str,
    max_tokens: int = 4096,
    system: str | None = None,
    retries: int = 5,
) -> str:
    """Use streaming: anthropic-sdk rejects long non-streaming calls (~10 min client-side limit)."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": user_text}],
            }
            if system:
                kwargs["system"] = system
            parts: list[str] = []
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    parts.append(text)
            return "".join(parts).strip()
        except Exception as e:
            last_err = e
            wait = min(60, 2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"API failed after retries: {last_err}")


def parse_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I)
    if fence:
        text = fence.group(1).strip()
    last = extract_last_json_object(text)
    if last is not None:
        return last
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group(0))


RUBRIC_KEYS = (
    "signal_enrichment",
    "population_reach",
    "data_evaluability",
    "clinical_feasibility",
    "risk_mitigation",
)


def _coerce_half_step(x: Any, lo: float, hi: float, name: str) -> float:
    """Coerce to [lo, hi] in 0.5 steps."""
    v = float(x)
    if v < lo or v > hi:
        raise ValueError(f"{name} out of {lo}-{hi}: {v}")
    v = round(v * 2.0) / 2.0
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v


def _rubric_from_obj(obj: dict[str, Any]) -> dict[str, float]:
    """Validate scorer output — each dimension must be a number."""
    scores: dict[str, float] = {}
    for k in RUBRIC_KEYS:
        if k not in obj:
            raise ValueError(f"missing key {k}")
        v = obj[k]
        if isinstance(v, dict):
            # legacy {score, issues} format — extract score
            if "score" not in v:
                raise ValueError(f"{k}.score missing")
            scores[k] = _coerce_half_step(v["score"], 1.0, 10.0, k)
        else:
            scores[k] = _coerce_half_step(v, 1.0, 10.0, k)
    return scores


# parse_rubric_scores removed — scoring now via criteria_agent.scorer (code-level JSON)











def gen_prompt_for_trial(
    trial: dict[str, Any],
) -> tuple[str, str]:
    ps = trial.get("protocolSection") or {}
    im = ps.get("identificationModule") or {}
    arms_mod = ps.get("armsInterventionsModule") or {}

    title_line = (im.get("officialTitle") or im.get("briefTitle") or "").strip()
    primary_block = format_primary_outcomes(ps.get("outcomesModule"))
    protocol = format_arms_interventions(arms_mod)

    gen_prompt = f"""You are an experienced clinical trialist. Draft **eligibility criteria** (Inclusion and Exclusion) for this study, in English, using bullet lists similar to ClinicalTrials.gov style.

Use only the information below (do not invent trial-specific numerical thresholds unless they are clearly implied by the title or interventions; prefer ranges only when justified).

1) **Clinical trial title:** {title_line}

2) **Arms and interventions (registry, armsInterventionsModule):**
{protocol}

3) **Primary outcomes (registry, outcomesModule.primaryOutcomes):**
{primary_block}

Output sections:
### Inclusion Criteria
(bullets)

### Exclusion Criteria
(bullets)
"""

    return gen_prompt, primary_block


def load_done_nct_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj.get("nct_id", ""))
            except json.JSONDecodeError:
                continue
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl_by_nct(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            nid = row.get("nct_id")
            if nid:
                out[nid] = row  # last row wins
    return out


def safe_backup(path: Path) -> None:
    """Per .cursor/rules/safe-deletes: never permanently delete; rename to .bak-<ts>."""
    if not path.is_file():
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak-{ts}")
    path.rename(bak)
    print(f"[safe-backup] {path.name} -> {bak.name}", flush=True)


def rewrite_dedup(path: Path) -> None:
    """Dedupe by nct_id (last wins) and rewrite atomically. Never deletes the live file."""
    if not path.is_file():
        return
    by_nct = load_jsonl_by_nct(path)
    if not by_nct:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in by_nct.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------- Generation backends ----------------

def find_rag_profile(nct: str) -> Path | None:
    if not nct or not PROFILE_DIR.is_dir():
        return None
    hits = sorted(PROFILE_DIR.glob(f"*{nct}*.json"))
    return hits[0] if hits else None


def ensure_trial_graph(profile_path: Path, nct: str) -> Path:
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = GRAPH_CACHE_DIR / f"{nct}_graph.json"
    if out.is_file():
        return out
    sys.path.insert(0, str(REPO_ROOT))
    from trial_graph.build import run_pipeline

    run_pipeline(str(profile_path), str(out))
    return out


def generate_criteria_for_trial(
    trial: dict[str, Any], client: Any, model: str,
) -> str:
    mode = GEN_MODE
    if mode == "direct":
        gen_p, _ = gen_prompt_for_trial(trial)
        return anthropic_message(client, model, gen_p, max_tokens=8192)

    ps = trial.get("protocolSection") or {}
    im = ps.get("identificationModule") or {}
    nct = (im.get("nctId") or "UNKNOWN").strip()
    profile = find_rag_profile(nct)
    if profile is None:
        raise RuntimeError(
            f"{nct}: no data/Bench_RAG_profiles/*{nct}*.json — required for gen_mode={mode}"
        )

    sys.path.insert(0, str(REPO_ROOT))
    from shared.trial_config import trial_config_from_study

    config = trial_config_from_study(trial)

    if mode == "vanilla_rag":
        from baselines.vanilla_rag.prompts import VANILLA_RAG_SYSTEM, build_user_prompt
        from shared.llm_client import call_text

        prof_obj = json.loads(profile.read_text(encoding="utf-8"))
        user = build_user_prompt(config.title, config.arms_text, config.primary_text, prof_obj)
        return call_text(client, system=VANILLA_RAG_SYSTEM, user=user, model=model)

    if mode == "criteria_agent":
        from criteria_agent.pipeline import run_pipeline
        import tempfile

        graph_path = ensure_trial_graph(profile, nct)
        skip_opt = os.environ.get("CRITERIA_BENCH_SKIP_OPTIMIZER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        with tempfile.TemporaryDirectory(prefix=f"criteria_agent_{nct}_") as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "title": config.title,
                        "arms": config.arms_text,
                        "primary_outcome": config.primary_text,
                        "nct_id": nct,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            out_dir = Path(tmp) / "out"
            trace = run_pipeline(
                graph_path,
                cfg_path,
                out_dir,
                skip_optimizer=skip_opt,
            )
            return str(trace.get("criteria_final") or (out_dir / "criteria_final.md").read_text(encoding="utf-8"))

    raise ValueError(f"Unknown CRITERIA_BENCH_GEN_MODE={mode!r} (use direct, vanilla_rag, criteria_agent)")


# ---------------- Stage 1: generation ----------------

def stage_generate(
    trial: dict[str, Any], client: Any, model: str,
) -> dict[str, Any]:
    ps = trial.get("protocolSection") or {}
    im = ps.get("identificationModule") or {}
    em = ps.get("eligibilityModule") or {}
    nct = im.get("nctId") or "UNKNOWN"
    _, primary_block = gen_prompt_for_trial(trial)
    generated = generate_criteria_for_trial(trial, client, model)
    return {
        "nct_id": nct,
        "title_used": (im.get("officialTitle") or im.get("briefTitle") or "").strip(),
        "primary_endpoint_hint": primary_block,
        "expert_eligibility_criteria": (em.get("eligibilityCriteria") or "").strip(),
        "generated_criteria": generated,
        "gen_mode": GEN_MODE,
    }


# ---------------- Stage 2: review ----------------

def stage_review(
    trial: dict[str, Any],
    generated: str,
    reviewer_client: Any,
    reviewer_model: str | None,
) -> dict[str, Any]:
    """Score generated criteria using the shared agent scorer (full reuse)."""
    ps = trial.get("protocolSection") or {}
    im = ps.get("identificationModule") or {}
    nct = im.get("nctId") or "UNKNOWN"
    config = trial_config_from_study(trial)
    result = score_draft(reviewer_client, config, generated, model=reviewer_model)
    try:
        _rubric_from_obj(result["scores"])
    except Exception:
        pass  # scorer enforces JSON at code level; validation is best-effort
    return {
        "nct_id": nct,
        "scores": result["scores"],
        "raw_review": result["raw_json"],
    }





def _run_pool(
    items: list[Any],
    fn,
    out_path: Path,
    desc: str,
    workers: int,
) -> None:
    """Parallel runner that appends each successful row to its own cache jsonl."""
    if not items:
        return
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            try:
                row = fut.result()
            except Exception as e:
                it = futs[fut]
                # try to recover nct from trial dict or tuple (trial, generated)
                trial = it[0] if isinstance(it, tuple) else it
                nid = (
                    (trial.get("protocolSection") or {})
                    .get("identificationModule", {})
                    .get("nctId")
                )
                print(f"FAIL[{desc}] {nid}: {e}", file=sys.stderr, flush=True)
                continue
            append_jsonl(out_path, row)


def plot_radar(
    means: dict[str, float],
    out_base: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [
        "Signal enrichment",
        "Population reach",
        "Data evaluability",
        "Clinical feasibility",
        "Risk mitigation",
    ]
    keys = [
        "signal_enrichment",
        "population_reach",
        "data_evaluability",
        "clinical_feasibility",
        "risk_mitigation",
    ]
    values = [means[k] for k in keys]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 14,
            "axes.linewidth": 2.0,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(1, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=9)
    pol = ax.spines.get("polar")
    if pol is not None:
        pol.set_visible(False)

    ax.plot(angles, values, "o-", linewidth=2.5, color=PALETTE["blue_main"])
    ax.fill(angles, values, alpha=0.22, color=PALETTE["blue_secondary"])

    ax.set_title(
        "CriteriaBench: mean rubric scores (1–10)",
        y=1.08,
        fontsize=14,
    )
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)





def load_joined_rows_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL; duplicate nct_id keeps last occurrence."""
    by_nct: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            nid = row.get("nct_id")
            if nid:
                by_nct[nid] = row
    return list(by_nct.values())





def write_reports_for_rows(
    joined_rows: list[dict[str, Any]],
    model: str,
) -> None:
    """Recompute summary.json from joined trial rows (no API)."""
    scored_rows = [r for r in joined_rows if "scores" in r]

    keys = [
        "signal_enrichment",
        "population_reach",
        "data_evaluability",
        "clinical_feasibility",
        "risk_mitigation",
    ]
    import numpy as np

    means = {k: float(np.mean([r["scores"][k] for r in scored_rows])) for k in keys} if scored_rows else {}

    summary = {
        "model": model,
        "n_trials": len(scored_rows),
        "mean_rubric_scores_1_to_10": means,
        "results_jsonl": str(JSONL_PATH),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def rebuild_reports_only() -> None:
    """Rebuild summary from existing per_trial_results.jsonl."""
    joined = load_joined_rows_jsonl(JSONL_PATH)
    model = "unknown_model"
    if SUMMARY_PATH.is_file():
        try:
            prev = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
            model = str(prev.get("model", model))
        except json.JSONDecodeError:
            pass
    write_reports_for_rows(joined, model)


def main() -> None:
    base_url, api_key, model = parse_api_key_file(API_KEY_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Reviewer uses OpenAI-compatible endpoint (shared with agent scorer).
    _, _, reviewer_model = resolve_reviewer_llm_config()

    trials = load_trials()
    if TRIAL_LIMIT:
        trials = trials[: int(TRIAL_LIMIT)]
    by_id = {
        (t.get("protocolSection") or {}).get("identificationModule", {}).get("nctId"): t
        for t in trials
    }

    print(
        f"[plan] trials={len(trials)}  stages={list(STAGES)}  gen_mode={GEN_MODE}  workers={MAX_WORKERS}",
        flush=True,
    )

    def new_client() -> Any:
        import anthropic as _anthropic

        return _anthropic.Anthropic(api_key=api_key, base_url=base_url)

    # ----- Stage 1: generation -----
    if "generate" in STAGES:
        rewrite_dedup(GENERATED_JSONL)
        have = load_jsonl_by_nct(GENERATED_JSONL)
        pending = [t for nid, t in by_id.items() if nid and nid not in have]
        print(f"[generate] have={len(have)}  pending={len(pending)}", flush=True)

        def _gen(tr: dict[str, Any]) -> dict[str, Any]:
            return stage_generate(tr, new_client(), model)

        _run_pool(pending, _gen, GENERATED_JSONL, "generate", MAX_WORKERS)
        rewrite_dedup(GENERATED_JSONL)

    gen_by_id = load_jsonl_by_nct(GENERATED_JSONL)

    # ----- Stage 2: review -----
    if "review" in STAGES:
        rewrite_dedup(REVIEW_JSONL)
        have = load_jsonl_by_nct(REVIEW_JSONL)
        targets: list[tuple[dict[str, Any], str]] = []
        for nid, grow in gen_by_id.items():
            if nid in have:
                continue
            tr = by_id.get(nid)
            if tr is None:
                continue
            targets.append((tr, grow["generated_criteria"]))
        print(f"[review]   have={len(have)}  pending={len(targets)}", flush=True)

        def _rev(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
            tr, gen = item
            return stage_review(tr, gen, new_openai_client(), reviewer_model)

        _run_pool(targets, _rev, REVIEW_JSONL, "review", MAX_WORKERS)
        rewrite_dedup(REVIEW_JSONL)

    # ----- Join the two caches into per_trial_results.jsonl -----
    gen_by_id = load_jsonl_by_nct(GENERATED_JSONL)
    rev_by_id = load_jsonl_by_nct(REVIEW_JSONL)

    # Back up the previous joined file (never delete; per safe-deletes rule).
    safe_backup(JSONL_PATH)

    joined_rows: list[dict[str, Any]] = []
    for nid, grow in gen_by_id.items():
        row: dict[str, Any] = dict(grow)
        rrow = rev_by_id.get(nid)
        if rrow:
            row.update(
                scores=rrow["scores"],
            )
        joined_rows.append(row)

    if joined_rows:
        with JSONL_PATH.open("w", encoding="utf-8") as f:
            for r in joined_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_reports_for_rows(joined_rows, model)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reports-only":
        rebuild_reports_only()
    else:
        main()
