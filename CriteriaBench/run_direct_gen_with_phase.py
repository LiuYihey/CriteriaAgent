"""
CriteriaBench — direct generation variant.

Reuses the existing direct-generation pipeline from
``run_criteria_bench_minimax.py`` (same Anthropic-compatible client, same
streaming call, same parallel runner, same JSONL resume) and extends the
input prompt with two extra trial-config fields:

  * ``designModule.studyType``   — e.g. INTERVENTIONAL / OBSERVATIONAL
  * ``designModule.phases``      — e.g. ["PHASE2"], ["PHASE1", "PHASE2"]

The 78 trial samples in ``CriteriaBench/final_bench/trials`` are processed
with the ``MiniMax-M2.7`` model configured in ``CriteriaBench/api_key.md``.

Output: ``CriteriaBench/outputs/direct_gen_with_phase/generated_criteria.jsonl``
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

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

# --- paths ---
TRIALS_DIR = ROOT / "final_bench" / "trials"
API_KEY_FILE = ROOT / "api_key.md"
OUT_DIR = ROOT / "outputs" / "direct_gen_with_phase"
GENERATED_JSONL = OUT_DIR / "generated_criteria.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

MAX_WORKERS = int(os.environ.get("CRITERIA_BENCH_WORKERS", "3"))
TRIAL_LIMIT = os.environ.get("CRITERIA_BENCH_LIMIT")
MAX_TOKENS = int(os.environ.get("CRITERIA_BENCH_MAX_TOKENS", "16384"))


def parse_api_key_file(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    base = re.search(r"ANTHROPIC_BASE_URL=(\S+)", text)
    key_m = re.search(r"API KEY:\s*(\S+)", text, re.I)
    model_m = re.search(r"MODEL:\s*(\S+)", text, re.I)
    if not key_m or not base or not model_m:
        raise RuntimeError(f"Could not parse BASE_URL, API KEY, MODEL from {path}")
    return base.group(1).rstrip("/"), key_m.group(1).strip(), model_m.group(1).strip()


def load_trials_from_dir(trials_dir: Path) -> list[dict[str, Any]]:
    files = sorted(trials_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No trial JSON files in {trials_dir}")
    out: list[dict[str, Any]] = []
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[skip] cannot parse {fp.name}: {e}", file=sys.stderr, flush=True)
            continue
        if not isinstance(obj, dict):
            print(f"[skip] {fp.name}: top-level is not a dict", file=sys.stderr, flush=True)
            continue
        out.append(obj)
    return out


def extract_trial_meta(trial: dict[str, Any]) -> dict[str, str]:
    """Return the (studyType, phases) for a CTGOV-style trial dict."""
    ps = trial.get("protocolSection") or {}
    design = ps.get("designModule") or {}
    study_type = str(design.get("studyType") or "").strip() or "(not reported)"
    phases = design.get("phases") or []
    if isinstance(phases, list):
        phases_clean = [str(p).strip() for p in phases if str(p).strip()]
        phase_text = ", ".join(phases_clean) if phases_clean else "(not reported)"
    else:
        phase_text = str(phases).strip() or "(not reported)"
    return {"study_type": study_type, "phase": phase_text}


def gen_prompt_with_phase(trial: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Build the direct-gen prompt, augmented with studyType + phase(s).

    Mirrors ``gen_prompt_for_trial`` from ``run_criteria_bench_minimax.py`` so
    the only change is the two extra sections the model sees.
    """
    from shared.ctgov_format import format_arms_interventions, format_primary_outcomes

    ps = trial.get("protocolSection") or {}
    im = ps.get("identificationModule") or {}
    arms_mod = ps.get("armsInterventionsModule") or {}

    title_line = (im.get("officialTitle") or im.get("briefTitle") or "").strip()
    primary_block = format_primary_outcomes(ps.get("outcomesModule"))
    protocol = format_arms_interventions(arms_mod)
    meta = extract_trial_meta(trial)

    gen_prompt = f"""You are an experienced clinical trialist. Draft **eligibility criteria** (Inclusion and Exclusion) for this study, in English, using bullet lists similar to ClinicalTrials.gov style.

Use only the information below (do not invent trial-specific numerical thresholds unless they are clearly implied by the title or interventions; prefer ranges only when justified).

1) **Clinical trial title:** {title_line}

2) **Trial type (registry, designModule.studyType):** {meta['study_type']}

3) **Trial phase (registry, designModule.phases):** {meta['phase']}

4) **Arms and interventions (registry, armsInterventionsModule):**
{protocol}

5) **Primary outcomes (registry, outcomesModule.primaryOutcomes):**
{primary_block}

Output sections:
### Inclusion Criteria
(bullets)

### Exclusion Criteria
(bullets)
"""

    return gen_prompt, meta


def anthropic_message(
    client: Any,
    model: str,
    user_text: str,
    max_tokens: int = 16384,
    retries: int = 5,
) -> str:
    """Anthropic-compatible streaming call. Reused from the existing direct-gen
    pipeline so behaviour is identical (10-minute stream cap, exponential back-off).

    If a call returns empty text (model exhausted output tokens on thinking,
    transient empty chunk, etc.) we retry the same prompt with a doubled
    ``max_tokens`` budget. This mirrors the recovery logic in
    ``shared.llm_client.call_text``.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            parts: list[str] = []
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_text}],
            ) as stream:
                for text in stream.text_stream:
                    parts.append(text)
            text = "".join(parts).strip()
            if text:
                return text
            # Empty result — bump budget and try again (model may be using the
            # whole budget on thinking). One retry is enough in practice.
            if max_tokens < 32768:
                print(
                    f"[anthropic_message] empty reply at {max_tokens} tokens, retrying with 2x",
                    flush=True,
                )
                max_tokens = min(32768, max_tokens * 2)
                continue
        except Exception as e:
            last_err = e
            wait = min(60, 2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"API failed after retries: {last_err}")


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
                out[nid] = row
    return out


def prune_empty(path: Path) -> int:
    """Drop rows whose ``generated_criteria`` is empty / whitespace only.

    A model that hit the thinking-truncation issue returns a row with
    non-empty metadata but empty content. Without pruning, the dedup
    "have" set would treat those as done and they would never be retried.
    Returns the number of rows removed.
    """
    if not path.is_file():
        return 0
    kept: list[str] = []
    removed = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            gc = (obj.get("generated_criteria") or "").strip()
            if not gc:
                removed += 1
                continue
            kept.append(raw)
    if removed == 0:
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(r + "\n")
    tmp.replace(path)
    print(f"[prune-empty] removed {removed} empty rows from {path.name}", flush=True)
    return removed


def rewrite_dedup(path: Path) -> None:
    """Dedupe by nct_id (last wins) and rewrite atomically."""
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


def stage_generate(trial: dict[str, Any], client: Any, model: str) -> dict[str, Any]:
    ps = trial.get("protocolSection") or {}
    im = ps.get("identificationModule") or {}
    em = ps.get("eligibilityModule") or {}
    nct = im.get("nctId") or "UNKNOWN"
    prompt, meta = gen_prompt_with_phase(trial)
    generated = anthropic_message(client, model, prompt, max_tokens=MAX_TOKENS)
    return {
        "nct_id": nct,
        "title_used": (im.get("officialTitle") or im.get("briefTitle") or "").strip(),
        "study_type": meta["study_type"],
        "phase": meta["phase"],
        "prompt_sent": prompt,
        "expert_eligibility_criteria": (em.get("eligibilityCriteria") or "").strip(),
        "generated_criteria": generated,
        "gen_mode": "direct_with_phase",
    }


def _run_pool(items: list[Any], fn, out_path: Path, desc: str, workers: int) -> None:
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
                trial = it[0] if isinstance(it, tuple) else it
                nid = (
                    (trial.get("protocolSection") or {})
                    .get("identificationModule", {})
                    .get("nctId")
                )
                print(f"FAIL[{desc}] {nid}: {e}", file=sys.stderr, flush=True)
                continue
            append_jsonl(out_path, row)


def main() -> None:
    base_url, api_key, model = parse_api_key_file(API_KEY_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trials = load_trials_from_dir(TRIALS_DIR)
    if TRIAL_LIMIT:
        trials = trials[: int(TRIAL_LIMIT)]
    by_id: dict[str, dict[str, Any]] = {}
    for t in trials:
        nid = (t.get("protocolSection") or {}).get("identificationModule", {}).get("nctId")
        if nid:
            by_id[nid] = t

    print(
        f"[plan] trials={len(by_id)}  gen_mode=direct_with_phase  workers={MAX_WORKERS}  "
        f"model={model}  max_tokens={MAX_TOKENS}",
        flush=True,
    )

    def new_client() -> Any:
        import anthropic as _anthropic

        return _anthropic.Anthropic(api_key=api_key, base_url=base_url)

    rewrite_dedup(GENERATED_JSONL)
    prune_empty(GENERATED_JSONL)
    have = load_jsonl_by_nct(GENERATED_JSONL)
    pending = [t for nid, t in by_id.items() if nid and nid not in have]
    print(f"[generate] have={len(have)}  pending={len(pending)}", flush=True)

    def _gen(tr: dict[str, Any]) -> dict[str, Any]:
        return stage_generate(tr, new_client(), model)

    _run_pool(pending, _gen, GENERATED_JSONL, "direct_gen_with_phase", MAX_WORKERS)
    rewrite_dedup(GENERATED_JSONL)

    rows = list(load_jsonl_by_nct(GENERATED_JSONL).values())
    summary = {
        "model": model,
        "n_trials": len(rows),
        "gen_mode": "direct_with_phase",
        "input_trials_dir": str(TRIALS_DIR),
        "output_jsonl": str(GENERATED_JSONL),
        "nct_ids": sorted(r["nct_id"] for r in rows),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("model", "n_trials", "gen_mode")}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
