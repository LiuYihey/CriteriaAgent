# CriteriaAgent

Multi-agent **clinical trial eligibility criteria** generation with a trial-specific knowledge graph, progressive subgraph disclosure, and **CriteriaBench** evaluation.

## Repository layout

```
CriteriaAgent/
├── criteria_agent/       # Planner → expert subagents → criteria writer
├── trial_graph/          # Graph construction, subgraph extraction, embeddings
├── shared/               # LLM client, trial config, ClinicalTrials.gov formatting
├── scripts/              # CLI entry points (generation, graph build, evaluation)
├── baselines/vanilla_rag/  # Single-pass RAG baseline
├── CriteriaBench/        # 78-trial benchmark corpus + evaluation runners
├── bench_profiles/       # Pre-built four-domain RAG profiles (78 trials)
├── examples/minoxidil/   # Small end-to-end smoke test
└── docs/                 # Method overview and graph design notes
```

## Quick start

```powershell
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your API key and embedding model path
```

**Smoke test (CriteriaAgent pipeline):**

```powershell
python scripts/run_criteria_agent.py `
  --graph examples/minoxidil/trial_graph.json `
  --config examples/minoxidil/trial_config.json `
  -o outputs/smoke_minoxidil
```

**Build a trial graph from a RAG profile:**

```powershell
python scripts/build_graph.py bench_profiles/<nct_id>.json -o outputs/my_graph.json
```

**CriteriaBench — direct generation baseline:**

```powershell
python CriteriaBench/run_direct_gen_with_phase.py
```

**CriteriaBench — full CriteriaAgent pipeline:**

```powershell
$env:CRITERIA_BENCH_GEN_MODE="criteria_agent"
python CriteriaBench/run_criteria_bench_minimax.py
```

**Evaluation (pairwise LLM judge + Consistency Index):**

```powershell
python scripts/run_llm_judge_pairwise_with_phase.py
python scripts/run_consistency_eval.py
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | API key for generation / judging |
| `ANTHROPIC_BASE_URL` | Anthropic-compatible endpoint (optional) |
| `GRAPH_MODEL` | LLM used for graph extraction and agents |
| `ST_EMBED_MODEL` | Local embedding model for graph routing & CI metric |
| `CRITERIA_BENCH_GEN_MODE` | `direct`, `vanilla_rag`, or `criteria_agent` |

See `.env.example` for the full list.

## CriteriaBench

The benchmark comprises **78** completed interventional drug trials drawn from ClinicalTrials.gov with start dates after base-model training cutoffs. Each trial provides generation inputs (title, arms/interventions, primary outcome, phase, study type); registry expert criteria are withheld until evaluation.

Construction script: `CriteriaBench/create_new_bench.py` (requires a local ClinicalTrials.gov snapshot; not shipped in this repo).

Corpus files: `CriteriaBench/final_bench/trials/*.json`, `CriteriaBench/final_bench/summary.json`.

## Documentation

- [Method overview](docs/Method%20Overview.md)
- [Multi-dimensional graph & subgraph extraction](docs/Multi_Dimensional_Graph_and_Subgraph_Extraction.md)
- [CriteriaBench evaluation methodology](CriteriaBench/docs/benchmark_evaluation_methodology.md)

## Citation

If you use this code or CriteriaBench, please cite our paper (forthcoming).

## License

TBD.
