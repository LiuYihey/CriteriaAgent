# CriteriaAgent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Graph-augmented multi-agent reasoning for clinical trial eligibility criteria** — with **[CriteriaBench](CriteriaBench/)** (78 trials) and progressive subgraph disclosure.

---

## Contents

- [Overview](#overview)
- [Knowledge graph & subgraph extraction](#knowledge-graph--subgraph-extraction)
- [Getting started](#getting-started)
- [CriteriaBench](#criteriabench)
- [Project structure](#project-structure)
- [Documentation](#documentation)
- [Citation & license](#citation--license)

---

## Overview

Eligibility criteria sit at the intersection of **safety**, **efficacy signal**, and **recruitment feasibility** — yet they must be drafted from a sparse protocol stub. CriteriaAgent treats this as a structured reasoning problem rather than one-shot text generation.

| Stage | Module | Role |
|-------|--------|------|
| Input | Trial configuration | Title, arms/interventions, primary endpoint, phase, study type |
| Retrieve | Multi-domain RAG | Disease, drug, literature, and precedent-trial evidence |
| Plan | Task planner | Non-overlapping safety / efficacy / recruitment subtasks |
| Reason | Expert subagents | Graph-routed answers per subtask |
| Write | Criteria writer | Registry-style inclusion & exclusion lists |
| Evaluate | LLM judge + CI | Pairwise quality scoring and expert-criteria agreement |

<p align="center">
  <img src="figures/Overview.png" alt="CriteriaAgent system overview" width="92%" />
</p>

<p align="center"><sub><b>Figure 1.</b> End-to-end pipeline — from trial configuration and multi-domain retrieval, through graph-augmented expert subagents and criteria writing, to post-hoc evaluation.</sub></p>

Each expert subagent does **not** receive the full retrieval dump. Instead, a trial-specific relation graph is built first; experts then query only the subgraph and passages relevant to their subtask (see next section).

---

## Knowledge graph & subgraph extraction

Retrieved evidence is compiled into a **trial-specific relation graph**: every node is anchored to a source passage, and every edge is labeled as textually *extracted* or clinically *inferred*. For each subtask, agentic progressive disclosure proceeds in three layers:

| Layer | What the expert sees |
|-------|----------------------|
| **L0** | Compact node catalog (names + metadata) |
| **L1** | Induced one-hop subgraph around agent-selected seeds |
| **L2** | Deduplicated source passages linked to active nodes |

<p align="center">
  <img src="figures/graph_subgraph_extraction.png" alt="Trial-specific graph construction and agentic subgraph extraction" width="92%" />
</p>

<p align="center"><sub><b>Figure 2.</b> Graph construction (chunk index → LLM extraction → merge → passage anchoring) and per-subtask subgraph disclosure (L0 → L1 → L2).</sub></p>

> Deeper design notes: [Multi-dimensional graph & subgraph extraction](docs/Multi_Dimensional_Graph_and_Subgraph_Extraction.md)

---

## Getting started

### 1 · Install

```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Configure `ANTHROPIC_API_KEY` and `ST_EMBED_MODEL` in `.env` before running.

### 2 · Run a smoke test

```bash
python scripts/run_criteria_agent.py \
  --graph examples/minoxidil/trial_graph.json \
  --config examples/minoxidil/trial_config.json \
  -o outputs/smoke_minoxidil
```

### 3 · Build a graph or run the benchmark

```bash
# Graph from a pre-built RAG profile
python scripts/build_graph.py bench_profiles/<nct_id>.json -o outputs/my_graph.json

# CriteriaBench — direct baseline
python CriteriaBench/run_direct_gen_with_phase.py

# CriteriaBench — full CriteriaAgent pipeline
CRITERIA_BENCH_GEN_MODE=criteria_agent python CriteriaBench/run_criteria_bench_minimax.py
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | API key for generation / judging |
| `ANTHROPIC_BASE_URL` | Anthropic-compatible endpoint (optional) |
| `GRAPH_MODEL` | LLM for graph extraction and agents |
| `ST_EMBED_MODEL` | Embedding model for routing & CI metric |
| `CRITERIA_BENCH_GEN_MODE` | `direct` · `vanilla_rag` · `criteria_agent` |

See [`.env.example`](.env.example) for the full list.

---

## CriteriaBench

A benchmark of **78** completed interventional drug trials from ClinicalTrials.gov (start dates after base-model training cutoffs). Models receive public protocol metadata only; registry expert criteria are withheld until evaluation.

**Evaluation metrics**

| Metric | What it measures |
|--------|------------------|
| Pairwise LLM judge | Safety, efficacy, recruitment (1–10) |
| Consistency Index (CI) | Bullet-level precision × document-level alignment with expert criteria |

**Key paths**

| Resource | Location |
|----------|----------|
| Trial corpus | `CriteriaBench/final_bench/trials/*.json` |
| Corpus summary | `CriteriaBench/final_bench/summary.json` |
| Eval methodology | [benchmark_evaluation_methodology.md](CriteriaBench/docs/benchmark_evaluation_methodology.md) |

```bash
# Scoring scripts
python scripts/run_llm_judge_pairwise_with_phase.py
python scripts/run_consistency_eval.py
```

---

## Project structure

```
criteria_agent/          Planner → expert subagents → criteria writer
trial_graph/             Graph build, subgraph extraction, embeddings
shared/                  LLM client, trial config, CT.gov formatting
scripts/                 CLI entry points
baselines/vanilla_rag/   Single-pass RAG baseline
CriteriaBench/           Benchmark corpus + evaluation runners
bench_profiles/          Pre-built four-domain RAG profiles (78 trials)
examples/minoxidil/      End-to-end smoke test
figures/                 Paper figures (Fig. 1 & 2)
docs/                    Method notes
```

---

## Documentation

- [Method overview](docs/Method%20Overview.md)
- [Multi-dimensional graph & subgraph extraction](docs/Multi_Dimensional_Graph_and_Subgraph_Extraction.md)
- [CriteriaBench evaluation methodology](CriteriaBench/docs/benchmark_evaluation_methodology.md)

---

## Citation & license

If you use this code or CriteriaBench, please cite our paper (forthcoming).

Released under the [MIT License](LICENSE).
