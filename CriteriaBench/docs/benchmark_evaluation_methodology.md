# CriteriaBench: Evaluation Methodology

This document summarizes how **CriteriaBench** evaluates large language models (LLMs) on the task of drafting clinical-trial eligibility criteria. The benchmark measures both **intrinsic quality** of generated criteria and **alignment** with registry expert criteria, using a three-stage, LLM-as-judge pipeline.

---

## 1. Task Definition

Given structured protocol metadata from ClinicalTrials.gov—trial title, arms and interventions, and the first primary outcome—the model must produce **Inclusion Criteria** and **Exclusion Criteria** in English, formatted as bullet lists in the style of registry entries.

**Closed-book generation.** The model does not see the registry `eligibilityCriteria` field during generation. Expert criteria are revealed only in the third evaluation stage. This design tests whether a model can infer appropriate eligibility rules from protocol context alone, rather than paraphrasing a known answer.

---

## 2. Benchmark Corpus

Trials are drawn from a filtered subset of ClinicalTrials.gov studies (`For_bench_ctg-studies.json`). Retention rules are:

| Criterion | Requirement |
|-----------|-------------|
| Status | `COMPLETED` |
| Start date | On or after 2025-09-01 |
| Primary completion | On or before 2026-05-09 |
| Intervention type | At least one `DRUG` intervention |
| Drug naming | Not excluded as candidate-code-only (e.g., `MK-4646`, `LW402`) |
| Protocol modules | Non-empty primary outcomes and arms/interventions modules |

The resulting bench-ready set is written to `filtered_drug_trials.json`. Each trial supplies generation inputs from `protocolSection` and holds expert criteria in `eligibilityModule.eligibilityCriteria` for agreement scoring only.

---

## 3. Evaluation Pipeline

Evaluation proceeds in three decoupled stages. Stages 2 and 3 depend on Stage 1 output but not on each other; results are cached per trial and joined for reporting.

```
Protocol metadata  ──►  [Stage 1: Generate]  ──►  Draft criteria
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      [Stage 2: Quality review]     [Stage 3: Expert agreement]
              │                               │
              ▼                               ▼
      5-dimension rubric (1–10)       3 agreement subscores (each 0–10)
```

### Stage 1 — Generation

The model receives:

1. Official or brief trial title  
2. Full arms and interventions text (`armsInterventionsModule`)  
3. The **first** primary outcome only (`outcomesModule.primaryOutcomes[0]`)

The prompt instructs the model to act as an experienced clinical trialist, avoid inventing trial-specific numeric thresholds unless clearly implied, and output `### Inclusion Criteria` and `### Exclusion Criteria` sections.

### Stage 2 — Quality Review (Intrinsic Rubric)

A separate LLM call scores the **generated** criteria against a five-dimension quality rubric (`Criteria reviewer.md`). The same trial context (title, arms/interventions, primary outcome) is provided alongside the draft.

| Dimension | Objective |
|-----------|-----------|
| **Signal enrichment** | Mechanistic alignment and cohort enrichment for detectable treatment effects |
| **Population reach** | Real-world enrollability given epidemiology and criteria stringency |
| **Data evaluability** | Clean, measurable endpoints; control of attrition and competing risks |
| **Clinical feasibility** | Screening burden aligned with standard-of-care workflows and site capacity |
| **Risk mitigation** | Safety boundaries, organ-function screens, and regulatory guardrails |

**Scoring procedure.** Each dimension starts at 10.0 and receives deductions in 0.5-step increments for concrete issues (minor: −0.5; meaningful gap: −1.0; substantive flaw: −1.5; critical: −2.0+). Scores are clipped to [1.0, 10.0]. The grader outputs structured JSON: per-dimension `score` and up to five `issues` (≤14 words each). A repair pass is triggered if JSON parsing fails.

### Stage 3 — Expert Agreement (Reference Alignment)

A third LLM call compares the closed-book draft to registry expert criteria for the **same** trial. Judgment is on **clinical substance**, not wording, ordering, or bullet count; paraphrase counts as a match.

Three sub-dimensions are scored on [0.0, 10.0] in 0.5 steps. **There is no pooled or weighted aggregate agreement score:** benchmarks and summaries report these three subscores only.

| Sub-dimension | What it measures |
|---------------|------------------|
| **Inclusion coverage** | Fraction of independently checkable Expert inclusion concepts reflected in the AI draft |
| **Exclusion coverage** | Same for Expert exclusions and safety screens (organ function, pregnancy, washout, DDI, etc.) |
| **Quantitative alignment** | Agreement on numeric intervals, ordered categorical levels (ECOG, Child-Pugh, line of therapy), and time/count windows |

Coverage sub-scores map from the ratio *M/C* (matched Expert units / total Expert units) to a 0–10 band, with flat penalties for direct contradictions or missing safety clusters. Quantitative alignment averages per-item alignment classes (identical, mostly overlapping, partial, contradictory, or one-sided).

---

## 4. Reported Metrics

Per-trial outputs are stored in JSONL caches (`generated_criteria.jsonl`, `reviews.jsonl`, `agreements.jsonl`) and joined into `per_trial_results.jsonl`. Summary statistics (`summary.json`) report:

- **Mean rubric scores** — arithmetic mean of each five-dimension score across trials (scale 1–10)  
- **Mean agreement subscores** — arithmetic mean of each of the three agreement dimensions (scale 0–10 each)  
- **Trial count** — number of trials with complete generation, review, and agreement records  

Visualization (see `final_bench/figures/` after running `finalize_final_bench.py`): a **radar chart** for mean rubric dimensions (1–10), and **horizontal bar charts with mean ± SEM** for the three agreement subscores (house style: discrete palette, black axis/spine styling).

---

## 5. Implementation Notes

- **Evaluator model.** In the reference run, the same model family serves as generator and grader (e.g., MiniMax-M2.7 via an Anthropic-compatible API). Stages use dedicated system prompts for review and agreement to enforce JSON-only outputs.  
- **Resumability.** Each stage appends to its cache; interrupted runs resume on pending trials.  
- **Parallelism.** Trials are processed concurrently (default: 3 workers) with independent API clients per task.  
- **Reproducibility.** Full rubric text is embedded in prompts; the agreement rubric is also materialized as `agreement_scoring_rule.md` for human inspection.

---

## 6. Interpretation

CriteriaBench separates two complementary views of model performance:

1. **Quality rubric** — Does the draft satisfy trial-design principles (enrichment, feasibility, safety) given the protocol context?  
2. **Expert agreement** — Does the draft recover the substance of criteria that experienced investigators actually registered?

A model can score well on quality while diverging from expert criteria (e.g., by adding reasonable but non-matching screens), or align partially while leaving gaps in safety or feasibility. Reporting both metrics is therefore necessary for a complete assessment of eligibility-criteria generation.

---

## 7. Limitations

- **LLM-as-judge variance.** Review and agreement stages inherit grader inconsistency and position bias; scores should be interpreted as structured rubric applications, not ground-truth clinical audits.  
- **Single primary outcome.** Only the first listed primary outcome is shown at generation time; multi-endpoint trials may be under-specified in the input.  
- **Registry expert as reference.** ClinicalTrials.gov eligibility text may be incomplete or outdated relative to the full protocol; agreement measures registry fidelity, not absolute clinical optimality.  
- **Same-model grading.** Using one model for generation and evaluation couples scores to that model’s grading behavior; cross-model or human adjudication would strengthen external validity.
