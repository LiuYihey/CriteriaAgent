# Vanilla RAG baseline

Comparison baseline for CriteriaBench: concatenates the four-domain `trial_profile` (disease, drug, papers, similar trials) into one prompt with the trial configuration. **Single-pass generation only** — no graph, no task planner, no expert subagents, no reviewer optimization loop.

This package is intentionally decoupled from `criteria_agent/`.

## Usage

```powershell
python baselines/vanilla_rag/run.py ^
  --config examples/minoxidil/trial_config.json ^
  --profile examples/minoxidil/trial_profile.json ^
  --output baselines/vanilla_rag/outputs/smoke_minoxidil
```

Output: `criteria.md`
