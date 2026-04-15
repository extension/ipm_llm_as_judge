# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an **LLM evaluation framework** that benchmarks AI assistants across two domains using Claude as an automated judge:
- **Control Benchmark**: General knowledge / non-agricultural questions
- **IPM Benchmark**: Agricultural Integrated Pest Management domain-specific questions

## Environment Setup

Requires a `.env` file with:
```
THIA_API_KEY="..."
OPENROUTER_API_KEY="..."
```

Python dependencies (no requirements.txt — install manually):
```
pip install requests python-dotenv
```

## Common Commands

### Collect responses from AI models
```bash
python3 control/collect_control_responses.py --limit 3
python3 ipm/collect_ipm_responses.py --limit 3
```

### Score/judge responses using Claude
```bash
python3 control/score_control_responses.py --input control/control_responses_<timestamp>.jsonl --output-prefix control/control_baseline_v1
python3 ipm/score_ipm_responses.py --input responses_<timestamp>.jsonl --output-prefix ipm_baseline_v1
```

### Re-aggregate existing judgments
```bash
python3 summarize_judgments.py --input eval_<timestamp>_judgments.jsonl --output-prefix results
```

### Prettify malformed JSONL
```bash
python3 prettify.py input.jsonl output_pretty.json
```

## Architecture

### Three-Stage Pipeline

```
Question CSV → collect_*.py → response JSONL → score_*.py → judgment JSONL → summarize_judgments.py → metric CSVs
```

### Key Files by Role

| File | Role |
|------|------|
| `control/control_question_set.csv` | 29 non-agricultural test questions |
| `ipm/ipm_questions.csv` | Agricultural pest management questions |
| `control/collect_control_responses.py` | Queries 3 AI models with control questions |
| `ipm/collect_ipm_responses.py` | Queries 3 AI models with IPM questions |
| `control/score_control_responses.py` | Claude judges control responses (4 metrics) |
| `ipm/score_ipm_responses.py` | Claude judges IPM responses (4 metrics, uses `reference_points`) |
| `summarize_judgments.py` | Re-aggregates judgments into metric CSVs |

### Evaluated Models

- **ExtBot** — via THIA platform API (`https://api.extensionbot.thiaplatform.ai/api/chat`)
- **GPT-5** — via OpenRouter
- **Gemini 2.5 Pro** — via OpenRouter

**Judge model**: `anthropic/claude-sonnet-4-6` via OpenRouter

### Evaluation Metrics (1–5 scale)

Each response is scored on: **relevance**, **completeness**, **groundedness**, **citation_quality**, plus an **overall_score** (mean of the four).

### Output Files Per Run

- `*_judgments.jsonl` — Full judgment records with scores
- `*_metrics_by_model.csv` — Average scores per model
- `*_metrics_by_category.csv` — Scores by question category
- `*_metrics_by_difficulty.csv` / `*_metrics_by_crop.csv` — Domain-specific breakdowns
- `*_metrics_trap.csv` — Performance on trap questions

### IPM vs. Control Differences

The IPM scorer passes `reference_points` (expected correct answer information) into the judge prompt, making it a reference-based evaluation. The control scorer does not use reference points. IPM aggregations include breakdowns by `crop`, `question_type`, and `risk_level`.

### Retry Logic

All API calls use `request_with_retries()` with exponential backoff on HTTP 429/500/502/503/504 errors.
