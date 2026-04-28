## Overview

This is an **LLM evaluation framework** that evaluates ExtensionBot against GPT & Gemini and using Claude as an automated judge:
- **Control Benchmark**: General knowledge / non-agricultural questions; the idea here is that when I run the evaluation I would expect to see ExtensionBot score lower on the Control Benchmark as compared to the IPM Benchmark.
- **IPM Benchmark**: Agricultural Integrated Pest Management domain-specific questions
  <img width="2898" height="1472" alt="image" src="https://github.com/user-attachments/assets/98734c1c-e68a-4757-ac4a-887639743493" />
## Environment Setup

Requires a `.env` file with:
```
EXTBOT_API_KEY="..." This is ExtensionBot specific
OPENROUTER_API_KEY="..."
```

Python dependencies (no requirements.txt — install manually):
```
pip install requests python-dotenv
```

## Common Commands

### Collect responses from AI models
```bash
python3 control/collect_control_responses.py
python3 ipm/collect_ipm_responses.py
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
### Evaluated Models

- **ExtBot** — via ExtensionBot API
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
