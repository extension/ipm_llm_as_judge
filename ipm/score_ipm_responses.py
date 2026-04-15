import os
import json
import csv
import time
import math
import argparse
from datetime import datetime, timezone
from collections import defaultdict

from dotenv import load_dotenv

import requests

load_dotenv("/opt/merlin_crawlers/llm_as_judge/.env")

# python3 score_ipm_responses.py --input responses_20260312T202848Z.jsonl
# python3 score_ipm_responses.py --input responses_20260312T202848Z.jsonl --limit 3
# python3 score_ipm_responses.py --input responses_20260312T202848Z.jsonl --label extbot
# python3 score_ipm_responses.py --input responses_20260312T202848Z.jsonl --output-prefix ipm_baseline_v1

# -----------------------------
# Config
# -----------------------------
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "anthropic/claude-sonnet-4.6"
# SECOND_JUDGE_MODEL = "openai/gpt-5.4"

HTTP_REFERER = ""
X_TITLE = "IPM Eval Scorer"

MAX_RETRIES = 3
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


# -----------------------------
# Helpers
# -----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_with_retries(method: str, url: str, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** (attempt - 1))
                continue
            raise last_exc


def safe_mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def strip_reasoning_encrypted(obj):
    if isinstance(obj, dict):
        if obj.get("type") == "reasoning.encrypted":
            return None
        cleaned = {}
        for key, value in obj.items():
            cleaned_value = strip_reasoning_encrypted(value)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned
    if isinstance(obj, list):
        cleaned = []
        for item in obj:
            cleaned_item = strip_reasoning_encrypted(item)
            if cleaned_item is not None:
                cleaned.append(cleaned_item)
        return cleaned
    return obj


# -----------------------------
# Judge prompt
# -----------------------------
def build_judge_prompt(record: dict) -> str:
    citations_blob = json.dumps(record.get("citations", []), ensure_ascii=False, indent=2)

    return f"""
You are evaluating the quality of an agricultural AI assistant response.

Your task is NOT to answer the question yourself.
Your task is to score the assistant's response only.

Use only the information provided below:
- the user question
- the assistant response
- the reference points
- the citations, if any

Do not use outside knowledge.
Do not add facts.
Do not reward style over substance.

Question:
{record.get("prompt", "")}

Reference Points:
{record.get("reference_points", "")}

Assistant Response:
{record.get("response_text", "")}

Citations:
{citations_blob}

Scoring Criteria:

1. Relevance
Did the response directly answer the user's question?
Score 1-5

2. Completeness
Did the response cover the main decision points or diagnostic points implied by the reference points?
Score 1-5

3. Groundedness
Does the response appear cautious, supported, and consistent with the provided reference points and citations, rather than making unsupported claims?
Score 1-5

4. Citation Quality
If citations are present:
Are they relevant, appropriate, and clearly supportive of the response?
Score 1-5

If no citations are present:
Set citation_quality.score to null and explain briefly.

Return ONLY valid JSON in this exact shape:

{{
  "relevance": {{
    "score": 1,
    "reasoning": "text"
  }},
  "completeness": {{
    "score": 1,
    "reasoning": "text"
  }},
  "groundedness": {{
    "score": 1,
    "reasoning": "text"
  }},
  "citation_quality": {{
    "score": null,
    "reasoning": "text"
  }},
  "strengths": "text",
  "weaknesses": "text"
}}

Rules:
- All scores must be integers 1-5, except citation_quality.score may be null
- Keep reasoning concise
- Return JSON only
""".strip()


# -----------------------------
# Judge call
# -----------------------------
def judge_record(record: dict) -> dict:
    prompt = build_judge_prompt(record)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if HTTP_REFERER:
        headers["HTTP-Referer"] = HTTP_REFERER
    if X_TITLE:
        headers["X-Title"] = X_TITLE

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful evaluation model. Return JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0
    }

    resp = request_with_retries(
        "POST",
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("Judge returned no choices")

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError("Judge returned empty content")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # small cleanup fallback
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())

    # normalize
    rel = parsed["relevance"]["score"]
    comp = parsed["completeness"]["score"]
    grd = parsed["groundedness"]["score"]
    cit = parsed["citation_quality"]["score"]

    overall = safe_mean([rel, comp, grd, cit])

    parsed["overall_score"] = overall
    return parsed


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_rows(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    out = []
    for key, items in groups.items():
        out.append({
            group_key: key,
            "n": len(items),
            "relevance": round(safe_mean([x["judge"]["relevance"]["score"] for x in items]), 3),
            "completeness": round(safe_mean([x["judge"]["completeness"]["score"] for x in items]), 3),
            "groundedness": round(safe_mean([x["judge"]["groundedness"]["score"] for x in items]), 3),
            "citation_quality": round(safe_mean([x["judge"]["citation_quality"]["score"] for x in items]), 3) if safe_mean([x["judge"]["citation_quality"]["score"] for x in items]) is not None else None,
            "overall_score": round(safe_mean([x["judge"]["overall_score"] for x in items]), 3),
        })
    return sorted(out, key=lambda x: str(x[group_key]))


def aggregate_traps(rows):
    groups = defaultdict(list)
    for row in rows:
        key = row["label"]
        groups[key].append(row)

    out = []
    for label, items in groups.items():
        trap_items = [x for x in items if parse_bool(x.get("trap_question"))]
        if not trap_items:
            continue
        out.append({
            "label": label,
            "n_trap": len(trap_items),
            "trap_relevance": round(safe_mean([x["judge"]["relevance"]["score"] for x in trap_items]), 3),
            "trap_completeness": round(safe_mean([x["judge"]["completeness"]["score"] for x in trap_items]), 3),
            "trap_groundedness": round(safe_mean([x["judge"]["groundedness"]["score"] for x in trap_items]), 3),
            "trap_citation_quality": round(safe_mean([x["judge"]["citation_quality"]["score"] for x in trap_items]), 3) if safe_mean([x["judge"]["citation_quality"]["score"] for x in trap_items]) is not None else None,
            "trap_overall_score": round(safe_mean([x["judge"]["overall_score"] for x in trap_items]), 3),
        })
    return sorted(out, key=lambda x: x["label"])


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Score collected benchmark responses with a judge model.")
    parser.add_argument("--input", required=True, help="Path to responses JSONL")
    parser.add_argument("--output-prefix", default=None, help="Optional output prefix")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    parser.add_argument("--label", default=None, help="Only score one model label, e.g. extbot, gpt, gemini")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.output_prefix or f"eval_{run_id}"

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue
            if args.label and row.get("label") != args.label:
                continue
            records.append(row)

    if args.limit is not None:
        records = records[:args.limit]

    if not records:
        raise SystemExit("No records to score.")

    judgments = []
    out_jsonl = f"{prefix}_judgments.jsonl"

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for i, record in enumerate(records, start=1):
            try:
                judge = judge_record(record)
                sanitized_record = strip_reasoning_encrypted(record)
                row = {
                    **sanitized_record,
                    "judge": judge,
                    "judged_at": utc_now(),
                    "judge_model": JUDGE_MODEL,
                }
                judgments.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{i}/{len(records)}] q{record['question_id']} {record['label']} OK")
            except Exception as e:
                print(f"[{i}/{len(records)}] q{record['question_id']} {record['label']} ERR: {e}")

    if not judgments:
        raise SystemExit("No judgments were successfully created.")

    by_model = aggregate_rows(judgments, "label")
    by_category = aggregate_rows(judgments, "category")
    by_crop = aggregate_rows(judgments, "crop")
    by_trap = aggregate_traps(judgments)

    write_csv(f"{prefix}_metrics_by_model.csv", by_model)
    write_csv(f"{prefix}_metrics_by_category.csv", by_category)
    write_csv(f"{prefix}_metrics_by_crop.csv", by_crop)
    write_csv(f"{prefix}_metrics_trap.csv", by_trap)

    print(f"Wrote {out_jsonl}")
    print(f"Wrote {prefix}_metrics_by_model.csv")
    print(f"Wrote {prefix}_metrics_by_category.csv")
    print(f"Wrote {prefix}_metrics_by_crop.csv")
    print(f"Wrote {prefix}_metrics_trap.csv")


if __name__ == "__main__":
    main()
