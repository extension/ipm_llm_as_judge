# python3 collect_ipm_responses.py --limit 3

import os
import csv
import json
import time
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv

import requests

load_dotenv("/opt/merlin_crawlers/llm_as_judge/.env")

# -----------------------------
# Config
# -----------------------------
DATASET_PATH = "ipm_questions.csv"
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
THIA_API_KEY = os.environ["THIA_API_KEY"]
EXTBOT_URL = os.environ.get(
    "EXTBOT_URL",
    "https://api.extensionbot.thiaplatform.ai/api/chat",
)

MODELS = {
    "gpt": "openai/gpt-5",
    "gemini": "google/gemini-2.5-pro",
}

EXTBOT_MODEL = os.environ.get("EXTBOT_MODEL", "openai/gpt-5.2")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an agricultural assistant answering integrated pest management "
    "questions for a U.S. agricultural audience. Be concise but complete. "
    "Do not browse the web. Do not invent citations. If uncertain, say so."
)

HTTP_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "")
X_TITLE = os.environ.get("OPENROUTER_X_TITLE", "IPM Eval Collector")

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


def normalize_extbot_reference(ref: dict) -> dict:
    return {
        "id": ref.get("id"),
        "url": ref.get("url"),
        "title": ref.get("title"),
        "domain": ref.get("domain"),
        "state": ref.get("state"),
        "score": ref.get("score"),
        "text": ref.get("text"),
        "chunk": ref.get("chunk"),
        "query": ref.get("query"),
        "created_at": ref.get("created_at"),
    }


# -----------------------------
# ExtBot
# -----------------------------
def ask_extbot(question: str) -> dict:
    start = time.time()

    payload = {
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "model": EXTBOT_MODEL,
        "config": {},
    }

    resp = request_with_retries(
        "POST",
        EXTBOT_URL,
        headers={
            "accept": "application/json",
            "Authorization": f"Bearer {EXTBOT_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    latency_ms = int((time.time() - start) * 1000)
    data = resp.json()

    response_text = data.get("content", "") or ""
    references = data.get("references", []) or []

    normalized_references = [
        normalize_extbot_reference(ref) for ref in references
    ]

    return {
        "response_text": response_text,
        "citations": normalized_references,
        "latency_ms": latency_ms,
        "raw": data,
        "extbot_message_id": data.get("message_id"),
        "extbot_conversation_id": data.get("conversation_id"),
        "extbot_returned_model": data.get("model"),
    }


# -----------------------------
# OpenRouter
# -----------------------------
def ask_openrouter(model_name: str, question: str) -> dict:
    start = time.time()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if HTTP_REFERER:
        headers["HTTP-Referer"] = HTTP_REFERER
    if X_TITLE:
        headers["X-Title"] = X_TITLE

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }

    resp = request_with_retries(
        "POST",
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    latency_ms = int((time.time() - start) * 1000)
    data = resp.json()

    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"No choices returned for model {model_name}")

    message = choices[0].get("message", {})
    response_text = message.get("content", "") or ""

    return {
        "response_text": response_text,
        "citations": [],
        "latency_ms": latency_ms,
        "raw": data,
    }


def collect_one(provider: str, model_name: str, question: str) -> dict:
    if provider == "extbot":
        return ask_extbot(question)
    if provider == "openrouter":
        return ask_openrouter(model_name, question)
    raise ValueError(f"Unknown provider: {provider}")


# -----------------------------
# Dataset selection
# -----------------------------
def load_rows(dataset_path: str) -> list[dict]:
    with open(dataset_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_rows(rows: list[dict], offset: int = 0, limit: int | None = None, question_id: str | None = None) -> list[dict]:
    if question_id is not None:
        selected = [r for r in rows if str(r["id"]) == str(question_id)]
        return selected

    selected = rows[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect benchmark responses from ExtBot and OpenRouter models.")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to benchmark CSV")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of questions to process")
    parser.add_argument("--offset", type=int, default=0, help="Number of rows to skip before processing")
    parser.add_argument("--question-id", default=None, help="Run only a single question by id")
    parser.add_argument("--output", default=None, help="Optional output JSONL path")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or f"responses_{run_id}.jsonl"

    targets = [
        {
            "provider": "extbot",
            "label": "extbot",
            "model_name": EXTBOT_MODEL,
        },
        {
            "provider": "openrouter",
            "label": "gpt",
            "model_name": MODELS["gpt"],
        },
        {
            "provider": "openrouter",
            "label": "gemini",
            "model_name": MODELS["gemini"],
        },
    ]

    all_rows = load_rows(args.dataset)
    rows = select_rows(
        all_rows,
        offset=args.offset,
        limit=args.limit,
        question_id=args.question_id,
    )

    if not rows:
        raise SystemExit("No questions matched the selection.")

    print(f"Loaded {len(all_rows)} total questions")
    print(f"Processing {len(rows)} questions")
    print(f"Writing to {output_path}")

    with open(output_path, "w", encoding="utf-8") as out:
        for row in rows:
            question = row["question"]

            for target in targets:
                record = {
                    "run_id": run_id,
                    "timestamp": utc_now(),
                    "question_id": row["id"],
                    "category": row["category"],
                    "crop": row["crop"],
                    "pest_type": row["pest_type"],
                    "question_type": row["question_type"],
                    "difficulty": row["difficulty"],
                    "risk_level": row["risk_level"],
                    "trap_question": row["trap_question"],
                    "reference_points": row["reference_points"],
                    "provider": target["provider"],
                    "label": target["label"],
                    "model_name": target["model_name"],
                    "prompt": question,
                    "error": None,
                }

                try:
                    result = collect_one(
                        provider=target["provider"],
                        model_name=target["model_name"],
                        question=question,
                    )
                    record.update(result)
                    record["citations_present"] = bool(result.get("citations"))
                except Exception as e:
                    record["response_text"] = ""
                    record["citations"] = []
                    record["citations_present"] = False
                    record["latency_ms"] = None
                    record["raw"] = None
                    record["error"] = str(e)

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    f"[q{record['question_id']}] {record['label']} "
                    f"{'OK' if not record['error'] else 'ERR'}"
                )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()