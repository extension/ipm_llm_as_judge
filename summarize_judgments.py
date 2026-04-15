#!/usr/bin/env python3

import json
import csv
import argparse
from collections import defaultdict

MODEL_ORDER = {
    "extbot": 0,
    "gemini": 1,
    "gpt": 2,
}


def safe_mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def round_or_none(value, digits=3):
    if value is None:
        return None
    return round(value, digits)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_judgments(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "judge" not in row:
                continue
            rows.append(row)
    return rows


def get_metric_scores(items, metric_name):
    values = []
    for x in items:
        judge = x.get("judge", {})
        metric = judge.get(metric_name, {})
        score = metric.get("score")
        values.append(score)
    return values


def get_overall_scores(items):
    values = []
    for x in items:
        judge = x.get("judge", {})
        overall = judge.get("overall_score")

        if overall is None:
            rel = judge.get("relevance", {}).get("score")
            comp = judge.get("completeness", {}).get("score")
            grd = judge.get("groundedness", {}).get("score")
            cit = judge.get("citation_quality", {}).get("score")
            overall = safe_mean([rel, comp, grd, cit])

        values.append(overall)
    return values


def aggregate_rows(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(group_key, "UNKNOWN")].append(row)

    output = []
    for key, items in groups.items():
        relevance = safe_mean(get_metric_scores(items, "relevance"))
        completeness = safe_mean(get_metric_scores(items, "completeness"))
        groundedness = safe_mean(get_metric_scores(items, "groundedness"))
        citation_quality = safe_mean(get_metric_scores(items, "citation_quality"))
        overall = safe_mean(get_overall_scores(items))

        output.append({
            group_key: key,
            "n": len(items),
            "relevance": round_or_none(relevance),
            "completeness": round_or_none(completeness),
            "groundedness": round_or_none(groundedness),
            "citation_quality": round_or_none(citation_quality),
            "overall_score": round_or_none(overall),
        })

    return sorted(output, key=lambda x: str(x[group_key]))


def aggregate_by_model_and_key(rows, group_key):
    groups = defaultdict(list)
    for row in rows:
        model = row.get("label", "UNKNOWN")
        key = row.get(group_key, "UNKNOWN")
        groups[(key, model)].append(row)

    output = []
    for (key, model), items in groups.items():
        relevance = safe_mean(get_metric_scores(items, "relevance"))
        completeness = safe_mean(get_metric_scores(items, "completeness"))
        groundedness = safe_mean(get_metric_scores(items, "groundedness"))
        citation_quality = safe_mean(get_metric_scores(items, "citation_quality"))
        overall = safe_mean(get_overall_scores(items))

        output.append({
            "model": model,
            group_key: key,
            "n": len(items),
            "relevance": round_or_none(relevance),
            "completeness": round_or_none(completeness),
            "groundedness": round_or_none(groundedness),
            "citation_quality": round_or_none(citation_quality),
            "overall_score": round_or_none(overall),
        })

    return sorted(
        output,
        key=lambda x: (
            str(x[group_key]),
            MODEL_ORDER.get(x["model"], 999),
            str(x["model"]),
        ),
    )


def aggregate_traps(rows):
    groups = defaultdict(list)
    for row in rows:
        label = row.get("label", "UNKNOWN")
        if parse_bool(row.get("trap_question")):
            groups[label].append(row)

    output = []
    for label, items in groups.items():
        relevance = safe_mean(get_metric_scores(items, "relevance"))
        completeness = safe_mean(get_metric_scores(items, "completeness"))
        groundedness = safe_mean(get_metric_scores(items, "groundedness"))
        citation_quality = safe_mean(get_metric_scores(items, "citation_quality"))
        overall = safe_mean(get_overall_scores(items))

        output.append({
            "label": label,
            "n_trap": len(items),
            "trap_relevance": round_or_none(relevance),
            "trap_completeness": round_or_none(completeness),
            "trap_groundedness": round_or_none(groundedness),
            "trap_citation_quality": round_or_none(citation_quality),
            "trap_overall_score": round_or_none(overall),
        })

    return sorted(output, key=lambda x: MODEL_ORDER.get(x["label"], 999))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    by_model = aggregate_rows(rows, "label")
    print("\nSummary by model:")
    for row in sorted(by_model, key=lambda x: MODEL_ORDER.get(x["label"], 999)):
        print(
            f"  {row['label']}: "
            f"n={row['n']}, "
            f"overall={row['overall_score']}, "
            f"relevance={row['relevance']}, "
            f"completeness={row['completeness']}, "
            f"groundedness={row['groundedness']}, "
            f"citation_quality={row['citation_quality']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate existing judgment scores from a judgments JSONL file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to eval_xxx_judgments.jsonl",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output CSVs",
    )
    args = parser.parse_args()

    rows = load_judgments(args.input)
    if not rows:
        raise SystemExit("No judgment rows found in input file.")

    prefix = args.output_prefix or args.input.rsplit(".", 1)[0]

    # Original rollups
    by_model = aggregate_rows(rows, "label")
    by_category = aggregate_rows(rows, "category")
    by_crop = aggregate_rows(rows, "crop")
    by_question_type = aggregate_rows(rows, "question_type")
    by_difficulty = aggregate_rows(rows, "difficulty")
    by_risk = aggregate_rows(rows, "risk_level")
    by_trap = aggregate_traps(rows)

    # Model-split rollups, sorted by type then model
    by_model_category = aggregate_by_model_and_key(rows, "category")
    by_model_crop = aggregate_by_model_and_key(rows, "crop")
    by_model_question_type = aggregate_by_model_and_key(rows, "question_type")
    by_model_difficulty = aggregate_by_model_and_key(rows, "difficulty")

    # Write original outputs
    write_csv(f"{prefix}_metrics_by_model.csv", by_model)
    write_csv(f"{prefix}_metrics_by_category.csv", by_category)
    write_csv(f"{prefix}_metrics_by_crop.csv", by_crop)
    write_csv(f"{prefix}_metrics_by_question_type.csv", by_question_type)
    write_csv(f"{prefix}_metrics_by_difficulty.csv", by_difficulty)
    write_csv(f"{prefix}_metrics_by_risk.csv", by_risk)
    write_csv(f"{prefix}_metrics_trap.csv", by_trap)

    # Write model-split outputs
    write_csv(f"{prefix}_metrics_by_model_category.csv", by_model_category)
    write_csv(f"{prefix}_metrics_by_model_crop.csv", by_model_crop)
    write_csv(f"{prefix}_metrics_by_model_question_type.csv", by_model_question_type)
    write_csv(f"{prefix}_metrics_by_model_difficulty.csv", by_model_difficulty)

    print_summary(rows)
    print("\nWrote:")
    print(f"  {prefix}_metrics_by_model.csv")
    print(f"  {prefix}_metrics_by_category.csv")
    print(f"  {prefix}_metrics_by_crop.csv")
    print(f"  {prefix}_metrics_by_question_type.csv")
    print(f"  {prefix}_metrics_by_difficulty.csv")
    print(f"  {prefix}_metrics_by_risk.csv")
    print(f"  {prefix}_metrics_trap.csv")
    print(f"  {prefix}_metrics_by_model_category.csv")
    print(f"  {prefix}_metrics_by_model_crop.csv")
    print(f"  {prefix}_metrics_by_model_question_type.csv")
    print(f"  {prefix}_metrics_by_model_difficulty.csv")


if __name__ == "__main__":
    main()