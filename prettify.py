#!/usr/bin/env python3

import json
import re
import sys
from pathlib import Path


def remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ]."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def strip_bom(text: str) -> str:
    """Remove UTF-8 BOM if present."""
    return text.lstrip("\ufeff")


def sanitize_control_chars(text: str) -> str:
    """
    Remove ASCII control characters that commonly break JSON parsing,
    while preserving tab, newline, and carriage return.
    """
    return "".join(
        ch for ch in text
        if ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) >= 32
    )


def try_parse_standard_json(text: str):
    return json.loads(text)


def try_parse_jsonl(text: str):
    """
    Parse JSONL / NDJSON:
    one JSON object per line.
    Returns a list of parsed records.
    """
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            cleaned_line = remove_trailing_commas(sanitize_control_chars(stripped))
            records.append(json.loads(cleaned_line))
    return records


def try_parse_multiple_json_values(text: str):
    """
    Parse multiple top-level JSON values, e.g.
    {"a":1}{"b":2}
    or
    {"a":1}
    {"b":2}
    """
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    values = []

    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1

        if idx >= length:
            break

        obj, end = decoder.raw_decode(text, idx)
        values.append(obj)
        idx = end

    return values


def parse_json_flexibly(raw: str):
    raw = strip_bom(raw)

    # 1) Standard JSON
    try:
        data = try_parse_standard_json(raw)
        print("✅ Parsed as standard JSON")
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️ Standard JSON parse failed: {e}")

    # 2) Standard JSON after light cleanup
    try:
        cleaned = remove_trailing_commas(sanitize_control_chars(raw))
        data = try_parse_standard_json(cleaned)
        print("✅ Parsed as standard JSON after cleanup")
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️ Cleaned standard JSON parse failed: {e}")

    # 3) JSONL / NDJSON
    try:
        data = try_parse_jsonl(raw)
        print("✅ Parsed as JSONL / NDJSON")
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️ JSONL parse failed: {e}")

    # 4) Multiple concatenated JSON values
    try:
        data = try_parse_multiple_json_values(raw)
        print("✅ Parsed as multiple JSON values")
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️ Multiple-value parse failed: {e}")

    # 5) Multiple values after cleanup
    cleaned = remove_trailing_commas(sanitize_control_chars(raw))
    data = try_parse_multiple_json_values(cleaned)
    print("✅ Parsed as multiple JSON values after cleanup")
    return data


def convert_to_valid_json(input_file: str, output_file: str):
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"❌ File not found: {input_file}")
        sys.exit(1)

    raw = input_path.read_text(encoding="utf-8", errors="replace")

    try:
        data = parse_json_flexibly(raw)
    except json.JSONDecodeError as e:
        print(f"❌ Could not parse file as JSON: {e}")
        sys.exit(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Pretty JSON written to: {output_file}")

    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            print(f"📊 Records in data array: {len(data['data'])}")
        else:
            print("📊 Top-level type: object")
    elif isinstance(data, list):
        print(f"📊 Top-level list length: {len(data)}")
    else:
        print(f"📊 Top-level type: {type(data).__name__}")


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python3 json_converter.py <input_file> <output_file>")
        print("")
        print("Examples:")
        print("  python3 json_converter.py responses_3qs.jsonl responses_3qs_pretty.json")
        print("  python3 json_converter.py raw.txt cleaned.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_to_valid_json(input_file, output_file)


if __name__ == "__main__":
    main()