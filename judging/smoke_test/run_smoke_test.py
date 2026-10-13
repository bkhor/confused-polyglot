#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Install with: pip install anthropic")

HERE = Path(__file__).parent
RUBRIC_PATH = HERE.parent / "rubric.md"
ITEMS_PATH = HERE / "items.yaml"
RESULTS_PATH = HERE / "results" / "raw_results.jsonl"

MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
DEFAULT_RUNS = 3
MAX_RETRIES = 3

OUTCOME_TOOL = {
    "name": "classify_outcome",
    "description": "Classify how the test model's response handled the degraded prompt.",
    "input_schema": {
        "type": "object",
        "properties": {
            "outcome_category": {
                "type": "string",
                "enum": [
                    "comprehended_complied",
                    "confabulated",
                    "honest_confusion",
                    "safety_refusal",
                    "degenerate",
                ],
            },
            "outcome_justification": {"type": "string"},
        },
        "required": ["outcome_category", "outcome_justification"],
    },
}

COMPREHENSION_TOOL = {
    "name": "score_comprehension",
    "description": "Score gist and instruction recovery for the response, independent of its outcome category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "gist_recovery": {"type": ["number", "null"]},
            "instruction_recovery": {"type": ["number", "null"]},
            "comprehension_justification": {"type": "string"},
        },
        "required": ["gist_recovery", "instruction_recovery", "comprehension_justification"],
    },
}


def render_user_message(item: dict) -> str:
    return (
        f"original_prompt:\n{item['original_prompt']}\n\n"
        f"degraded_prompt:\n{item['degraded_prompt']}\n\n"
        f"response:\n{item['response']}"
    )


def build_request(model: str, rubric_text: str, item: dict, call_type: str) -> dict:
    tool = OUTCOME_TOOL if call_type == "outcome" else COMPREHENSION_TOOL
    return dict(
        model=model,
        max_tokens=512,
        tools=[tool],
        system=[
            {
                "type": "text",
                "text": rubric_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": render_user_message(item)}],
    )


def is_retryable(e: Exception) -> bool:
    if isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    if isinstance(e, anthropic.APIStatusError):
        return e.status_code >= 500
    return False


def call_with_retry(client: "anthropic.Anthropic", request: dict) -> tuple:
    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(**request), None
        except Exception as e:
            if not is_retryable(e) or attempt == MAX_RETRIES - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(2**attempt)
    return None, "exhausted retries"


def extract_tool_use(response) -> tuple:
    stop_reason = response.stop_reason
    for block in response.content:
        if block.type == "tool_use":
            return True, block.name, block.input, stop_reason
    return False, None, None, stop_reason


def load_existing_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["item_id"], row["model"], row["run_idx"], row["call_type"]))
    return keys


def run_one_call(client, model, rubric_text, item, call_type, run_idx):
    request = build_request(model, rubric_text, item, call_type)
    response, error = call_with_retry(client, request)

    row = {
        "item_id": item["id"],
        "model": model,
        "run_idx": run_idx,
        "call_type": call_type,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if error is not None:
        row.update(
            tool_invoked=False,
            tool_name_used=None,
            correct_tool=False,
            stop_reason=None,
            parsed=None,
            raw_tool_input=None,
            error=error,
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_creation_tokens=None,
        )
        return row

    tool_invoked, tool_name_used, raw_input, stop_reason = extract_tool_use(response)
    usage = response.usage

    expected_tool_name = OUTCOME_TOOL["name"] if call_type == "outcome" else COMPREHENSION_TOOL["name"]
    correct_tool = tool_invoked and tool_name_used == expected_tool_name

    parsed = None
    if correct_tool:
        if call_type == "outcome":
            parsed = {
                "outcome_category": raw_input.get("outcome_category"),
                "outcome_justification": raw_input.get("outcome_justification"),
            }
        else:
            parsed = {
                "gist_recovery": raw_input.get("gist_recovery"),
                "instruction_recovery": raw_input.get("instruction_recovery"),
                "comprehension_justification": raw_input.get("comprehension_justification"),
            }

    row.update(
        tool_invoked=tool_invoked,
        tool_name_used=tool_name_used,
        correct_tool=correct_tool,
        stop_reason=stop_reason,
        parsed=parsed,
        raw_tool_input=raw_input,
        error=None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None),
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None),
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print rendered payloads for item 1, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N items")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model override")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Self-consistency runs per item/model")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    rubric_text = RUBRIC_PATH.read_text()
    with open(ITEMS_PATH) as f:
        items = yaml.safe_load(f)

    if args.limit:
        items = items[: args.limit]

    models = args.models.split(",") if args.models else MODELS

    if args.dry_run:
        item = items[0]
        for call_type in ("outcome", "comprehension"):
            request = build_request(models[0], rubric_text, item, call_type)
            print(f"=== {call_type} call, model={models[0]} ===")
            print(f"system (cached, {len(rubric_text)} chars): {rubric_text[:200]}...")
            print(f"tool: {request['tools'][0]['name']}")
            print(f"user message:\n{request['messages'][0]['content']}")
            print()
        return

    client = anthropic.Anthropic()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_keys(RESULTS_PATH)

    total = len(items) * len(models) * args.runs * 2
    done = 0

    with open(RESULTS_PATH, "a") as out:
        for item in items:
            for model in models:
                for run_idx in range(args.runs):
                    for call_type in ("outcome", "comprehension"):
                        key = (item["id"], model, run_idx, call_type)
                        done += 1
                        if key in existing:
                            print(f"[{done}/{total}] skip (already have) {key}")
                            continue
                        print(f"[{done}/{total}] {key}")
                        row = run_one_call(client, model, rubric_text, item, call_type, run_idx)
                        out.write(json.dumps(row) + "\n")
                        out.flush()

    print(f"\nDone. Results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
