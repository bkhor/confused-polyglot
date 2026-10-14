#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Install with: pip install anthropic")

HERE = Path(__file__).parent
RUBRIC_PATH = HERE / "rubric.md"
RESPONSES_PATH = HERE.parent / "data" / "responses.jsonl"
CORPUS_PATH = HERE.parent / "data" / "corpus.jsonl"
JUDGED_PATH = HERE.parent / "data" / "judged.jsonl"
BATCH_STATE_PATH = HERE / "batch_state.json"

JUDGE_MODEL = "claude-haiku-5-5"
MAX_RETRIES = 3
POLL_INTERVAL_SECONDS = 30

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


def render_user_message(original_prompt: str, degraded_prompt: str, response: str) -> str:
    return (
        f"original_prompt:\n{original_prompt}\n\n"
        f"degraded_prompt:\n{degraded_prompt}\n\n"
        f"response:\n{response}"
    )


def build_request(rubric_text: str, user_message: str, call_type: str) -> dict:
    tool = OUTCOME_TOOL if call_type == "outcome" else COMPREHENSION_TOOL
    return dict(
        model=JUDGE_MODEL,
        max_tokens=1024,
        tools=[tool],
        system=[
            {
                "type": "text",
                "text": rubric_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user_message}],
    )


def load_responses(limit: int | None = None) -> list[dict]:
    rows = []
    with open(RESPONSES_PATH) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def corpus_row_id(row: dict) -> str:
    pool = row["pool_size"]
    pool_str = f"p{int(pool)}" if pool is not None else "pNA"
    return f"{row['item_id']}-{row['chain_type']}-{pool_str}-g{row['generation']}-s{row['degradation_seed']}"


def load_prompt_lookups() -> tuple[dict, dict]:
    original_by_item = {}
    degraded_by_row = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            degraded_by_row[corpus_row_id(row)] = row["text"]
            if row["chain_type"] == "gen0":
                original_by_item[row["item_id"]] = row["text"]
    return original_by_item, degraded_by_row


def load_existing_keys(path: Path) -> set:
    keys = set()
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((row["response_id"], row["call_type"]))
    return keys


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


def extract_tool_use(message) -> tuple:
    stop_reason = message.stop_reason
    for block in message.content:
        if block.type == "tool_use":
            return True, block.name, block.input, stop_reason
    return False, None, None, stop_reason


def parse_tool_result(response_id: str, call_type: str, message, error: str | None) -> dict:
    row = {
        "response_id": response_id,
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

    tool_invoked, tool_name_used, raw_input, stop_reason = extract_tool_use(message)
    usage = message.usage

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


def build_pending_requests(rows, rubric_text, original_by_item, degraded_by_row, existing_keys):
    pending = []
    for row in rows:
        original_prompt = original_by_item[row["item_id"]]
        degraded_prompt = degraded_by_row[row["corpus_row_id"]]
        user_message = render_user_message(original_prompt, degraded_prompt, row["response_text"])
        for call_type in ("outcome", "comprehension"):
            key = (row["response_id"], call_type)
            if key in existing_keys:
                continue
            request = build_request(rubric_text, user_message, call_type)
            pending.append((row["response_id"], call_type, request))
    return pending


def run_dry_run(rubric_text, original_by_item, degraded_by_row, rows):
    row = rows[0]
    original_prompt = original_by_item[row["item_id"]]
    degraded_prompt = degraded_by_row[row["corpus_row_id"]]
    user_message = render_user_message(original_prompt, degraded_prompt, row["response_text"])
    for call_type in ("outcome", "comprehension"):
        request = build_request(rubric_text, user_message, call_type)
        print(f"=== {call_type} call ===")
        print(f"system (cached, {len(rubric_text)} chars): {rubric_text[:200]}...")
        print(f"tool: {request['tools'][0]['name']}")
        print(f"user message:\n{request['messages'][0]['content']}")
        print()


def run_synchronous(client, pending, out_path: Path):
    with open(out_path, "a") as out:
        for i, (response_id, call_type, request) in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {response_id} ({call_type})")
            message, error = call_with_retry(client, request)
            row = parse_tool_result(response_id, call_type, message, error)
            out.write(json.dumps(row) + "\n")
            out.flush()


MAX_BATCH_BYTES = 150 * 1024 * 1024  # safety margin under the API's 256MB cap


def load_batch_state() -> tuple[list[str], dict]:
    if not BATCH_STATE_PATH.exists():
        return [], {}
    state = json.loads(BATCH_STATE_PATH.read_text())
    return state["batch_ids"], state["id_map"]


def save_batch_state(batch_ids: list[str], id_map: dict) -> None:
    BATCH_STATE_PATH.write_text(json.dumps({"batch_ids": batch_ids, "id_map": id_map}))


def clear_batch_state() -> None:
    if BATCH_STATE_PATH.exists():
        BATCH_STATE_PATH.unlink()


def make_custom_id(response_id: str, call_type: str) -> str:
    return hashlib.sha256(f"{response_id}|{call_type}".encode()).hexdigest()[:32]


def chunk_by_size(pending, max_bytes: int):
    chunk = []
    chunk_bytes = 0
    for response_id, call_type, request in pending:
        payload = {"custom_id": make_custom_id(response_id, call_type), "params": request}
        size = len(json.dumps(payload).encode("utf-8"))
        if chunk and chunk_bytes + size > max_bytes:
            yield chunk
            chunk = []
            chunk_bytes = 0
        chunk.append(payload)
        chunk_bytes += size
    if chunk:
        yield chunk


def submit_batches(client, pending) -> tuple[list[str], dict]:
    id_map = {make_custom_id(response_id, call_type): [response_id, call_type] for response_id, call_type, _ in pending}
    batch_ids = []
    chunks = list(chunk_by_size(pending, MAX_BATCH_BYTES))
    for i, chunk in enumerate(chunks, 1):
        batch = client.messages.batches.create(requests=chunk)
        print(f"Submitted batch {batch.id} ({i}/{len(chunks)}, {len(chunk)} requests)")
        batch_ids.append(batch.id)
        save_batch_state(batch_ids, id_map)
    return batch_ids, id_map


def wait_for_batch(client, batch_id: str):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"batch {batch_id}: {batch.processing_status} "
            f"(succeeded={counts.succeeded} errored={counts.errored} "
            f"processing={counts.processing} canceled={counts.canceled} expired={counts.expired})"
        )
        if batch.processing_status == "ended":
            return batch
        time.sleep(POLL_INTERVAL_SECONDS)


def write_batch_results(client, batch_id: str, id_map: dict, out_path: Path) -> None:
    with open(out_path, "a") as out:
        for item in client.messages.batches.results(batch_id):
            response_id, call_type = id_map[item.custom_id]
            if item.result.type == "succeeded":
                row = parse_tool_result(response_id, call_type, item.result.message, None)
            else:
                row = parse_tool_result(response_id, call_type, None, f"{item.result.type}: {getattr(item.result, 'error', '')}")
            out.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print rendered payloads for the first row, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N response rows (implies synchronous calls)")
    args = parser.parse_args()

    load_dotenv(HERE.parent / ".env")

    rubric_text = RUBRIC_PATH.read_text()
    rows = load_responses(limit=args.limit)
    original_by_item, degraded_by_row = load_prompt_lookups()

    if args.dry_run:
        run_dry_run(rubric_text, original_by_item, degraded_by_row, rows)
        return

    client = anthropic.Anthropic()
    JUDGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_keys(JUDGED_PATH)

    pending = build_pending_requests(rows, rubric_text, original_by_item, degraded_by_row, existing_keys)
    if not pending:
        print("Nothing to do, all requested rows already judged.")
        return

    if args.limit:
        print(f"{len(pending)} calls pending, running synchronously (--limit set)")
        run_synchronous(client, pending, JUDGED_PATH)
        print(f"\nDone. Results in {JUDGED_PATH}")
        return

    batch_ids, id_map = load_batch_state()
    if batch_ids:
        print(f"Resuming {len(batch_ids)} in-flight batch(es)")
    else:
        print(f"{len(pending)} calls pending, submitting as batches")
        batch_ids, id_map = submit_batches(client, pending)

    for batch_id in list(batch_ids):
        wait_for_batch(client, batch_id)
        write_batch_results(client, batch_id, id_map, JUDGED_PATH)
        batch_ids.remove(batch_id)
        save_batch_state(batch_ids, id_map)

    clear_batch_state()
    print(f"\nDone. Results in {JUDGED_PATH}")


if __name__ == "__main__":
    main()
