import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Install with: pip install anthropic")

from run_inference import MAX_TOKENS, TEMPERATURE, build_messages, now_iso, system_prompt_hash

INPUT_PATH = "data/inference_input.jsonl"
OUTPUT_PATH = "data/responses.jsonl"
BATCH_STATE_PATH = Path(__file__).parent / "frontier_batch_state.json"

MODEL_NAME = "claude-sonnet-5-5"
MAX_RETRIES = 3
POLL_INTERVAL_SECONDS = 30
MAX_BATCH_BYTES = 150 * 1024 * 1024


def is_frontier_scope(row: dict) -> bool:
    if row["chain_type"] == "gen0":
        return True
    if row["chain_type"] == "pwt" and row["generation"] == 1:
        return True
    return False


def build_response_row(row: dict, response_text: str, model_name: str) -> dict:
    return {
        "response_id": f"{row['corpus_row_id']}-{model_name}-0",
        "corpus_row_id": row["corpus_row_id"],
        "item_id": row["item_id"],
        "condition": row["condition"],
        "chain_type": row["chain_type"],
        "pool_size": row["pool_size"],
        "generation": row["generation"],
        "degradation_seed": row["degradation_seed"],
        "model": model_name,
        "param_count": None,
        "sample_idx": 0,
        "temperature": TEMPERATURE,
        "system_prompt_hash": system_prompt_hash(),
        "response_text": response_text,
        "mean_token_logprob": None,
        "created_at": now_iso(),
    }


def load_existing_corpus_row_ids(path: Path, model_name: str) -> set:
    ids = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["model"] == model_name:
                ids.add(row["corpus_row_id"])
    return ids


def load_input_rows() -> list[dict]:
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return [row for row in rows if is_frontier_scope(row)]


def load_batch_state() -> tuple[list[str], dict]:
    if not BATCH_STATE_PATH.exists():
        return [], {}
    state = json.loads(BATCH_STATE_PATH.read_text())
    return state["batch_ids"], state["id_map"]


def save_batch_state(batch_ids: list[str], id_map: dict) -> None:
    tmp_path = BATCH_STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps({"batch_ids": batch_ids, "id_map": id_map}))
    os.replace(tmp_path, BATCH_STATE_PATH)


def clear_batch_state() -> None:
    if BATCH_STATE_PATH.exists():
        BATCH_STATE_PATH.unlink()


def make_custom_id(corpus_row_id: str) -> str:
    return hashlib.sha256(corpus_row_id.encode()).hexdigest()[:32]


def build_request(row: dict) -> dict:
    return dict(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=build_messages(row["text"]),
    )


def extract_response_text(message) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def is_retryable(e: Exception) -> bool:
    if isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    if isinstance(e, anthropic.APIStatusError):
        return e.status_code >= 500
    return False


def call_with_retry(client, request: dict) -> tuple:
    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(**request), None
        except Exception as e:
            if not is_retryable(e) or attempt == MAX_RETRIES - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(2**attempt)
    return None, "exhausted retries"


def run_dry_run(row: dict) -> None:
    request = build_request(row)
    print(f"=== request for {row['corpus_row_id']} ===")
    print(f"model: {request['model']}, max_tokens: {request['max_tokens']}, temperature: {request['temperature']}")
    print(f"messages:\n{request['messages']}")


def run_synchronous(client, pending: list[dict], out_path: Path) -> None:
    with open(out_path, "a") as out:
        for i, row in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {row['corpus_row_id']}")
            message, error = call_with_retry(client, build_request(row))
            if error is not None:
                print(f"WARNING: {row['corpus_row_id']} failed ({error}), will retry on next run")
                continue
            response_text = extract_response_text(message)
            out_row = build_response_row(row, response_text, MODEL_NAME)
            out.write(json.dumps(out_row) + "\n")
            out.flush()


def chunk_by_size(pending, max_bytes: int):
    chunk = []
    chunk_bytes = 0
    for row in pending:
        payload = {"custom_id": make_custom_id(row["corpus_row_id"]), "params": build_request(row)}
        size = len(json.dumps(payload).encode("utf-8"))
        if chunk and chunk_bytes + size > max_bytes:
            yield chunk
            chunk = []
            chunk_bytes = 0
        chunk.append(payload)
        chunk_bytes += size
    if chunk:
        yield chunk


def submit_batches(client, pending: list[dict]) -> tuple[list[str], dict]:
    id_map = {make_custom_id(row["corpus_row_id"]): row for row in pending}
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


def write_frontier_batch_results(client, batch_id: str, id_map: dict, model_name: str, out_path: Path) -> None:
    existing = load_existing_corpus_row_ids(out_path, model_name)
    with open(out_path, "a") as out:
        for item in client.messages.batches.results(batch_id):
            row = id_map[item.custom_id]
            if row["corpus_row_id"] in existing:
                continue
            if item.result.type != "succeeded":
                print(f"WARNING: {row['corpus_row_id']} failed ({item.result.type}), will retry on next run")
                continue
            response_text = extract_response_text(item.result.message)
            out_row = build_response_row(row, response_text, model_name)
            out.write(json.dumps(out_row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print the first pending request, no API calls")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (implies synchronous calls)")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")

    all_rows = load_input_rows()
    existing = load_existing_corpus_row_ids(Path(OUTPUT_PATH), MODEL_NAME)
    pending = [row for row in all_rows if row["corpus_row_id"] not in existing]
    if args.limit is not None:
        pending = pending[: args.limit]

    if args.dry_run:
        if pending:
            run_dry_run(pending[0])
        else:
            print("Nothing pending to show.")
        return

    if not pending:
        print("Nothing to do, all frontier-scope rows already have responses.")
        return

    client = anthropic.Anthropic()

    if args.limit is not None:
        print(f"{len(pending)} rows pending, running synchronously (--limit set)")
        run_synchronous(client, pending, Path(OUTPUT_PATH))
        print(f"\nDone. Responses appended to {OUTPUT_PATH}")
        return

    batch_ids, id_map = load_batch_state()
    if batch_ids:
        print(f"Resuming {len(batch_ids)} in-flight batch(es)")
    else:
        print(f"{len(pending)} rows pending (of {len(all_rows)} in frontier scope), submitting as batches")
        batch_ids, id_map = submit_batches(client, pending)

    for batch_id in list(batch_ids):
        wait_for_batch(client, batch_id)
        write_frontier_batch_results(client, batch_id, id_map, MODEL_NAME, Path(OUTPUT_PATH))
        batch_ids.remove(batch_id)
        save_batch_state(batch_ids, id_map)

    clear_batch_state()
    print(f"\nDone. Responses appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
