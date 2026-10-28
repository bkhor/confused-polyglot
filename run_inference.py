import argparse
import hashlib
import json
import time
from pathlib import Path

INPUT_PATH = "data/inference_input.jsonl"
OUTPUT_PATH = "data/responses.jsonl"

M_SAMPLES = 5
TEMPERATURE = 0.7
MAX_TOKENS = 3584
MAX_MODEL_LEN = 4096
BATCH_SIZE = 100
SYSTEM_PROMPT = None

MODELS = [
    {
        "name": "qwen3-8b",
        "repo_id": "Qwen/Qwen3-8B",
        "param_count": 8_200_000_000,
        "chat_kwargs": {"enable_thinking": False},
    },
    {
        "name": "llama-3.1-8b-instruct",
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "param_count": 8_000_000_000,
        "chat_kwargs": {},
    },
    {
        "name": "gemma-4-12b-it",
        "repo_id": "google/gemma-4-12B-it",
        "param_count": 12_000_000_000,
        "chat_kwargs": {},
    },
    {
        "name": "phi-4",
        "repo_id": "microsoft/phi-4",
        "param_count": 14_000_000_000,
        "chat_kwargs": {},
    },
]


def system_prompt_hash() -> str:
    text = SYSTEM_PROMPT or ""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def build_messages(prompt_text: str) -> list[dict]:
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt_text})
    return messages


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_input_rows(limit: int | None = None) -> list[dict]:
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def load_existing_counts(path: Path, model_name: str) -> dict:
    counts = {}
    if not path.exists():
        return counts
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["model"] != model_name:
                continue
            counts[row["corpus_row_id"]] = counts.get(row["corpus_row_id"], 0) + 1
    return counts


def mean_logprob_for_completion(completion) -> float | None:
    if not completion.logprobs:
        return None
    token_logprobs = []
    for token_id, logprob_dict in zip(completion.token_ids, completion.logprobs):
        if token_id in logprob_dict:
            token_logprobs.append(logprob_dict[token_id].logprob)
    if not token_logprobs:
        return None
    return sum(token_logprobs) / len(token_logprobs)


def rows_with_sample_plan(rows: list[dict], existing: dict, m_samples: int) -> list[tuple[dict, int, int]]:
    plan = []
    for row in rows:
        have = existing.get(row["corpus_row_id"], 0)
        needed = m_samples - have
        if needed > 0:
            plan.append((row, needed, have))
    return plan


def run_model(model_cfg: dict, rows: list[dict], out_path: Path) -> None:
    from vllm import LLM, SamplingParams

    existing = load_existing_counts(out_path, model_cfg["name"])
    plan = rows_with_sample_plan(rows, existing, M_SAMPLES)

    if not plan:
        print(f"{model_cfg['name']}: nothing to do, all {len(rows)} rows already complete")
        return

    print(f"{model_cfg['name']}: loading model ({model_cfg['repo_id']})...")
    llm = LLM(
        model=model_cfg["repo_id"],
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
    )

    with open(out_path, "a") as out:
        for i in range(0, len(plan), BATCH_SIZE):
            batch = plan[i : i + BATCH_SIZE]
            messages_list = [build_messages(row["text"]) for row, _, _ in batch]
            sampling_params_list = [
                SamplingParams(n=needed, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, logprobs=1)
                for _, needed, _ in batch
            ]
            outputs = llm.chat(
                messages_list,
                sampling_params_list,
                chat_template_kwargs=model_cfg["chat_kwargs"] or None,
            )
            for (row, needed, start_idx), output in zip(batch, outputs):
                for offset, completion in enumerate(output.outputs):
                    sample_idx = start_idx + offset
                    resp_row = {
                        "response_id": f"{row['corpus_row_id']}-{model_cfg['name']}-{sample_idx}",
                        "corpus_row_id": row["corpus_row_id"],
                        "item_id": row["item_id"],
                        "condition": row["condition"],
                        "chain_type": row["chain_type"],
                        "pool_size": row["pool_size"],
                        "generation": row["generation"],
                        "degradation_seed": row["degradation_seed"],
                        "model": model_cfg["name"],
                        "param_count": model_cfg["param_count"],
                        "sample_idx": sample_idx,
                        "temperature": TEMPERATURE,
                        "system_prompt_hash": system_prompt_hash(),
                        "response_text": completion.text,
                        "mean_token_logprob": mean_logprob_for_completion(completion),
                        "created_at": now_iso(),
                    }
                    out.write(json.dumps(resp_row) + "\n")
            out.flush()
            done = min(i + BATCH_SIZE, len(plan))
            print(f"{model_cfg['name']}: batch done ({done}/{len(plan)} rows)")

    del llm
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N corpus rows")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model name override")
    args = parser.parse_args()

    rows = load_input_rows(limit=args.limit)
    models = MODELS
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in MODELS if m["name"] in wanted]

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for model_cfg in models:
        run_model(model_cfg, rows, out_path)

    print(f"\nDone. Responses in {out_path}")


if __name__ == "__main__":
    main()
