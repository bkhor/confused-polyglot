import json
import math
from pathlib import Path

import pandas as pd


def corpus_row_id(row: dict) -> str:
    pool = row["pool_size"]
    pool_str = f"p{int(pool)}" if pool is not None else "pNA"
    return f"{row['item_id']}-{row['chain_type']}-{pool_str}-g{row['generation']}-s{row['degradation_seed']}"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_original_prompt_lookup(corpus_rows: list[dict]) -> dict[str, str]:
    lookup = {}
    for row in corpus_rows:
        if row["chain_type"] == "gen0":
            lookup[row["item_id"]] = row["text"]
    return lookup


def join_corpus_responses(corpus_rows: list[dict], response_rows: list[dict]) -> pd.DataFrame:
    corpus_by_row_id = {corpus_row_id(row): row for row in corpus_rows}
    original_by_item = build_original_prompt_lookup(corpus_rows)

    joined = []
    for response in response_rows:
        corpus_row = corpus_by_row_id[response["corpus_row_id"]]
        joined.append(
            {
                **response,
                "degraded_prompt": corpus_row["text"],
                "pool_composition": corpus_row["pool_composition"],
                "realized_language_count": corpus_row["realized_language_count"],
                "distinct_script_count": corpus_row["distinct_script_count"],
                "pass_through_rate": corpus_row["pass_through_rate"],
                "semantic_distance_from_gen0": corpus_row["semantic_distance_from_gen0"],
                "token_count": corpus_row["token_count"],
                "original_prompt": original_by_item[response["item_id"]],
            }
        )
    return pd.DataFrame(joined)


JUDGED_COLUMNS = [
    "response_id",
    "outcome_category",
    "outcome_justification",
    "gist_recovery",
    "instruction_recovery",
    "comprehension_justification",
    "judge_error",
]


def pivot_judged(judged_rows: list[dict]) -> pd.DataFrame:
    by_response = {}
    for row in judged_rows:
        record = by_response.setdefault(
            row["response_id"],
            {
                "response_id": row["response_id"],
                "outcome_category": None,
                "outcome_justification": None,
                "gist_recovery": None,
                "instruction_recovery": None,
                "comprehension_justification": None,
                "judge_error": None,
            },
        )
        if row["call_type"] == "outcome":
            if row["parsed"]:
                record["outcome_category"] = row["parsed"]["outcome_category"]
                record["outcome_justification"] = row["parsed"]["outcome_justification"]
            if row["error"] and not record["judge_error"]:
                record["judge_error"] = row["error"]
        else:
            if row["parsed"]:
                record["gist_recovery"] = row["parsed"]["gist_recovery"]
                record["instruction_recovery"] = row["parsed"]["instruction_recovery"]
                record["comprehension_justification"] = row["parsed"]["comprehension_justification"]
            if row["error"] and not record["judge_error"]:
                record["judge_error"] = row["error"]

    if not by_response:
        return pd.DataFrame(columns=JUDGED_COLUMNS)
    return pd.DataFrame(list(by_response.values()), columns=JUDGED_COLUMNS)


def build_case_table(corpus_rows: list[dict], response_rows: list[dict], judged_rows: list[dict]) -> pd.DataFrame:
    base = join_corpus_responses(corpus_rows, response_rows)
    judged = pivot_judged(judged_rows)
    merged = base.merge(judged, on="response_id", how="left")
    merged["perplexity"] = merged["mean_token_logprob"].apply(
        lambda lp: math.exp(-lp) if pd.notna(lp) else None
    )
    return merged


def load_case_table(corpus_path: Path, responses_path: Path, judged_path: Path) -> pd.DataFrame:
    corpus_rows = load_jsonl(corpus_path)
    response_rows = load_jsonl(responses_path)
    judged_rows = load_jsonl(judged_path) if judged_path.exists() else []
    return build_case_table(corpus_rows, response_rows, judged_rows)
