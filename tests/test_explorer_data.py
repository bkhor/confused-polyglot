import json
import math

import pandas as pd

from explorer.data import (
    build_case_table,
    build_original_prompt_lookup,
    corpus_row_id,
    join_corpus_responses,
    load_case_table,
    load_jsonl,
    pivot_judged,
)


def _corpus_row(item_id, chain_type, pool_size, generation, seed, text, **extra):
    base = {
        "item_id": item_id,
        "chain_type": chain_type,
        "pool_size": pool_size,
        "generation": generation,
        "degradation_seed": seed,
        "text": text,
        "pool_composition": None,
        "realized_language_count": 1,
        "distinct_script_count": 1,
        "pass_through_rate": 1.0,
        "semantic_distance_from_gen0": 0.0,
        "token_count": 5,
    }
    base.update(extra)
    return base


def _response_row(response_id, corpus_row_id_value, item_id, chain_type, pool_size, generation, seed, model, sample_idx, mean_token_logprob):
    return {
        "response_id": response_id,
        "corpus_row_id": corpus_row_id_value,
        "item_id": item_id,
        "condition": "meta",
        "chain_type": chain_type,
        "pool_size": pool_size,
        "generation": generation,
        "degradation_seed": seed,
        "model": model,
        "sample_idx": sample_idx,
        "response_text": "A response.",
        "mean_token_logprob": mean_token_logprob,
    }


def test_corpus_row_id_with_pool_size():
    row = {
        "item_id": "META-01",
        "chain_type": "pwt",
        "pool_size": 3,
        "generation": 1,
        "degradation_seed": 1,
    }
    assert corpus_row_id(row) == "META-01-pwt-p3-g1-s1"


def test_corpus_row_id_without_pool_size():
    row = {
        "item_id": "META-01",
        "chain_type": "gen0",
        "pool_size": None,
        "generation": 0,
        "degradation_seed": 1,
    }
    assert corpus_row_id(row) == "META-01-gen0-pNA-g0-s1"


def test_load_jsonl_reads_lines_and_skips_blanks(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    rows = load_jsonl(path)
    assert rows == [{"a": 1}, {"a": 2}]


def test_build_original_prompt_lookup_uses_gen0_text():
    corpus_rows = [
        {"item_id": "META-01", "chain_type": "gen0", "text": "The original prompt."},
        {"item_id": "META-01", "chain_type": "pwt", "text": "Degraded garbled text."},
        {"item_id": "META-02", "chain_type": "gen0", "text": "Second item's prompt."},
    ]
    lookup = build_original_prompt_lookup(corpus_rows)
    assert lookup == {
        "META-01": "The original prompt.",
        "META-02": "Second item's prompt.",
    }


def test_join_corpus_responses_merges_fields_by_corpus_row_id():
    corpus_rows = [
        {
            "item_id": "META-01",
            "chain_type": "gen0",
            "pool_size": None,
            "generation": 0,
            "degradation_seed": 1,
            "text": "The original prompt.",
            "pool_composition": None,
            "realized_language_count": 1,
            "distinct_script_count": 1,
            "pass_through_rate": 1.0,
            "semantic_distance_from_gen0": 0.0,
            "token_count": 5,
        },
        {
            "item_id": "META-01",
            "chain_type": "pwt",
            "pool_size": 3,
            "generation": 1,
            "degradation_seed": 1,
            "text": "Degraded garbled text.",
            "pool_composition": ["es", "ru", "ko"],
            "realized_language_count": 3,
            "distinct_script_count": 2,
            "pass_through_rate": 0.4,
            "semantic_distance_from_gen0": 0.7,
            "token_count": 8,
        },
    ]
    response_rows = [
        {
            "response_id": "META-01-pwt-p3-g1-s1-qwen3-8b-0",
            "corpus_row_id": "META-01-pwt-p3-g1-s1",
            "item_id": "META-01",
            "condition": "meta",
            "chain_type": "pwt",
            "pool_size": 3,
            "generation": 1,
            "degradation_seed": 1,
            "model": "qwen3-8b",
            "sample_idx": 0,
            "response_text": "A response.",
            "mean_token_logprob": -0.5,
        },
    ]

    df = join_corpus_responses(corpus_rows, response_rows)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["response_id"] == "META-01-pwt-p3-g1-s1-qwen3-8b-0"
    assert row["degraded_prompt"] == "Degraded garbled text."
    assert row["original_prompt"] == "The original prompt."
    assert row["pool_composition"] == ["es", "ru", "ko"]
    assert row["realized_language_count"] == 3
    assert row["pass_through_rate"] == 0.4


def test_join_corpus_responses_gen0_case_has_matching_original_and_degraded():
    corpus_rows = [
        {
            "item_id": "META-02",
            "chain_type": "gen0",
            "pool_size": None,
            "generation": 0,
            "degradation_seed": 1,
            "text": "Clean prompt text.",
            "pool_composition": None,
            "realized_language_count": 1,
            "distinct_script_count": 1,
            "pass_through_rate": 1.0,
            "semantic_distance_from_gen0": 0.0,
            "token_count": 4,
        },
    ]
    response_rows = [
        {
            "response_id": "META-02-gen0-pNA-g0-s1-qwen3-8b-0",
            "corpus_row_id": "META-02-gen0-pNA-g0-s1",
            "item_id": "META-02",
            "condition": "meta",
            "chain_type": "gen0",
            "pool_size": None,
            "generation": 0,
            "degradation_seed": 1,
            "model": "qwen3-8b",
            "sample_idx": 0,
            "response_text": "A response.",
            "mean_token_logprob": -0.2,
        },
    ]

    df = join_corpus_responses(corpus_rows, response_rows)

    row = df.iloc[0]
    assert row["original_prompt"] == row["degraded_prompt"] == "Clean prompt text."


def test_pivot_judged_combines_outcome_and_comprehension_rows():
    judged_rows = [
        {
            "response_id": "r1",
            "call_type": "outcome",
            "parsed": {"outcome_category": "comprehended_complied", "outcome_justification": "Good."},
            "error": None,
        },
        {
            "response_id": "r1",
            "call_type": "comprehension",
            "parsed": {
                "gist_recovery": 1.0,
                "instruction_recovery": 0.9,
                "comprehension_justification": "Recovered both.",
            },
            "error": None,
        },
    ]
    df = pivot_judged(judged_rows)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["response_id"] == "r1"
    assert row["outcome_category"] == "comprehended_complied"
    assert row["gist_recovery"] == 1.0
    assert row["instruction_recovery"] == 0.9
    assert row["judge_error"] is None


def test_pivot_judged_handles_only_outcome_present():
    judged_rows = [
        {
            "response_id": "r2",
            "call_type": "outcome",
            "parsed": {"outcome_category": "safety_refusal", "outcome_justification": "Declined."},
            "error": None,
        },
    ]
    df = pivot_judged(judged_rows)
    row = df.iloc[0]
    assert row["outcome_category"] == "safety_refusal"
    assert pd.isna(row["gist_recovery"])
    assert pd.isna(row["comprehension_justification"])


def test_pivot_judged_surfaces_error_from_either_call():
    judged_rows = [
        {"response_id": "r3", "call_type": "outcome", "parsed": None, "error": "RateLimitError: boom"},
    ]
    df = pivot_judged(judged_rows)
    row = df.iloc[0]
    assert row["judge_error"] == "RateLimitError: boom"
    assert pd.isna(row["outcome_category"])


def test_pivot_judged_with_no_rows_returns_empty_dataframe_with_expected_columns():
    df = pivot_judged([])
    assert list(df.columns) == [
        "response_id",
        "outcome_category",
        "outcome_justification",
        "gist_recovery",
        "instruction_recovery",
        "comprehension_justification",
        "judge_error",
    ]
    assert len(df) == 0


def test_build_case_table_computes_perplexity_when_logprob_present():
    corpus_rows = [_corpus_row("META-01", "gen0", None, 0, 1, "Clean prompt.")]
    response_rows = [
        _response_row("META-01-gen0-pNA-g0-s1-qwen3-8b-0", "META-01-gen0-pNA-g0-s1", "META-01", "gen0", None, 0, 1, "qwen3-8b", 0, -0.5)
    ]
    df = build_case_table(corpus_rows, response_rows, judged_rows=[])
    row = df.iloc[0]
    assert math.isclose(row["perplexity"], math.exp(0.5))


def test_build_case_table_omits_perplexity_when_logprob_is_none():
    corpus_rows = [_corpus_row("META-01", "gen0", None, 0, 1, "Clean prompt.")]
    response_rows = [
        _response_row("META-01-gen0-pNA-g0-s1-qwen3-8b-0", "META-01-gen0-pNA-g0-s1", "META-01", "gen0", None, 0, 1, "qwen3-8b", 0, None)
    ]
    df = build_case_table(corpus_rows, response_rows, judged_rows=[])
    assert pd.isna(df.iloc[0]["perplexity"])


def test_build_case_table_left_joins_judged_leaving_nulls_when_not_yet_judged():
    corpus_rows = [_corpus_row("META-01", "gen0", None, 0, 1, "Clean prompt.")]
    response_rows = [
        _response_row("META-01-gen0-pNA-g0-s1-qwen3-8b-0", "META-01-gen0-pNA-g0-s1", "META-01", "gen0", None, 0, 1, "qwen3-8b", 0, -0.1)
    ]
    df = build_case_table(corpus_rows, response_rows, judged_rows=[])
    row = df.iloc[0]
    assert len(df) == 1
    assert pd.isna(row["outcome_category"])
    assert pd.isna(row["judge_error"])


def test_build_case_table_attaches_judged_verdict_when_present():
    corpus_rows = [_corpus_row("META-01", "gen0", None, 0, 1, "Clean prompt.")]
    response_rows = [
        _response_row("META-01-gen0-pNA-g0-s1-qwen3-8b-0", "META-01-gen0-pNA-g0-s1", "META-01", "gen0", None, 0, 1, "qwen3-8b", 0, -0.1)
    ]
    judged_rows = [
        {
            "response_id": "META-01-gen0-pNA-g0-s1-qwen3-8b-0",
            "call_type": "outcome",
            "parsed": {"outcome_category": "comprehended_complied", "outcome_justification": "Fine."},
            "error": None,
        },
    ]
    df = build_case_table(corpus_rows, response_rows, judged_rows)
    assert df.iloc[0]["outcome_category"] == "comprehended_complied"


def test_load_case_table_reads_and_joins_real_files(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    judged_path = tmp_path / "judged.jsonl"

    corpus_path.write_text(json.dumps(_corpus_row("META-01", "gen0", None, 0, 1, "Clean prompt.")) + "\n")
    responses_path.write_text(
        json.dumps(
            _response_row(
                "META-01-gen0-pNA-g0-s1-qwen3-8b-0",
                "META-01-gen0-pNA-g0-s1",
                "META-01",
                "gen0",
                None,
                0,
                1,
                "qwen3-8b",
                0,
                -0.1,
            )
        )
        + "\n"
    )
    judged_path.write_text(
        json.dumps(
            {
                "response_id": "META-01-gen0-pNA-g0-s1-qwen3-8b-0",
                "call_type": "outcome",
                "parsed": {"outcome_category": "comprehended_complied", "outcome_justification": "Fine."},
                "error": None,
            }
        )
        + "\n"
    )

    df = load_case_table(corpus_path, responses_path, judged_path)
    assert len(df) == 1
    assert df.iloc[0]["outcome_category"] == "comprehended_complied"


def test_load_case_table_handles_missing_judged_file(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    judged_path = tmp_path / "judged.jsonl"

    corpus_path.write_text(json.dumps(_corpus_row("META-01", "gen0", None, 0, 1, "Clean prompt.")) + "\n")
    responses_path.write_text(
        json.dumps(
            _response_row(
                "META-01-gen0-pNA-g0-s1-qwen3-8b-0",
                "META-01-gen0-pNA-g0-s1",
                "META-01",
                "gen0",
                None,
                0,
                1,
                "qwen3-8b",
                0,
                -0.1,
            )
        )
        + "\n"
    )

    df = load_case_table(corpus_path, responses_path, judged_path)
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["outcome_category"])
