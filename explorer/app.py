import random
from pathlib import Path

import pandas as pd
import streamlit as st

from explorer.data import load_case_table

HERE = Path(__file__).parent
CORPUS_PATH = HERE.parent / "data" / "corpus.jsonl"
RESPONSES_PATH = HERE.parent / "data" / "responses.jsonl"
JUDGED_PATH = HERE.parent / "data" / "judged.jsonl"

FILTER_COLUMNS = ["model", "condition", "chain_type", "pool_size", "outcome_category"]


@st.cache_data
def get_case_table() -> pd.DataFrame:
    return load_case_table(CORPUS_PATH, RESPONSES_PATH, JUDGED_PATH)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df
    for column, value in filters.items():
        if value != "Any":
            filtered = filtered[filtered[column] == value]
    return filtered


def render_filters(df: pd.DataFrame) -> dict:
    filters = {}
    for column in FILTER_COLUMNS:
        options = ["Any"] + sorted(df[column].dropna().unique().tolist(), key=str)
        filters[column] = st.sidebar.selectbox(column, options)
    return filters


def render_case(row: pd.Series) -> None:
    st.header(f"Case: {row['response_id']}")

    st.subheader("Prompt")
    st.write("**Original prompt:**", row["original_prompt"])
    st.write("**Degraded prompt:**", row["degraded_prompt"])
    st.write(
        "**Condition / chain_type / pool_size / generation / seed:**",
        f"{row['condition']} / {row['chain_type']} / {row['pool_size']} / {row['generation']} / {row['degradation_seed']}",
    )
    if row["pool_composition"]:
        st.write("**Languages used:**", ", ".join(row["pool_composition"]))

    st.subheader("Degradation covariates")
    st.write("**Realized language count:**", row["realized_language_count"])
    st.write("**Distinct script count:**", row["distinct_script_count"])
    st.write("**Pass-through rate:**", row["pass_through_rate"])
    st.write("**Semantic distance from gen0:**", row["semantic_distance_from_gen0"])
    st.write("**Token count:**", row["token_count"])

    st.subheader("Response")
    st.write("**Model / sample:**", f"{row['model']} / sample {row['sample_idx']}")
    st.write(row["response_text"])
    if pd.notna(row["perplexity"]):
        st.write("**Perplexity:**", row["perplexity"])

    st.subheader("Judge verdict")
    if pd.notna(row["judge_error"]):
        st.error(f"Judge call failed: {row['judge_error']}")
    if pd.isna(row["outcome_category"]) and pd.isna(row["judge_error"]):
        st.info("Not yet judged.")
    else:
        if pd.notna(row["outcome_category"]):
            st.write("**Outcome category:**", row["outcome_category"])
            st.write("**Outcome justification:**", row["outcome_justification"])
        else:
            st.info("Outcome call not yet judged.")
        if pd.notna(row["gist_recovery"]) or pd.notna(row["instruction_recovery"]):
            st.write("**Gist recovery:**", row["gist_recovery"])
            st.write("**Instruction recovery:**", row["instruction_recovery"])
            st.write("**Comprehension justification:**", row["comprehension_justification"])
        else:
            st.info("Comprehension call not yet judged.")


def main() -> None:
    st.title("PWT Case Explorer")

    df = get_case_table()
    filters = render_filters(df)
    filtered = apply_filters(df, filters)

    if filtered.empty:
        st.warning("No matching cases for the current filters.")
        return

    response_ids = filtered["response_id"].tolist()

    if "selected_response_id" not in st.session_state or st.session_state.selected_response_id not in response_ids:
        st.session_state.selected_response_id = response_ids[0]

    selected = st.selectbox(
        "Case (response_id)",
        response_ids,
        index=response_ids.index(st.session_state.selected_response_id),
    )
    st.session_state.selected_response_id = selected

    col1, col2, col3 = st.columns(3)
    if col1.button("Random"):
        st.session_state.selected_response_id = random.choice(response_ids)
        st.rerun()
    if col2.button("Previous"):
        idx = response_ids.index(st.session_state.selected_response_id)
        st.session_state.selected_response_id = response_ids[(idx - 1) % len(response_ids)]
        st.rerun()
    if col3.button("Next"):
        idx = response_ids.index(st.session_state.selected_response_id)
        st.session_state.selected_response_id = response_ids[(idx + 1) % len(response_ids)]
        st.rerun()

    row = filtered[filtered["response_id"] == st.session_state.selected_response_id].iloc[0]
    render_case(row)


if __name__ == "__main__":
    main()
