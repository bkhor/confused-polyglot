import json

INPUT_PATH = "data/corpus.jsonl"
OUTPUT_PATH = "data/inference_input.jsonl"

ANALYSIS_GENERATIONS = {
    3: {1},
    5: {1},
    15: {1},
    10: {1, 3},
    30: {1, 3, 8},
}


def corpus_row_id(row: dict) -> str:
    pool = row["pool_size"]
    pool_str = f"p{int(pool)}" if pool is not None else "pNA"
    return f"{row['item_id']}-{row['chain_type']}-{pool_str}-g{row['generation']}-s{row['degradation_seed']}"


def is_analysis_cell(row: dict) -> bool:
    if row["chain_type"] == "gen0":
        return True
    if row["chain_type"] == "roundtrip":
        return row["generation"] in {1, 3, 8}
    if row["chain_type"] == "pwt":
        pool = int(row["pool_size"])
        return row["generation"] in ANALYSIS_GENERATIONS[pool]
    return False


def main():
    kept = 0
    total = 0
    with open(INPUT_PATH) as f_in, open(OUTPUT_PATH, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if not is_analysis_cell(row):
                continue
            row["corpus_row_id"] = corpus_row_id(row)
            f_out.write(json.dumps(row) + "\n")
            kept += 1

    print(f"Read {total} corpus rows, kept {kept} analysis-cell rows.")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
