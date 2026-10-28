import csv
import hashlib
import json
import time
from pathlib import Path

import yaml

from . import embedding, pwt, roundtrip, translator

MANIFEST_PATH = "data/prompts_manifest.yaml"
XSTEST_PATH = "data/external/xstest_sample_105.csv"
OUTPUT_PATH = "data/corpus.jsonl"

SEEDS = [1, 2, 3]
POOL_SWEEP = [3, 5, 10, 15, 30]


def load_manifest_items() -> list[dict]:
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def load_xstest_items() -> list[dict]:
    items = []
    with open(XSTEST_PATH) as f:
        for row in csv.DictReader(f):
            items.append(
                {
                    "item_id": f"XSTEST-{row['id']}",
                    "condition": "xstest",
                    "roles": ["xstest_depth"],
                    "text": row["prompt"],
                    "contains_proper_noun": row["contains_proper_noun"] == "True",
                    "type": row["type"],
                }
            )
    return items


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def pool_composition_hash(pool: list[str]) -> str:
    return hashlib.sha256(",".join(pool).encode()).hexdigest()[:16]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gen0_row(item: dict, seed: int, prompt_id: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "item_id": item["item_id"],
        "condition": item["condition"],
        "type": item.get("type"),
        "chain_type": "gen0",
        "pool_size": None,
        "pool_composition": None,
        "pool_composition_hash": None,
        "degradation_seed": seed,
        "generation": 0,
        "text": item["text"],
        "word_language_assignments": None,
        "realized_language_count": None,
        "distinct_script_count": None,
        "pass_through_rate": None,
        "token_count": translator.count_tokens(item["text"]),
        "semantic_distance_from_gen0": 0.0,
        "step_delta_from_prev_gen": None,
        "parent_generation_text_hash": None,
        "translator_model": translator.current_model_name(),
        "translator_decoding": "greedy",
        "retention_p": None,
        "created_at": now_iso(),
    }


def rows_from_pwt_trajectory(
    item: dict, seed: int, pool_size: int, pool: list[str], trajectory: list[dict], prompt_id: str
) -> list[dict]:
    rows = []
    gen0_text = trajectory[0]["text"]
    pool_hash = pool_composition_hash(pool)
    prev_text = None
    for step in trajectory:
        gen = step["generation"]
        text = step["text"]
        if gen == 0:
            prev_text = text
            continue
        assignments = step["word_language_assignments"]
        rows.append(
            {
                "prompt_id": prompt_id,
                "item_id": item["item_id"],
                "condition": item["condition"],
                "type": item.get("type"),
                "chain_type": "pwt",
                "pool_size": pool_size,
                "pool_composition": pool,
                "pool_composition_hash": pool_hash,
                "degradation_seed": seed,
                "generation": gen,
                "text": text,
                "word_language_assignments": assignments,
                "realized_language_count": pwt.realized_language_count(assignments),
                "distinct_script_count": pwt.distinct_script_count(assignments),
                "pass_through_rate": pwt.pass_through_rate(assignments),
                "token_count": translator.count_tokens(text),
                "semantic_distance_from_gen0": embedding.semantic_distance(gen0_text, text),
                "step_delta_from_prev_gen": embedding.semantic_distance(prev_text, text),
                "parent_generation_text_hash": hash_text(prev_text),
                "translator_model": translator.current_model_name(),
                "translator_decoding": "greedy",
                "retention_p": pwt.DEFAULT_RETENTION_P,
                "created_at": now_iso(),
            }
        )
        prev_text = text
    return rows


def rows_from_roundtrip_trajectory(item: dict, seed: int, trajectory: list[dict], prompt_id: str) -> list[dict]:
    rows = []
    gen0_text = trajectory[0]["text"]
    prev_text = None
    for step in trajectory:
        gen = step["generation"]
        text = step["text"]
        if gen == 0:
            prev_text = text
            continue
        rows.append(
            {
                "prompt_id": prompt_id,
                "item_id": item["item_id"],
                "condition": item["condition"],
                "type": item.get("type"),
                "chain_type": "roundtrip",
                "pool_size": None,
                "pool_composition": None,
                "pool_composition_hash": None,
                "degradation_seed": seed,
                "generation": gen,
                "text": text,
                "word_language_assignments": None,
                "realized_language_count": None,
                "distinct_script_count": None,
                "pass_through_rate": None,
                "token_count": translator.count_tokens(text),
                "semantic_distance_from_gen0": embedding.semantic_distance(gen0_text, text),
                "step_delta_from_prev_gen": embedding.semantic_distance(prev_text, text),
                "parent_generation_text_hash": hash_text(prev_text),
                "translator_model": translator.current_model_name(),
                "translator_decoding": "greedy",
                "retention_p": None,
                "created_at": now_iso(),
            }
        )
        prev_text = text
    return rows


def process_core_grid_item(item: dict, seed: int) -> list[dict]:
    prompt_id = f"{item['item_id']}-s{seed}"
    rows = [gen0_row(item, seed, prompt_id)]

    for pool_size in (3, 5, 15):
        pool = pwt.POOL_TIERS[pool_size]
        traj = pwt.pwt_generate(item["text"], pool=pool, iterations=1, seed=seed)
        rows.extend(rows_from_pwt_trajectory(item, seed, pool_size, pool, traj, prompt_id))

    pool10 = pwt.POOL_TIERS[10]
    traj10 = pwt.pwt_generate(item["text"], pool=pool10, iterations=3, seed=seed)
    rows.extend(rows_from_pwt_trajectory(item, seed, 10, pool10, traj10, prompt_id))

    pool30 = pwt.POOL_TIERS[30]
    traj30 = pwt.pwt_generate(item["text"], pool=pool30, iterations=8, seed=seed)
    rows.extend(rows_from_pwt_trajectory(item, seed, 30, pool30, traj30, prompt_id))

    traj_rt = roundtrip.roundtrip_generate(item["text"], iterations=8)
    rows.extend(rows_from_roundtrip_trajectory(item, seed, traj_rt, prompt_id))

    return rows


def process_depth_item(item: dict, seed: int) -> list[dict]:
    prompt_id = f"{item['item_id']}-s{seed}"
    rows = [gen0_row(item, seed, prompt_id)]

    for pool_size in POOL_SWEEP:
        pool = pwt.POOL_TIERS[pool_size]
        traj = pwt.pwt_generate(item["text"], pool=pool, iterations=1, seed=seed)
        rows.extend(rows_from_pwt_trajectory(item, seed, pool_size, pool, traj, prompt_id))

    return rows


def load_existing_prompt_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["prompt_id"])
    return ids


def main():
    manifest_items = load_manifest_items()
    core_grid_items = [it for it in manifest_items if "core_grid" in it["roles"]]
    calibration_only_items = [
        it for it in manifest_items if "calibration_anchor" in it["roles"] and "core_grid" not in it["roles"]
    ]
    xstest_items = load_xstest_items()
    depth_items = calibration_only_items + xstest_items

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_prompt_ids = load_existing_prompt_ids(out_path)

    total_units = (len(core_grid_items) + len(depth_items)) * len(SEEDS)
    done = 0

    with open(out_path, "a") as out:
        for item in core_grid_items:
            for seed in SEEDS:
                prompt_id = f"{item['item_id']}-s{seed}"
                done += 1
                if prompt_id in existing_prompt_ids:
                    print(f"[{done}/{total_units}] skip (already have) {prompt_id}")
                    continue
                print(f"[{done}/{total_units}] core-grid {prompt_id}")
                for row in process_core_grid_item(item, seed):
                    out.write(json.dumps(row) + "\n")
                out.flush()

        for item in depth_items:
            for seed in SEEDS:
                prompt_id = f"{item['item_id']}-s{seed}"
                done += 1
                if prompt_id in existing_prompt_ids:
                    print(f"[{done}/{total_units}] skip (already have) {prompt_id}")
                    continue
                print(f"[{done}/{total_units}] depth {prompt_id}")
                for row in process_depth_item(item, seed):
                    out.write(json.dumps(row) + "\n")
                out.flush()

    print(f"\nDone. Corpus written to {out_path}")


if __name__ == "__main__":
    main()
