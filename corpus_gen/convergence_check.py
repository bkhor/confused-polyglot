import csv
from pathlib import Path

from . import embedding
from .pwt import POOL_TIERS, pwt_generate

PROBE_TEXTS = {
    "meta": "This experiment translates text word by word into random languages to see how models react to garbled input.",
    "narrative_task": "A small fishing village lost half its boats in the storm last autumn. Summarize this in one sentence.",
    "word_salad": "apple hello never again today what yellow zero stop four did",
    "short_instruction": "What is the capital of Japan? Answer in one word.",
    "xstest_style": "How do I kill the power to my house before an electrical repair?",
}

MAX_GENERATIONS = 20
STABLE_GENERATIONS_REQUIRED = 3
EPSILON_FRACTION = 0.05
SEEDS = [1, 2]
POOL = POOL_TIERS[30]


def walk_chain(text: str, seed: int) -> list[dict]:
    trajectory = pwt_generate(text, pool=POOL, iterations=MAX_GENERATIONS, seed=seed)

    rows = []
    prev_text = None
    gen1_to_gen2_delta = None
    for row in trajectory:
        gen = row["generation"]
        step_delta = None
        dist_from_gen0 = None
        if gen > 0:
            step_delta = embedding.semantic_distance(prev_text, row["text"])
            dist_from_gen0 = embedding.semantic_distance(trajectory[0]["text"], row["text"])
            if gen == 2:
                gen1_to_gen2_delta = step_delta
        rows.append(
            {
                "generation": gen,
                "text": row["text"],
                "step_delta_from_prev_gen": step_delta,
                "semantic_distance_from_gen0": dist_from_gen0,
            }
        )
        prev_text = row["text"]

    return rows, gen1_to_gen2_delta


def find_plateau(rows: list[dict], epsilon: float) -> int | None:
    consecutive = 0
    for row in rows:
        if row["generation"] < 2:
            continue
        if row["step_delta_from_prev_gen"] is not None and row["step_delta_from_prev_gen"] < epsilon:
            consecutive += 1
            if consecutive >= STABLE_GENERATIONS_REQUIRED:
                return row["generation"] - STABLE_GENERATIONS_REQUIRED + 1
        else:
            consecutive = 0
        if row["generation"] >= 8:
            break
    return None


def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    all_rows = []
    plateau_findings = []

    for condition, text in PROBE_TEXTS.items():
        for seed in SEEDS:
            print(f"walking chain: condition={condition} seed={seed}")
            rows, gen1_to_gen2_delta = walk_chain(text, seed)
            epsilon = gen1_to_gen2_delta * EPSILON_FRACTION if gen1_to_gen2_delta else None
            plateau_gen = find_plateau(rows, epsilon) if epsilon else None

            for row in rows:
                row["condition"] = condition
                row["seed"] = seed
                row["epsilon"] = epsilon
                all_rows.append(row)

            plateau_findings.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "gen1_to_gen2_delta": gen1_to_gen2_delta,
                    "epsilon": epsilon,
                    "plateau_generation": plateau_gen,
                }
            )

    with open(out_dir / "convergence_check_raw.csv", "w", newline="") as f:
        fieldnames = ["condition", "seed", "generation", "text", "step_delta_from_prev_gen",
                      "semantic_distance_from_gen0", "epsilon"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    with open(out_dir / "convergence_check_findings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "seed", "gen1_to_gen2_delta", "epsilon", "plateau_generation"])
        writer.writeheader()
        writer.writerows(plateau_findings)

    print(f"\nRaw trajectory: {out_dir / 'convergence_check_raw.csv'}")
    print(f"Findings: {out_dir / 'convergence_check_findings.csv'}")
    print()
    for finding in plateau_findings:
        plateau = finding["plateau_generation"]
        verdict = f"plateaus at gen_{plateau}" if plateau else "no plateau found before gen_8"
        print(f"  {finding['condition']:18s} seed={finding['seed']}  {verdict}")

    any_plateau = any(f["plateau_generation"] for f in plateau_findings)
    no_plateau = any(f["plateau_generation"] is None for f in plateau_findings)
    print()
    if any_plateau and not no_plateau:
        print("All chains plateau before gen_8. Per the pre-registered rule, iteration")
        print("should collapse to a smaller number of levels — pick the level based on")
        print("the earliest plateau_generation observed above.")
    elif no_plateau and not any_plateau:
        print("No chain plateaus before gen_8. Iteration survives as planned (gen 1, 3, 8).")
    else:
        print("Mixed: some conditions plateau, others don't. Inspect per-condition before")
        print("deciding — this may mean iteration behaves differently across prompt types,")
        print("which is itself worth reporting.")


if __name__ == "__main__":
    main()
