import pandas as pd

SOURCE_PATH = "data/external/xstest_prompts.csv"
OUTPUT_PATH = "data/external/xstest_sample_30.csv"
PER_CATEGORY = 3
SEED = 42


def main():
    df = pd.read_csv(SOURCE_PATH)
    safe = df[df["label"] == "safe"]

    groups = []
    for category in sorted(safe["type"].unique()):
        groups.append(safe[safe["type"] == category].sample(n=PER_CATEGORY, random_state=SEED))
    sampled = pd.concat(groups).reset_index(drop=True)

    sampled.to_csv(OUTPUT_PATH, index=False)

    print(f"Sampled {len(sampled)} items across {sampled['type'].nunique()} categories.")
    print(sampled["type"].value_counts())
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
