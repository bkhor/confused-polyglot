import csv
from pathlib import Path

from . import translator
from .pwt import FLORES_CODES, POOL_TIERS

PROBE_WORDS = {
    "misspelling": ["teh", "recieve", "definately", "occured", "seperate"],
    "proper_noun": ["Shakespeare", "Tokyo", "Zendaya", "Kubernetes", "Patagonia"],
    "number_date": ["42", "2026", "3.14", "January", "9:30am"],
    "rare_word": ["sesquipedalian", "petrichor", "ersatz", "ultracrepidarian", "apophenia"],
    "slang": ["lol", "yeet", "rn", "tbh", "sus"],
}

EN_FLORES = FLORES_CODES["en"]


def run_probe(pool: list[str] = None) -> list[dict]:
    pool = pool or POOL_TIERS[30]
    rows = []
    for category, words in PROBE_WORDS.items():
        for word in words:
            for lang in pool:
                tgt_flores = FLORES_CODES[lang]
                translated = translator.translate_word(word, EN_FLORES, tgt_flores)
                passed_through = translated.strip().lower() == word.strip().lower()
                rows.append(
                    {
                        "category": category,
                        "word": word,
                        "lang": lang,
                        "translated": translated,
                        "passed_through": passed_through,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = {}
    for row in rows:
        key = (row["category"], row["lang"])
        entry = summary.setdefault(key, {"category": row["category"], "lang": row["lang"], "total": 0, "passed": 0})
        entry["total"] += 1
        entry["passed"] += int(row["passed_through"])
    out = []
    for entry in summary.values():
        entry["pass_through_rate"] = entry["passed"] / entry["total"]
        out.append(entry)
    return out


def main():
    rows = run_probe()
    summary = summarize(rows)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "oov_probe_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "word", "lang", "translated", "passed_through"])
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "oov_probe_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "lang", "total", "passed", "pass_through_rate"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"Probed {len(rows)} (word, language) pairs across {len(PROBE_WORDS)} categories.")
    print(f"Raw results: {out_dir / 'oov_probe_raw.csv'}")
    print(f"Summary: {out_dir / 'oov_probe_summary.csv'}")

    by_category = {}
    for row in summary:
        by_category.setdefault(row["category"], []).append(row["pass_through_rate"])
    for category, rates in by_category.items():
        avg = sum(rates) / len(rates)
        print(f"  {category:15s} avg pass-through across languages: {avg:.0%}")


if __name__ == "__main__":
    main()
