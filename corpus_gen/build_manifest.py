import yaml

SOURCE_PATH = "data/prompts_source.yaml"
OUTPUT_PATH = "data/prompts_manifest.yaml"


def find_instruction_span(text: str, instruction_sentence: str | None) -> list[int] | None:
    if instruction_sentence is None:
        return None
    words = text.split()
    instr_words = instruction_sentence.split()
    n = len(instr_words)
    for start in range(len(words) - n, -1, -1):
        if words[start : start + n] == instr_words:
            return [start, start + n]
    raise ValueError(f"instruction sentence not found verbatim in text: {instruction_sentence!r}")


def count_content_words(text: str) -> int:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "in", "on", "at", "to", "of", "for", "with", "by", "from", "as",
        "and", "or", "but", "if", "so", "that", "this", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
        "whom", "would", "will", "shall", "should", "can", "could", "do",
        "does", "did", "has", "have", "had", "not", "no", "yes", "answer",
        "one", "word", "number", "surname",
    }
    words = [w.strip(".,?!:;\"'()").lower() for w in text.split()]
    return sum(1 for w in words if w and w not in stopwords)


def build(source_items: list[dict]) -> list[dict]:
    manifest = []
    for item in source_items:
        text = item["text"]
        span = find_instruction_span(text, item["instruction_sentence"])
        entry = {
            "item_id": item["item_id"],
            "condition": item["condition"],
            "roles": item["roles"],
            "text": text,
            "instruction_span": span,
            "expected_answer": item["expected_answer"],
            "accept_set": item["accept_set"],
            "contains_proper_noun": item["contains_proper_noun"],
            "explicitly_told_degraded": item["explicitly_told_degraded"],
            "language_origin": item["language_origin"],
            "token_count": len(text.split()),
            "content_word_count": count_content_words(text),
        }
        manifest.append(entry)
    return manifest


def main():
    with open(SOURCE_PATH) as f:
        source_items = yaml.safe_load(f)

    manifest = build(source_items)

    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(manifest, f, sort_keys=False, allow_unicode=True, width=100)

    print(f"Read {len(source_items)} items from {SOURCE_PATH}")
    print(f"Wrote {len(manifest)} items to {OUTPUT_PATH}\n")
    for entry in manifest:
        span = entry["instruction_span"]
        print(
            f"{entry['item_id']:8s} [{entry['condition']:18s}] tokens={entry['token_count']:3d} "
            f"content_words={entry['content_word_count']:2d} proper_noun={entry['contains_proper_noun']!s:5s} "
            f"instr_span={span}"
        )


if __name__ == "__main__":
    main()
