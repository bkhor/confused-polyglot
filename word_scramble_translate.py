#!/usr/bin/env python3
import random
import sys

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    sys.exit("Missing dependencies. Install with: pip install torch transformers sentencepiece")

MODEL_NAME = "facebook/nllb-200-distilled-600M"

LANGUAGES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "zh-CN": "zho_Hans",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "sv": "swe_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "fi": "fin_Latn",
    "da": "dan_Latn",
    "hu": "hun_Latn",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
    "fa": "pes_Arab",
    "ta": "tam_Taml",
}

_tokenizer = None
_model = None


def load_model() -> None:
    global _tokenizer, _model
    if _model is None:
        print("Loading model (first run downloads ~2.4GB, then cached)...", file=sys.stderr)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def translate_word(word: str, current_flores: str, target_code: str) -> tuple[str, str]:
    if target_code == "en":
        return word, current_flores

    load_model()
    target_flores = LANGUAGES[target_code]
    _tokenizer.src_lang = current_flores
    inputs = _tokenizer(word, return_tensors="pt")
    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(target_flores)
    generated = _model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=20)
    new_word = _tokenizer.decode(generated[0], skip_special_tokens=True)
    return new_word, target_flores


def scramble_translate(text: str, iterations: int) -> str:
    state = [(w, "eng_Latn") for w in text.split()]

    for _ in range(iterations):
        next_state = []
        for word, flores in state:
            target_code = random.choice(list(LANGUAGES))
            next_state.append(translate_word(word, flores, target_code))
        state = next_state

    return " ".join(word for word, _ in state)


def main() -> None:
    if len(sys.argv) > 1:
        text = sys.argv[1]
        iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    else:
        text = input("Enter text: ")
        raw = input("Number of iterations [1]: ").strip()
        iterations = int(raw) if raw else 1

    result = scramble_translate(text, iterations)
    print(result)


if __name__ == "__main__":
    main()
