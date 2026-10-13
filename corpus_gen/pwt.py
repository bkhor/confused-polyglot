import random

from . import translator

FLORES_CODES = {
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

LANG_SCRIPTS = {
    "en": "Latin", "es": "Latin", "fr": "Latin", "de": "Latin", "pt": "Latin",
    "it": "Latin", "nl": "Latin", "pl": "Latin", "tr": "Latin", "sv": "Latin",
    "vi": "Latin", "id": "Latin", "cs": "Latin", "ro": "Latin", "fi": "Latin",
    "da": "Latin", "hu": "Latin", "sw": "Latin",
    "ru": "Cyrillic", "uk": "Cyrillic",
    "ar": "Arabic", "fa": "Arabic",
    "hi": "Devanagari",
    "ko": "Hangul",
    "ja": "Jpan",
    "zh-CN": "Hans",
    "el": "Greek",
    "he": "Hebrew",
    "th": "Thai",
    "bn": "Bengali",
    "ta": "Tamil",
}

_POOL_3 = ["es", "ru", "ko"]
_POOL_5 = _POOL_3 + ["hi", "ja"]
_POOL_10 = _POOL_5 + ["el", "he", "th", "bn", "ta"]
_POOL_15 = _POOL_10 + ["fr", "de", "pt", "it", "nl"]
_POOL_30 = _POOL_15 + [
    "pl", "tr", "sv", "vi", "id", "cs", "ro", "fi", "da", "hu", "sw",
    "ar", "fa", "uk", "zh-CN",
]

POOL_TIERS = {3: _POOL_3, 5: _POOL_5, 10: _POOL_10, 15: _POOL_15, 30: _POOL_30}

DEFAULT_RETENTION_P = 0.15


def pwt_hop(word: str, current_flores: str, pool: list[str], retention_p: float, rng: random.Random) -> tuple:
    if rng.random() < retention_p:
        return word, current_flores, "en"
    lang_code = rng.choice(pool)
    tgt_flores = FLORES_CODES[lang_code]
    new_word = translator.translate_word(word, current_flores, tgt_flores)
    return new_word, tgt_flores, lang_code


def pwt_generate(
    text: str,
    pool: list[str],
    iterations: int,
    seed: int,
    retention_p: float = DEFAULT_RETENTION_P,
) -> list[dict]:
    rng = random.Random(seed)
    words = text.split()
    state = [{"word": w, "flores": FLORES_CODES["en"]} for w in words]

    trajectory = [
        {
            "generation": 0,
            "text": text,
            "word_language_assignments": ["en"] * len(words),
        }
    ]

    for gen in range(1, iterations + 1):
        new_state = []
        assignments = []
        for s in state:
            new_word, new_flores, lang_code = pwt_hop(s["word"], s["flores"], pool, retention_p, rng)
            new_state.append({"word": new_word, "flores": new_flores})
            assignments.append(lang_code)
        state = new_state
        trajectory.append(
            {
                "generation": gen,
                "text": " ".join(s["word"] for s in state),
                "word_language_assignments": assignments,
            }
        )

    return trajectory


def pass_through_rate(word_language_assignments: list[str]) -> float:
    if not word_language_assignments:
        return 0.0
    return sum(1 for a in word_language_assignments if a == "en") / len(word_language_assignments)


def realized_language_count(word_language_assignments: list[str]) -> int:
    return len({a for a in word_language_assignments if a != "en"})


def distinct_script_count(word_language_assignments: list[str]) -> int:
    return len({LANG_SCRIPTS[a] for a in word_language_assignments if a != "en"})
