import re

from . import translator

BRIDGE_FLORES_DEFAULT = "deu_Latn"
EN_FLORES = "eng_Latn"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]


def roundtrip_sentence(sentence: str, bridge_flores: str, max_new_tokens: int) -> str:
    bridged = translator.translate_word(sentence, EN_FLORES, bridge_flores, max_new_tokens=max_new_tokens)
    back = translator.translate_word(bridged, bridge_flores, EN_FLORES, max_new_tokens=max_new_tokens)
    return back


def roundtrip_generate(
    text: str, iterations: int, bridge_flores: str = BRIDGE_FLORES_DEFAULT, max_new_tokens: int = 256
) -> list[dict]:
    sentences = split_sentences(text)
    trajectory = [{"generation": 0, "text": text}]
    current_sentences = sentences
    for gen in range(1, iterations + 1):
        current_sentences = [
            roundtrip_sentence(s, bridge_flores, max_new_tokens) for s in current_sentences
        ]
        trajectory.append({"generation": gen, "text": " ".join(current_sentences)})
    return trajectory
