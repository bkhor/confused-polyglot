from . import translator

BRIDGE_FLORES_DEFAULT = "deu_Latn"
EN_FLORES = "eng_Latn"


def roundtrip_generate(
    text: str, iterations: int, bridge_flores: str = BRIDGE_FLORES_DEFAULT, max_new_tokens: int = 256
) -> list[dict]:
    trajectory = [{"generation": 0, "text": text}]
    current = text
    for gen in range(1, iterations + 1):
        bridged = translator.translate_word(current, EN_FLORES, bridge_flores, max_new_tokens=max_new_tokens)
        back = translator.translate_word(bridged, bridge_flores, EN_FLORES, max_new_tokens=max_new_tokens)
        current = back
        trajectory.append({"generation": gen, "text": current})
    return trajectory
