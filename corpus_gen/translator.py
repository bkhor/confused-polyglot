import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("PWT_TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M")

_tokenizer = None
_model = None
_device = None
_loaded_model_name = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(model_name: str | None = None) -> None:
    global _tokenizer, _model, _loaded_model_name
    model_name = model_name or DEFAULT_MODEL
    if _model is not None and _loaded_model_name == model_name:
        return
    device = get_device()
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    _model.eval()
    _loaded_model_name = model_name


def current_model_name() -> str:
    return _loaded_model_name or DEFAULT_MODEL


def translate_batch(
    words: list[str],
    src_flores: str,
    tgt_flores: str,
    model_name: str | None = None,
    max_new_tokens: int = 30,
) -> list[str]:
    load_model(model_name)
    device = get_device()
    _tokenizer.src_lang = src_flores
    inputs = _tokenizer(words, return_tensors="pt", padding=True).to(device)
    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(tgt_flores)
    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    return _tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_word(
    word: str,
    src_flores: str,
    tgt_flores: str,
    model_name: str | None = None,
    max_new_tokens: int = 30,
) -> str:
    return translate_batch([word], src_flores, tgt_flores, model_name=model_name, max_new_tokens=max_new_tokens)[0]
