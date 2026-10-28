import os
import threading

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("PWT_TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M")

_tokenizer = None
_model = None
_device = None
_loaded_model_name = None
_thread_local = threading.local()
_load_lock = threading.Lock()


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(model_name: str | None = None) -> None:
    # Not safe to call with a different model_name once worker threads are
    # already dispatched -- in-flight threads would get this call's new
    # _model paired with a stale thread-local tokenizer for the old one.
    # Nothing in this codebase does that today (only PWT_TRANSLATOR_MODEL,
    # set once at import time, varies the model).
    global _tokenizer, _model, _loaded_model_name
    model_name = model_name or DEFAULT_MODEL
    if _model is not None and _loaded_model_name == model_name:
        return
    with _load_lock:
        if _model is not None and _loaded_model_name == model_name:
            return
        device = get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        _model.eval()
        _model.generation_config.max_length = None
        _loaded_model_name = model_name


def _get_tokenizer():
    if getattr(_thread_local, "tokenizer_model_name", None) != _loaded_model_name:
        _thread_local.tokenizer = AutoTokenizer.from_pretrained(_loaded_model_name)
        _thread_local.tokenizer_model_name = _loaded_model_name
    return _thread_local.tokenizer


def current_model_name() -> str:
    return _loaded_model_name or DEFAULT_MODEL


def count_tokens(text: str, model_name: str | None = None) -> int:
    load_model(model_name)
    tokenizer = _get_tokenizer()
    return len(tokenizer(text)["input_ids"])


def translate_batch(
    words: list[str],
    src_flores: str,
    tgt_flores: str,
    model_name: str | None = None,
    max_new_tokens: int = 30,
) -> list[str]:
    load_model(model_name)
    device = get_device()
    tokenizer = _get_tokenizer()
    tokenizer.src_lang = src_flores
    inputs = tokenizer(words, return_tensors="pt", padding=True).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_flores)
    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def translate_word(
    word: str,
    src_flores: str,
    tgt_flores: str,
    model_name: str | None = None,
    max_new_tokens: int = 30,
) -> str:
    return translate_batch([word], src_flores, tgt_flores, model_name=model_name, max_new_tokens=max_new_tokens)[0]
