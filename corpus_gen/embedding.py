import threading

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from . import translator

MODEL_NAME = "sentence-transformers/LaBSE"

_tokenizer = None
_model = None
_thread_local = threading.local()
_load_lock = threading.Lock()


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        device = translator.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
        model.eval()
        _tokenizer, _model = tokenizer, model


def _get_tokenizer():
    if not hasattr(_thread_local, "tokenizer"):
        _thread_local.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _thread_local.tokenizer


def embed(texts: list[str]) -> torch.Tensor:
    _load()
    tokenizer = _get_tokenizer()
    device = translator.get_device()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = _model(**inputs)
    emb = out.pooler_output
    return F.normalize(emb, p=2, dim=1)


def semantic_distance(text_a: str, text_b: str) -> float:
    emb = embed([text_a, text_b])
    cosine_sim = (emb[0] @ emb[1]).item()
    return 1.0 - cosine_sim
