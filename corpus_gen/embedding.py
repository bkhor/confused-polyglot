import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from . import translator

MODEL_NAME = "sentence-transformers/LaBSE"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        device = translator.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
        _model.eval()


def embed(texts: list[str]) -> torch.Tensor:
    _load()
    device = translator.get_device()
    inputs = _tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = _model(**inputs)
    emb = out.pooler_output
    return F.normalize(emb, p=2, dim=1)


def semantic_distance(text_a: str, text_b: str) -> float:
    emb = embed([text_a, text_b])
    cosine_sim = (emb[0] @ emb[1]).item()
    return 1.0 - cosine_sim
