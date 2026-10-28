import torch

from corpus_gen import embedding


class _FakeModel:
    def __init__(self):
        self.to_device = None

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        return self


class _FakeTokenizer:
    pass


def _reset_embedding_globals(monkeypatch):
    monkeypatch.setattr(embedding, "_tokenizer", None)
    monkeypatch.setattr(embedding, "_model", None)


def test_load_moves_model_to_cuda_with_bfloat16(monkeypatch):
    _reset_embedding_globals(monkeypatch)
    monkeypatch.setattr(embedding.translator, "get_device", lambda: torch.device("cuda"))

    captured = {}

    def fake_from_pretrained(name, **kwargs):
        captured.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(embedding.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())
    monkeypatch.setattr(embedding.AutoModel, "from_pretrained", fake_from_pretrained)

    embedding._load()

    assert captured["torch_dtype"] == torch.bfloat16
    assert embedding._model.to_device == torch.device("cuda")


def test_load_uses_float32_on_cpu(monkeypatch):
    _reset_embedding_globals(monkeypatch)
    monkeypatch.setattr(embedding.translator, "get_device", lambda: torch.device("cpu"))

    captured = {}

    def fake_from_pretrained(name, **kwargs):
        captured.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(embedding.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())
    monkeypatch.setattr(embedding.AutoModel, "from_pretrained", fake_from_pretrained)

    embedding._load()

    assert captured["torch_dtype"] == torch.float32
    assert embedding._model.to_device == torch.device("cpu")
