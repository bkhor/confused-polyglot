import torch

from corpus_gen import translator


class _FakeGenerationConfig:
    pass


class _FakeModel:
    def __init__(self):
        self.to_device = None
        self.generation_config = _FakeGenerationConfig()

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        return self


class _FakeTokenizer:
    pass


def _reset_translator_globals(monkeypatch):
    monkeypatch.setattr(translator, "_tokenizer", None)
    monkeypatch.setattr(translator, "_model", None)
    monkeypatch.setattr(translator, "_device", None)
    monkeypatch.setattr(translator, "_loaded_model_name", None)


def test_load_model_uses_bfloat16_on_cuda(monkeypatch):
    _reset_translator_globals(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(translator.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())
    monkeypatch.setattr(translator.AutoModelForSeq2SeqLM, "from_pretrained", fake_from_pretrained)

    translator.load_model("fake-model")

    assert captured["torch_dtype"] == torch.bfloat16
    assert translator._model.to_device == torch.device("cuda")


def test_load_model_uses_float32_on_cpu(monkeypatch):
    _reset_translator_globals(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(translator.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())
    monkeypatch.setattr(translator.AutoModelForSeq2SeqLM, "from_pretrained", fake_from_pretrained)

    translator.load_model("fake-model")

    assert captured["torch_dtype"] == torch.float32
    assert translator._model.to_device == torch.device("cpu")
