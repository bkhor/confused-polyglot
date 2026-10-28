import threading

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


def test_get_tokenizer_returns_same_instance_within_a_thread_and_different_across_threads(monkeypatch):
    _reset_translator_globals(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(translator.AutoModelForSeq2SeqLM, "from_pretrained", lambda name, **kw: _FakeModel())
    monkeypatch.setattr(translator.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    translator.load_model("fake-model")

    results = {}

    def worker(name):
        results[name] = translator._get_tokenizer()
        results[name + "-again"] = translator._get_tokenizer()

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["t1"] is results["t1-again"]
    assert results["t2"] is results["t2-again"]
    assert results["t1"] is not results["t2"]


def test_load_model_invoked_once_under_concurrent_first_callers(monkeypatch):
    _reset_translator_globals(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(translator.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    call_count = {"n": 0}
    load_started = threading.Event()
    proceed = threading.Event()

    def fake_from_pretrained(name, **kwargs):
        call_count["n"] += 1
        load_started.set()
        proceed.wait(timeout=2)
        return _FakeModel()

    monkeypatch.setattr(translator.AutoModelForSeq2SeqLM, "from_pretrained", fake_from_pretrained)

    threads = [threading.Thread(target=translator.load_model, args=("fake-model",)) for _ in range(5)]
    for t in threads:
        t.start()
    load_started.wait(timeout=2)
    proceed.set()
    for t in threads:
        t.join()

    assert call_count["n"] == 1
