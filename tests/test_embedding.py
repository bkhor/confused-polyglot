import threading

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


def test_load_calls_eval_before_publishing_model_globally(monkeypatch):
    _reset_embedding_globals(monkeypatch)
    monkeypatch.setattr(embedding.translator, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(embedding.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    observed = {"value": None}

    class _TrackingModel:
        def __init__(self):
            self.to_device = None

        def to(self, device):
            self.to_device = device
            return self

        def eval(self):
            observed["value"] = embedding._model is None
            return self

    monkeypatch.setattr(embedding.AutoModel, "from_pretrained", lambda name, **kw: _TrackingModel())

    embedding._load()

    assert observed["value"] is True


def test_get_tokenizer_returns_same_instance_within_a_thread_and_different_across_threads(monkeypatch):
    _reset_embedding_globals(monkeypatch)
    monkeypatch.setattr(embedding.translator, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(embedding.AutoModel, "from_pretrained", lambda name, **kw: _FakeModel())
    monkeypatch.setattr(embedding.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    embedding._load()

    results = {}

    def worker(name):
        results[name] = embedding._get_tokenizer()
        results[name + "-again"] = embedding._get_tokenizer()

    t1 = threading.Thread(target=worker, args=("t1",))
    t2 = threading.Thread(target=worker, args=("t2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results["t1"] is results["t1-again"]
    assert results["t2"] is results["t2-again"]
    assert results["t1"] is not results["t2"]


def test_load_invoked_once_under_concurrent_first_callers(monkeypatch):
    _reset_embedding_globals(monkeypatch)
    monkeypatch.setattr(embedding.translator, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(embedding.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    call_count = {"n": 0}
    load_started = threading.Event()
    proceed = threading.Event()

    def fake_from_pretrained(name, **kwargs):
        call_count["n"] += 1
        load_started.set()
        proceed.wait(timeout=2)
        return _FakeModel()

    monkeypatch.setattr(embedding.AutoModel, "from_pretrained", fake_from_pretrained)

    threads = [threading.Thread(target=embedding._load) for _ in range(5)]
    for t in threads:
        t.start()
    load_started.wait(timeout=2)
    proceed.set()
    for t in threads:
        t.join()

    assert call_count["n"] == 1
