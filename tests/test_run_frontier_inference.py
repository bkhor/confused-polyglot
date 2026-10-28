import json

import run_frontier_inference
import run_inference
from run_frontier_inference import build_response_row, is_frontier_scope, load_existing_corpus_row_ids


def test_temperature_and_max_tokens_are_imported_not_duplicated():
    assert run_frontier_inference.TEMPERATURE is run_inference.TEMPERATURE
    assert run_frontier_inference.MAX_TOKENS is run_inference.MAX_TOKENS


def test_save_batch_state_writes_atomically_no_stray_temp_file(tmp_path, monkeypatch):
    from run_frontier_inference import load_batch_state, save_batch_state

    state_path = tmp_path / "frontier_batch_state.json"
    monkeypatch.setattr(run_frontier_inference, "BATCH_STATE_PATH", state_path)

    id_map = {"abc": _sample_row("r1")}
    save_batch_state(["b1", "b2"], id_map)

    assert json.loads(state_path.read_text()) == {"batch_ids": ["b1", "b2"], "id_map": id_map}
    leftover = [p for p in tmp_path.iterdir() if p != state_path]
    assert leftover == []

    batch_ids, loaded_id_map = load_batch_state()
    assert batch_ids == ["b1", "b2"]
    assert loaded_id_map == id_map


def test_gen0_rows_are_always_in_scope():
    assert is_frontier_scope({"chain_type": "gen0", "generation": 0, "pool_size": None}) is True


def test_pwt_generation_1_is_in_scope():
    assert is_frontier_scope({"chain_type": "pwt", "generation": 1, "pool_size": 10}) is True


def test_pwt_generation_3_is_out_of_scope():
    assert is_frontier_scope({"chain_type": "pwt", "generation": 3, "pool_size": 10}) is False


def test_pwt_generation_8_is_out_of_scope():
    assert is_frontier_scope({"chain_type": "pwt", "generation": 8, "pool_size": 30}) is False


def test_roundtrip_rows_are_always_out_of_scope():
    assert is_frontier_scope({"chain_type": "roundtrip", "generation": 1, "pool_size": None}) is False
    assert is_frontier_scope({"chain_type": "roundtrip", "generation": 8, "pool_size": None}) is False


def test_build_response_row_produces_expected_schema():
    row = {
        "corpus_row_id": "META-01-gen0-pNA-g0-s1",
        "item_id": "META-01",
        "condition": "meta",
        "chain_type": "gen0",
        "pool_size": None,
        "generation": 0,
        "degradation_seed": 1,
    }
    out = build_response_row(row, "This is a test response.", "claude-sonnet-5-5")

    assert out["response_id"] == "META-01-gen0-pNA-g0-s1-claude-sonnet-5-5-0"
    assert out["corpus_row_id"] == "META-01-gen0-pNA-g0-s1"
    assert out["item_id"] == "META-01"
    assert out["condition"] == "meta"
    assert out["chain_type"] == "gen0"
    assert out["pool_size"] is None
    assert out["generation"] == 0
    assert out["degradation_seed"] == 1
    assert out["model"] == "claude-sonnet-5-5"
    assert out["param_count"] is None
    assert out["sample_idx"] == 0
    assert out["response_text"] == "This is a test response."
    assert out["mean_token_logprob"] is None
    assert isinstance(out["temperature"], float)
    assert isinstance(out["system_prompt_hash"], str)
    assert isinstance(out["created_at"], str)


def test_build_response_row_preserves_pwt_pool_size():
    row = {
        "corpus_row_id": "META-01-pwt-p10-g1-s1",
        "item_id": "META-01",
        "condition": "meta",
        "chain_type": "pwt",
        "pool_size": 10,
        "generation": 1,
        "degradation_seed": 1,
    }
    out = build_response_row(row, "Response text.", "claude-sonnet-5-5")
    assert out["pool_size"] == 10
    assert out["chain_type"] == "pwt"


def test_load_existing_corpus_row_ids_filters_by_model(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({"corpus_row_id": "r1", "model": "claude-sonnet-5-5"}) + "\n"
        + json.dumps({"corpus_row_id": "r2", "model": "qwen3-8b"}) + "\n"
        + json.dumps({"corpus_row_id": "r3", "model": "claude-sonnet-5-5"}) + "\n"
    )
    existing = load_existing_corpus_row_ids(path, "claude-sonnet-5-5")
    assert existing == {"r1", "r3"}


def test_load_existing_corpus_row_ids_empty_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.jsonl"
    assert load_existing_corpus_row_ids(path, "claude-sonnet-5-5") == set()


class _FakeMessage:
    def __init__(self, content_blocks):
        self.content = content_blocks


def _text_block(text):
    return type("Block", (), {"type": "text", "text": text})()


class _FakeSucceededResult:
    type = "succeeded"

    def __init__(self, text):
        self.message = _FakeMessage([_text_block(text)])


class _FakeSucceededEmptyContentResult:
    """Simulates a real API outcome: stop_reason='refusal' or a max_tokens cutoff
    with zero emitted text, where content is an empty list (no text block at all)."""

    type = "succeeded"

    def __init__(self):
        self.message = _FakeMessage([])


class _FakeErrorResult:
    type = "errored"


class _FakeItem:
    def __init__(self, custom_id, result):
        self.custom_id = custom_id
        self.result = result


class _FakeBatches:
    def __init__(self, items):
        self._items = items

    def results(self, batch_id):
        return self._items


class _FakeMessages:
    def __init__(self, items):
        self.batches = _FakeBatches(items)


class _FakeClient:
    def __init__(self, items):
        self.messages = _FakeMessages(items)


def _sample_row(corpus_row_id):
    return {
        "corpus_row_id": corpus_row_id,
        "item_id": "META-01",
        "condition": "meta",
        "chain_type": "gen0",
        "pool_size": None,
        "generation": 0,
        "degradation_seed": 1,
    }


def test_write_frontier_batch_results_writes_succeeded_and_skips_errored(tmp_path):
    from run_frontier_inference import write_frontier_batch_results

    out_path = tmp_path / "responses.jsonl"
    id_map = {"abc": _sample_row("r1"), "def": _sample_row("r2")}
    items = [
        _FakeItem("abc", _FakeSucceededResult("A real response.")),
        _FakeItem("def", _FakeErrorResult()),
    ]
    client = _FakeClient(items)

    write_frontier_batch_results(client, "batch-1", id_map, "claude-sonnet-5-5", out_path)

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["corpus_row_id"] == "r1"
    assert lines[0]["response_text"] == "A real response."


def test_write_frontier_batch_results_skips_rows_already_present(tmp_path):
    from run_frontier_inference import write_frontier_batch_results

    out_path = tmp_path / "responses.jsonl"
    out_path.write_text(json.dumps({"corpus_row_id": "r1", "model": "claude-sonnet-5-5", "other": "stale"}) + "\n")

    id_map = {"abc": _sample_row("r1")}
    items = [_FakeItem("abc", _FakeSucceededResult("Should not overwrite."))]
    client = _FakeClient(items)

    write_frontier_batch_results(client, "batch-1", id_map, "claude-sonnet-5-5", out_path)

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0] == {"corpus_row_id": "r1", "model": "claude-sonnet-5-5", "other": "stale"}


def test_write_frontier_batch_results_records_empty_content_as_empty_string_without_crashing(tmp_path):
    from run_frontier_inference import write_frontier_batch_results

    out_path = tmp_path / "responses.jsonl"
    id_map = {
        "abc": _sample_row("r1"),  # empty content -- e.g. a refusal with no text block
        "def": _sample_row("r2"),  # comes after the bad item in the result stream
    }
    items = [
        _FakeItem("abc", _FakeSucceededEmptyContentResult()),
        _FakeItem("def", _FakeSucceededResult("A real response after the bad one.")),
    ]
    client = _FakeClient(items)

    write_frontier_batch_results(client, "batch-1", id_map, "claude-sonnet-5-5", out_path)

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["corpus_row_id"] == "r1"
    assert lines[0]["response_text"] == ""
    assert lines[1]["corpus_row_id"] == "r2"
    assert lines[1]["response_text"] == "A real response after the bad one."
