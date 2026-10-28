import json

from judging.run_judging import load_batch_state, render_user_message, save_batch_state, summarize_stop_reasons, write_batch_results


def test_render_user_message_includes_condition():
    message = render_user_message("Original text.", "Degraded text.", "A response.", "word_salad")
    assert "condition:\nword_salad" in message
    assert "original_prompt:\nOriginal text." in message
    assert "degraded_prompt:\nDegraded text." in message
    assert "response:\nA response." in message


class _FakeErrorResult:
    type = "errored"
    error = "boom"


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


def test_write_batch_results_skips_rows_already_in_output_file(tmp_path):
    out_path = tmp_path / "judged.jsonl"
    out_path.write_text(json.dumps({"response_id": "r1", "call_type": "outcome", "other": "stale"}) + "\n")

    id_map = {"abc": ["r1", "outcome"], "def": ["r2", "outcome"]}
    items = [
        _FakeItem("abc", _FakeErrorResult()),  # r1/outcome already in file -- must be skipped
        _FakeItem("def", _FakeErrorResult()),  # r2/outcome is new -- must be written
    ]
    client = _FakeClient(items)

    write_batch_results(client, "batch-1", id_map, out_path)

    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0] == {"response_id": "r1", "call_type": "outcome", "other": "stale"}
    assert lines[1]["response_id"] == "r2"


def test_write_batch_results_is_idempotent_if_run_twice(tmp_path):
    out_path = tmp_path / "judged.jsonl"
    id_map = {"abc": ["r1", "outcome"]}
    client = _FakeClient([_FakeItem("abc", _FakeErrorResult())])

    write_batch_results(client, "batch-1", id_map, out_path)
    write_batch_results(client, "batch-1", id_map, out_path)  # simulates a resumed/re-run batch

    lines = out_path.read_text().splitlines()
    assert len(lines) == 1


def test_save_batch_state_writes_atomically_no_stray_temp_file(tmp_path, monkeypatch):
    import judging.run_judging as rj

    state_path = tmp_path / "batch_state.json"
    monkeypatch.setattr(rj, "BATCH_STATE_PATH", state_path)

    save_batch_state(["b1", "b2"], {"abc": ["r1", "outcome"]})

    assert json.loads(state_path.read_text()) == {"batch_ids": ["b1", "b2"], "id_map": {"abc": ["r1", "outcome"]}}
    leftover = [p for p in tmp_path.iterdir() if p != state_path]
    assert leftover == []

    batch_ids, id_map = load_batch_state()
    assert batch_ids == ["b1", "b2"]
    assert id_map == {"abc": ["r1", "outcome"]}


def test_summarize_stop_reasons_counts_each_value(tmp_path):
    path = tmp_path / "judged.jsonl"
    rows = [
        {"stop_reason": "tool_use"},
        {"stop_reason": "tool_use"},
        {"stop_reason": "max_tokens"},
        {"stop_reason": "refusal"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    counts = summarize_stop_reasons(path)

    assert counts["tool_use"] == 2
    assert counts["max_tokens"] == 1
    assert counts["refusal"] == 1


def test_summarize_stop_reasons_handles_missing_file(tmp_path):
    counts = summarize_stop_reasons(tmp_path / "does_not_exist.jsonl")
    assert len(counts) == 0
