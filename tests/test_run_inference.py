from run_inference import rows_with_sample_plan


def test_rows_with_sample_plan_requests_only_shortfall():
    rows = [{"corpus_row_id": "r1"}, {"corpus_row_id": "r2"}]
    existing = {"r1": 3}
    plan = rows_with_sample_plan(rows, existing, m_samples=5)

    assert len(plan) == 2
    row1, needed1, start1 = plan[0]
    assert row1["corpus_row_id"] == "r1"
    assert needed1 == 2
    assert start1 == 3

    row2, needed2, start2 = plan[1]
    assert row2["corpus_row_id"] == "r2"
    assert needed2 == 5
    assert start2 == 0


def test_rows_with_sample_plan_excludes_fully_complete_rows():
    rows = [{"corpus_row_id": "r1"}, {"corpus_row_id": "r2"}]
    existing = {"r1": 5, "r2": 2}
    plan = rows_with_sample_plan(rows, existing, m_samples=5)

    assert len(plan) == 1
    assert plan[0][0]["corpus_row_id"] == "r2"


def test_rows_with_sample_plan_empty_when_nothing_needed():
    rows = [{"corpus_row_id": "r1"}]
    existing = {"r1": 5}
    plan = rows_with_sample_plan(rows, existing, m_samples=5)
    assert plan == []
