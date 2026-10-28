from corpus_gen.pwt import POOL_TIERS, sample_pool


def test_sample_pool_returns_requested_size_with_no_duplicates():
    pool = sample_pool(5, "META-01", 1)
    assert len(pool) == 5
    assert len(set(pool)) == 5
    assert set(pool).issubset(set(POOL_TIERS[30]))


def test_sample_pool_is_deterministic_for_same_inputs():
    first = sample_pool(5, "META-01", 1)
    second = sample_pool(5, "META-01", 1)
    assert first == second


def test_sample_pool_known_value_for_fixed_inputs():
    assert sample_pool(5, "META-01", 1) == ["th", "ko", "tr", "el", "pl"]


def test_sample_pool_size_30_always_returns_full_universe():
    pool_a = sample_pool(30, "META-01", 1)
    pool_b = sample_pool(30, "XSTEST-17", 3)
    assert set(pool_a) == set(POOL_TIERS[30])
    assert set(pool_b) == set(POOL_TIERS[30])


def test_sample_pool_differs_across_item_ids():
    pool_a = sample_pool(5, "META-01", 1)
    pool_b = sample_pool(5, "META-02", 1)
    assert set(pool_a) != set(pool_b)


def test_sample_pool_differs_across_seeds():
    pool_a = sample_pool(5, "META-01", 1)
    pool_b = sample_pool(5, "META-01", 2)
    assert set(pool_a) != set(pool_b)
