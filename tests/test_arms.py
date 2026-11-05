import numpy as np
import pandas as pd
import pytest

from panel import arms


def test_sex_calls_rule_and_ambiguity():
    X = pd.DataFrame({"Y": [8.0, 0.5, 1.0, 6.0, 4.0],
                      "X": [0.2, 9.0, 0.5, 7.0, 0.1]}, index=list("abcde"))
    out = arms.sex_calls(X, "Y", "X", thr=3.0)
    assert out["sex"].tolist() == ["male", "female", "male", "female", "male"]
    assert out["ambiguous"].tolist() == [False, False, True, True, False]
    assert out["ambiguous_reason"].tolist() == ["", "", "both_low", "both_high", ""]


def test_filter_split_drops_excluded_keeps_order():
    s = {"train": ["a", "b", "c"], "test": ["d", "e"]}
    assert arms.filter_split(s, ["a", "c", "e"]) == {"train": ["a", "c"], "test": ["e"]}


def test_sex_residualize_uses_training_coefficients():
    rng = np.random.default_rng(0)
    m_tr = np.array([True] * 30 + [False] * 30)
    m_te = np.array([True, False])
    Xtr = rng.normal(size=(60, 3)) + np.outer(m_tr, [5.0, 0.0, -2.0])
    Xte = np.array([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    Rtr, Rte, a, b = arms.sex_residualize(Xtr, Xte, m_tr, m_te)
    np.testing.assert_allclose(Rtr[m_tr].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Rtr[~m_tr].mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Rte[0], Xte[0] - a - b)   # male test row: training a + b
    np.testing.assert_allclose(Rte[1], Xte[1] - a)       # female test row: training a


def test_sex_residualize_needs_both_sexes():
    with pytest.raises(ValueError):
        arms.sex_residualize(np.zeros((4, 2)), np.zeros((1, 2)), [True] * 4, [True])


def _toy(n=80, seed=1):
    rng = np.random.default_rng(seed)
    male = np.arange(n) % 2 == 0
    X = rng.normal(size=(n, 6))
    X[:, 0] += 10 * male          # variance only from sex
    X[:, 1] *= 3                  # real non-sex variance
    ids = [f"s{i}" for i in range(n)]
    df = pd.DataFrame(X, index=ids, columns=[f"g{j}" for j in range(6)])
    split = {"train": ids[:60], "test": ids[60:]}
    return df, split, pd.Series(male, index=ids)


def test_arm_b_drops_a_gene_whose_variance_is_only_sex():
    df, split, male = _toy()
    a = arms.arm_a(df, split, 1)
    b = arms.arm_b(df, split, male, 1)
    assert a.gene_ids.tolist() == ["g0"]   # pilot pipeline picks the sex gene
    assert b.gene_ids.tolist() == ["g1"]   # sex-adjusted arm does not
    np.testing.assert_allclose(b.Ztr.mean(0), 0, atol=1e-12)


def test_arm_c_blocks_are_disjoint_complete_and_reproducible():
    df, split, male = _toy()
    b = arms.arm_b(df, split, male, 6)
    c1 = arms.arm_c(b, seed=3, candidate_fraction=0.5, stream=11)
    c2 = arms.arm_c(b, seed=3, candidate_fraction=0.5, stream=11)
    assert np.array_equal(c1.cand, c2.cand)
    assert set(c1.cand).isdisjoint(c1.targ)
    assert sorted(set(c1.cand) | set(c1.targ)) == list(range(6))
    assert len(c1.cand) == 3 and len(c1.targ) == 3
    assert c1.Ztr is b.Ztr               # same data as Arm B, only the gene roles differ
