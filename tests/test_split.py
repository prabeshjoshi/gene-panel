import numpy as np

from panel import split as sp


def _toy():
    donors = []
    ids = []
    for d in range(40):
        for j in range(1 + d % 2):  # some donors have 2 samples (only for leakage test)
            donors.append(f"GTEX-{d:03d}")
            ids.append(f"GTEX-{d:03d}-S{j}")
    return np.array(ids), np.array(donors)


def test_no_donor_on_both_sides():
    ids, donors = _toy()
    s = sp.donor_split(ids, donors, test_frac=0.2, seed=1)
    tr = {donors[list(ids).index(i)] for i in s["train"]}
    te = {donors[list(ids).index(i)] for i in s["test"]}
    assert tr.isdisjoint(te)
    assert set(s["train"]) | set(s["test"]) == set(ids)


def test_test_donor_count_matches_fraction():
    ids, donors = _toy()  # 40 donors
    s = sp.donor_split(ids, donors, test_frac=0.2, seed=3)
    n_test_donors = len({donors[list(ids).index(i)] for i in s["test"]})
    assert n_test_donors == 8


def test_same_seed_same_split_different_seed_differs():
    ids, donors = _toy()
    a = sp.donor_split(ids, donors, 0.2, seed=7)
    b = sp.donor_split(ids, donors, 0.2, seed=7)
    c = sp.donor_split(ids, donors, 0.2, seed=8)
    assert a == b
    assert a != c
