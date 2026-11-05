import numpy as np
import pandas as pd
import pytest

from panel import preprocess as pp


def test_cpm_columns_sum_to_one_million():
    counts = pd.DataFrame({"s1": [1, 3, 6], "s2": [10, 0, 10]}, index=["a", "b", "c"])
    lib = pp.library_sizes(counts)
    cpm = pp.to_cpm(counts, lib)
    np.testing.assert_allclose(cpm.sum(axis=0).to_numpy(), [1e6, 1e6])
    assert cpm.loc["a", "s1"] == pytest.approx(1e5)


def test_log2cpm_is_log2_of_cpm_plus_one():
    cpm = np.array([0.0, 1.0, 3.0])
    np.testing.assert_allclose(pp.log2cpm(cpm), [0.0, 1.0, 2.0])


def test_mt_mask_drops_only_prefix():
    sym = pd.Series(["MT-CO1", "MTND1", "GAPDH", "MT-ND4L"], index=["g1", "g2", "g3", "g4"])
    keep = pp.mt_keep_mask(sym, "MT-")
    assert keep.tolist() == [False, True, True, False]


def test_dedupe_keeps_higher_score_per_symbol():
    sym = pd.Series(["TP53", "TP53", "BRCA1", ""], index=["g1", "g2", "g3", "g4"])
    score = pd.Series([1.0, 5.0, 2.0, 0.0], index=sym.index)
    keep = pp.dedupe_symbol_keep_mask(sym, score)
    assert keep.tolist() == [False, True, True, True]


def test_expression_filter_thresholds():
    cpm = pd.DataFrame(
        {"s1": [2.0, 0.5, 2.0], "s2": [2.0, 0.5, 0.0], "s3": [0.0, 0.5, 0.0], "s4": [0.0, 0.5, 0.0],
         "s5": [0.0, 0.5, 0.0]},
        index=["hi", "lo", "mid"],
    )
    # hi: 2/5 = 0.4 >= 0.2 ; lo: 0/5 ; mid: 1/5 = 0.2 >= 0.2 (boundary kept)
    keep = pp.expression_keep_mask(cpm, min_cpm=1.0, min_frac=0.2)
    assert keep.tolist() == [True, False, True]


def test_donor_keep_keeps_one_per_donor_largest_library():
    sids = ["GTEX-A-1", "GTEX-A-2", "GTEX-B-1", "GTEX-C-1"]
    donors = ["GTEX-A", "GTEX-A", "GTEX-B", "GTEX-C"]
    lib = np.array([100, 300, 50, 70])
    mask = pp.donor_keep_mask(sids, donors, lib, "max_library_size")
    assert mask.tolist() == [False, True, True, True]


def test_donor_keep_rejects_unknown_rule():
    with pytest.raises(ValueError):
        pp.donor_keep_mask(["a"], ["d"], [1], "first")


def test_library_qc_flags_outlier_only():
    lib = pd.Series([1e6, 1.1e6, 0.9e6, 1.05e6, 0.95e6, 1e6, 1e4], index=list("abcdefg"))
    flags = pp.library_qc_flags(lib, threshold=3.0)
    assert flags.tolist() == [False] * 6 + [True]
