"""Donor-level train/test splits. All samples of one donor go to the same side."""
from __future__ import annotations

import numpy as np


def donor_split(sample_ids, donors, test_frac: float, seed: int) -> dict:
    """Split samples into train/test by donor.

    The number of test donors is round(test_frac * n_donors). The permutation comes from
    numpy's default_rng(seed), so a seed always gives the same split.
    """
    sample_ids = np.asarray(sample_ids)
    donors = np.asarray(donors)
    unique = np.array(sorted(set(donors.tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_test = int(round(test_frac * len(unique)))
    test_donors = set(perm[:n_test].tolist())
    is_test = np.array([d in test_donors for d in donors.tolist()], dtype=bool)
    return {
        "train": sample_ids[~is_test].tolist(),
        "test": sample_ids[is_test].tolist(),
    }
