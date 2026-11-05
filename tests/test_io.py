import gzip

import numpy as np
import pytest

from panel import io as pio


def test_strip_version_removes_suffix_only():
    out = pio.strip_version(["ENSG00000223972.5", "ENSG00000227232", "ENSG1.12"])
    assert list(out) == ["ENSG00000223972", "ENSG00000227232", "ENSG1"]


def test_donor_from_sample_takes_first_two_fields():
    assert pio.donor_from_sample("GTEX-1117F-0426-SM-5EGHI") == "GTEX-1117F"
    assert pio.donor_from_sample("GTEX-111CU") == "GTEX-111CU"


def _write_gct(path, rows, samples):
    header = ["Name", "Description"] + samples
    lines = ["#1.2", f"{len(rows)}\t{len(samples)}", "\t".join(header)]
    for name, desc, vals in rows:
        lines.append("\t".join([name, desc] + [str(v) for v in vals]))
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def test_read_gct_parses_counts_and_meta(tmp_path):
    p = tmp_path / "tiny.gct.gz"
    rows = [
        ("ENSG1.3", "GENEA", [10, 0, 5]),
        ("ENSG2.1", "MT-CO1", [7, 7, 7]),
    ]
    _write_gct(p, rows, ["GTEX-AAA-1", "GTEX-BBB-1", "GTEX-CCC-1"])
    counts, meta = pio.read_gct(p)
    assert list(counts.index) == ["ENSG1", "ENSG2"]
    assert list(counts.columns) == ["GTEX-AAA-1", "GTEX-BBB-1", "GTEX-CCC-1"]
    assert counts.dtypes.eq(np.int64).all()
    assert counts.loc["ENSG1", "GTEX-AAA-1"] == 10
    assert meta.loc["ENSG1", "symbol"] == "GENEA"
    assert meta.loc["ENSG1", "gene_name_versioned"] == "ENSG1.3"


def test_read_gct_rejects_bad_version_line(tmp_path):
    p = tmp_path / "bad.gct.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("#1.3\n1\t3\nName\tDescription\tS1\nE1\tG\t1\n")
    with pytest.raises(ValueError):
        pio.read_gct(p)


def test_read_gct_rejects_shape_mismatch(tmp_path):
    p = tmp_path / "shape.gct.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("#1.2\n5\t1\nName\tDescription\tS1\nE1\tG\t1\n")
    with pytest.raises(ValueError):
        pio.read_gct(p)
