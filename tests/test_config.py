import pytest

from panel import io as pio


def test_parses_project_config():
    cfg = pio.load_config()
    assert cfg["tissue"] == "Muscle - Skeletal"
    assert cfg["gct_url"].startswith("https://storage.googleapis.com/")
    assert cfg["min_frac_samples"] == 0.20
    assert cfg["seeds_gates"] == [1, 2, 3, 4, 5]
    assert cfg["protein_coding_filter"] is False
    assert cfg["ridge_alphas"][0] == 0.01


def test_parses_types_and_comments(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "# comment\n"
        'name: "a # b: c"   # trailing\n'
        "n: 3\n"
        "x: 1e-3  # sci\n"
        "flag: true\n"
        "xs: [1, 2.5, 3]\n"
        "bare: word\n"
    )
    cfg = pio.load_config(p)
    assert cfg == {"name": "a # b: c", "n": 3, "x": 0.001, "flag": True,
                   "xs": [1, 2.5, 3], "bare": "word"}


def test_rejects_duplicate_keys(tmp_path):
    p = tmp_path / "d.yaml"
    p.write_text("a: 1\na: 2\n")
    with pytest.raises(ValueError):
        pio.load_config(p)
