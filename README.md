# gene-panel pilot

Does choosing a k-gene panel cleverly matter for predicting all other genes, and do common
heuristics miss the true optimum? Small, fast runs on one dataset (GTEx v10, Muscle - Skeletal),
one phase at a time, each ending in a go/stop gate set in `config.yaml`.

## Run

Python 3.11 with numpy, pandas, scipy, scikit-learn, matplotlib, pyarrow and pytest
(`pip install -r requirements.txt`).

```bash
# 1. download the raw file into data/raw/ (URL and md5 in data/raw/SOURCES.md and config.yaml)
curl -sSL -o data/raw/gene_reads_v10_muscle_skeletal.gct.gz \
  https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/counts-by-tissue/gene_reads_v10_muscle_skeletal.gct.gz
# 2. tests, then each phase in order
python -m pytest -q tests
python scripts/00_data.py      # parse, filter, normalise, splits  -> data/processed/, results/setup/
python scripts/01_lowrank.py   # Phase 0                           -> results/phase0/
python scripts/02_headroom.py  # Phase 1                           -> results/phase1/
```

Data is never committed. Every run writes `results/<phase>/log.json` (config, seeds, timings,
outputs).

## Layout

| Path | Contents |
|------|----------|
| `config.yaml` | every threshold, seed and gate (flat `key: value`) |
| `src/panel/io.py` | config loader, GCT parser, JSON writer |
| `src/panel/preprocess.py` | library size, CPM, log2(CPM+1), MT / duplicate / expression filters, donor dedup, QC flags |
| `src/panel/split.py` | donor-level train/test splits |
| `src/panel/select.py` | per-split prep (training-variance top genes, training centring/scaling), random panels |
| `src/panel/evaluate.py` | R2_all, Q2_nonpanel, PCA reconstruction, ridge predictor, alpha CV |
| `src/panel/plots.py` | figures |

## Leakage rules

Gene ranking, centring/scaling, panel selection and models are fitted on training samples only;
test samples are transformed with training parameters. The gene filters in `00_data.py`
(MT, duplicate symbols, CPM >= 1 in >= 20% of samples) use all samples; they are unsupervised
and fit no model.

## Results so far

| Phase | Gate | Value | Verdict |
|-------|------|-------|---------|
| 0 low rank | mean test R2_all at q = 20 >= 0.5 | 0.694 (SD 0.003) | GO |
| 1 headroom | (ceiling - random median) at k = 20 >= 3 x SD across seeds | 0.177 vs SD 0.0054 (33x) | GO |

Data after filtering: 818 samples (one per donor) x 14,424 genes; 654 train / 164 test per seed;
top 1000 genes by training variance per split. Ridge alpha = 10 (5-fold CV, seed 1 training data).

Caveat: the Phase 1 ceiling is PCA reconstruction, which uses every test gene, so no k-gene panel
can reach it. The best achievable panel lies between the best random panel and the ceiling.
