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
python scripts/03_heuristics.py # Phase 2                          -> results/phase2/
python scripts/04_toy_exact.py  # Phase 3                          -> results/phase3/
python scripts/05_stability.py  # Phase 4                          -> results/phase4/
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
| 2 heuristics | at k = 20, best beats random median AND >= 1 method pair differs, beyond paired SD | greedy +0.080 (SD 0.005); 10/10 pairs differ | GO |
| 3 toy exact | k >= 4: greedy misses the optimum in >= 20% of pools AND mean test cost of misses > 2 x bootstrap SE, on seeds 1-3 | misses 72-85%; cost +0.005 to +0.009 R2 (2.6-4.1 x SE) | GO |
| 4 stability | k = 20 over 10 seeds: per-seed ranking equals consensus in more than half of seeds | 9/10 (only mtl/pls_vip swap once) | GO |

Phase 2, test R2_all at k = 20 (random median 0.518, ceiling 0.694): greedy 0.597, clustering
(complete linkage) 0.562, MultiTaskLasso 0.529, PLS2-VIP 0.506, leverage 0.323. Leverage is far
below random because it concentrates on one tight module (11 of 20 genes are sex-chromosome genes).
Average-linkage clustering (first run) gave 0.331: it collapses into one ~936-gene cluster plus
outliers, so the medoids are outlier genes; `cluster_linkage` is therefore "complete".

Phase 3 (40-gene pools, all 1000 genes as targets, exact optimum by exhaustive search): greedy
finds the exact optimum in 15-80% of pools, less often as k grows, but when it misses its panel
is still near the top (median rank 2-19 of up to 658,008 subsets) and the test cost is small
(about 0.005-0.009 R2, roughly the size of the seed-to-seed noise in Phase 2). In 5-12 of ~30
misses per seed the greedy panel is better on test. Leverage never and clustering rarely hit the
optimum.

Phase 4 (k = 20, seeds 1-10): greedy > clustering > MultiTaskLasso > PLS2-VIP > leverage
(0.595, 0.559, 0.526, 0.492, 0.316; random median 0.513). Greedy is first and clustering second in
every seed. Panel stability runs opposite to performance: leverage and PLS2-VIP pick nearly the
same genes every seed (Jaccard 0.86-0.89) but predict worst; greedy (0.28) and clustering (0.08)
change genes between seeds, but swap in correlated stand-ins (soft overlap 0.81 and 0.73 vs 0.55
for random). Swapping ridge for PLS2 changes nothing: inner CV picks all 20 LVs (= OLS).

Data after filtering: 818 samples (one per donor) x 14,424 genes; 654 train / 164 test per seed;
top 1000 genes by training variance per split. Ridge alpha = 10 (5-fold CV, seed 1 training data).

Caveat: the Phase 1 ceiling is PCA reconstruction, which uses every test gene, so no k-gene panel
can reach it. The best achievable panel lies between the best random panel and the ceiling.
