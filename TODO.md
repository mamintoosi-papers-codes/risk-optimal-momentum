# TODO — runs and checks on the RTX 3090 machine

**Repo:** `risk-optimal-momentum`  
**For:** final paper numbers (tables 8–10 and figures)  
**Last local runs here were short CPU quick/smoke only** (2–3 seeds, few hundred steps). Replace table inputs with these full runs before submission.

---

## 1. Paper tables (regression + classification)

```bash
cd /path/to/risk-optimal-momentum
git pull
bash scripts/run_paper_tables.sh full
# equivalent:
#   cd code
#   python -u paper_tables.py --seeds 5 --steps 800
```

**Writes:**

- `out/paper_tables.json`
- `paper/tab_regression.tex`  → manuscript Table **8** (`tab:reg`)
- `paper/tab_classification.tex` → manuscript Table **9** (`tab:clf`)

**Note:** classification table is **clean labels only** (noisy rows are in the noise table). Wine classification is omitted (100% both methods).

---

## 2. Label / target-noise sweep

```bash
cd code
python -u noise_sweep.py --seeds 5 --steps 800
```

**Writes:**

- `out/noise_sweep.json`
- `paper/tab_noise.tex` → manuscript Table **10** (`tab:noise`)
- `paper/fig_noise_sweep.pdf` / `.png` → manuscript Figure (noise)

**Grid:** digits & breast-cancer logistic label noise `{0, 0.1, 0.2, 0.3, 0.5}`; linear regression **wine_quality, diabetes, california_housing** with target noise `{0, 0.5, 1.0}`.

---

## 3. Optional (if you want extra confidence)

| Command | Purpose |
|---------|---------|
| `bash scripts/run_grouping_grid.sh full` | MNIST + CIFAR-10 block/filter/channel (already run once; CNN left out of the paper) |
| `python -u path4_dim_sweep.py` | Dimension sweep (already in `out/`; re-run only if defaults change) |
| `python -u test_sr_adam.py` | After any `sr_adam.py` edit |

---

## 4. After the runs

1. Check that **bold** in Tables 8 and 10 still marks **positive** gaps (generators already do `\bm{...}`).  
2. Recompile `paper/sn-article.tex` (pdflatex + bibtex); fix any overfull tables if needed.  
3. Skim `out/paper_tables.json` / `noise_sweep.json` for mixed cells (e.g. blog linear, iris, breast-cancer high noise) — decide whether to keep as honest mixed results or add a short remark in the text.  
4. `git add` the updated `paper/tab_*.tex`, figures, and `out/*.json`, commit, push.  
5. Update `reports/REPORT_for_Arashi/REPORT.md` if the full run changes the story (e.g. stronger/weaker noise gaps).

---

## 5. Not required for the stats submission

- CIFAR / MNIST CNN grids (`train_cifar.py`, `train_grouping.py`) — kept out of the manuscript; only if a referee asks.  
- `YearPredictionMSD` download (mirrors failed earlier) — optional extra regression set.

---

## 6. Quick local fallback (not for final tables)

```bash
cd code
python -u paper_tables.py --quick
python -u noise_sweep.py --quick
```

These are for smoke tests only.
