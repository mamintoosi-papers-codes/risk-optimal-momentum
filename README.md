# Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions

Reference implementation and manuscript for **SR-Adam**: a risk-optimal replacement for the fixed momentum weight in adaptive gradient methods.

Paper title:
*Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions*

Authors: Mohammad Arashi, Mahmood Amintoosi (Ferdowsi University of Mashhad)

Code URL used in the manuscript: `https://github.com/mamintoosi-papers-codes/risk-optimal-momentum`

## What this repository contains

| Path | Contents |
|------|----------|
| `code/` | SR-Adam optimizer, CIFAR driver, Monte Carlo studies, theory checks |
| `figs/` | Simulation figures embedded in the manuscript |
| `out/` | Machine-readable Monte Carlo results used by `make_figures.py` |
| `paper/` | LaTeX sources (Springer Nature `sn-jnl`) and figure PDFs |
| `scripts/` | Helper scripts for the CIFAR experimental grid |
| `data/` | CIFAR download directory (gitignored) |

This repo is the **v1 / revision** codebase for the rewritten paper. The earlier rejected submission (old optimizer pipeline, old CIFAR results, reviewer files) lives in the archived repository:

- `https://github.com/mamintoosi-papers-codes/SR-Adam-v0-rejected` *(or the renamed legacy clone of `SR-Adam`)*

Do **not** mix old `results/` / `runs/` numbers with the tables of this manuscript: the algorithm, baselines (AdamW / AMSGrad), and evaluation protocol (final-epoch accuracy, not best-over-epochs) are different.

## Method in one paragraph

Adam’s bias-corrected first moment can be written as a convex combination
`mhat_t = mhat_{t-1} + w_t (g_t - mhat_{t-1})` with a fixed weight `w_t = (1-beta1)/(1-beta1^t)`.
SR-Adam keeps that form and replaces `w_t` by a positive-part Stein weight
`k_t = [1 - c_t / Q_t]^+` whose constant and statistic are set by the block dimension and the coordinatewise gradient-noise variance. Moments are formed from the **raw** gradients; the variance estimate is **debiased** by the exact EMA factor `1 - phi_t(beta1)` and optionally pooled; an **effective-dimension gate** reverts a block to the Adam weight when the variance estimates look unreliable. The preconditioner is the AMSGrad running maximum. When the gate is closed the update coincides with AMSGrad-flavoured AdamW.

## Installation

```bash
# from the repository root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch CUDA wheels: install the build that matches your driver from https://pytorch.org/get-started/locally/ if the default CPU wheel is not enough. On a Linux + RTX 3090 machine, a CUDA 12.x wheel is appropriate.

## Quick correctness check

```bash
cd code
python test_sr_adam.py
```

Expected: all four tests pass, including exact agreement with AMSGrad-AdamW when the peff gate is forced shut.

## Monte Carlo / theory verification

These scripts need only NumPy / SciPy / Matplotlib (and PyTorch for `simulation_batch.py` / `simulation_noise.py`).

```bash
cd code
python verify_theory.py        # closed-form constants vs Monte Carlo
python verify_theory2.py
python simulation_study.py     # writes ../out/*.json and ../figs/*.pdf
python simulation_study3.py
python simulation_batch.py
python simulation_noise.py
python make_figures.py         # rebuild paper figures from stored ../out JSON
```

Stored outputs in `out/` and `figs/` already match the current manuscript text. Re-run only if you change theory constants or sampling settings.

## CIFAR experimental grid

The manuscript still has red `\torerun{...}` placeholders in the image-classification tables. Fill them by running the grids **on a GPU machine** from `code/`:

```bash
cd code
python train_cifar.py --grid diagnostics --out results.csv
python train_cifar.py --grid scope       --seeds 3 --out results.csv
python train_cifar.py --grid batch       --seeds 3 --out results.csv
python train_cifar.py --grid main        --seeds 5 --out results.csv
```

Or use the helper:

```bash
bash scripts/run_cifar_grids.sh
```

Grids:

| `--grid` | Purpose | Paper tables |
|----------|---------|----------------|
| `main` | optimizers × noise levels, batch 512 | `tab:main_acc`, timings |
| `batch` | batch-size ablation, noise 0.05 | `tab:batch_cifar` |
| `scope` | component ablation | `tab:ablation` |
| `diagnostics` | per-layer peff / gate stats | `tab:arch` gate column |

Notes:

- CIFAR is downloaded to `./data` relative to the working directory (run from `code/`, data lands in `code/data`; symlink or move to repo-root `data/` if you prefer).
- Results append to a CSV (default `results.csv`). Aggregate final-epoch rows per (dataset, noise, optimizer, batch_size, seed) before pasting into the tex.
- Optimizers used in `main`: `sgd`, `momentum`, `adam`, `adamw`, `amsgrad`, `sradam`.
- Ablation variants: `sradam`, `sradam_conv`, `sradam_nogate`, `sradam_peff`, `sradam_nodebias`.
- Full `main` grid is large (2×3×6×5×20 epochs). On one RTX 3090 expect on the order of a day; reduce `--seeds` only for smoke tests, not for paper numbers.

### Filling `\torerun` in the manuscript

1. Run the grids above.
2. Compute **final-epoch** mean ± std over seeds for each cell.
3. Replace every `\torerun{...}` in `paper/sn-article.tex` (search for `torerun`).
4. Recompile `paper/` with `pdflatex` + `bibtex` (Springer Nature `sn-jnl` class is included).

The tex comment block at the head of §Image classification lists the exact commands expected for each table.

## Repository layout details

```
risk-optimal-momentum/
├── README.md
├── requirements.txt
├── .gitignore
├── code/                 # all Python; import paths assume this CWD
│   ├── sr_adam.py        # SRAdam optimizer
│   ├── train_cifar.py    # CIFAR-10/100 grids
│   ├── test_sr_adam.py
│   ├── simulation_*.py
│   ├── verify_theory*.py
│   └── make_figures.py   # reads ../out, writes ../figs
├── figs/                 # simulation PDFs for the paper
├── out/                  # Monte Carlo JSON
├── paper/                # sn-article.tex, sample.bib, class/bst, figure PDFs
├── scripts/
│   └── run_cifar_grids.sh
└── data/                 # CIFAR root (ignored by git)
```

`make_figures.py` and the simulation scripts resolve `../figs` and `../out` relative to `code/`. Keep that folder relationship.

## Citation

If you use this code, please cite the manuscript *Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions*.

```bibtex
@article{arashi2026sradam,
  title   = {Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions},
  author  = {Arashi, Mohammad and Amintoosi, Mahmood},
  journal = {under review},
  year    = {2026},
  note    = {Code: https://github.com/mamintoosi-papers-codes/risk-optimal-momentum}
}
```

## Relationship to the rejected v0 paper

| | v0 (rejected) | this repo (revision) |
|--|---------------|----------------------|
| Title | Stein-Rule Shrinkage for Stochastic Gradient Estimation… | Risk-Optimal Momentum Weights… |
| Optimizer | `SRAdamAdaptiveLocal` in old `src/optimizers.py` | `SRAdam` in `code/sr_adam.py` |
| Variance estimate | raw \(v-m^2\), global, clipped shrink | debiased + pooled, peff gate |
| Moments | updated from shrunk direction | updated from raw gradients |
| Preconditioner | Adam | AMSGrad |
| Baselines | SGD / Momentum / Adam | + AdamW / AMSGrad |
| Metric | best test accuracy over epochs | final-epoch test accuracy |
| Status | rejected; archived | active; CIFAR tables to be filled |

## License / declarations

Research code accompanying a manuscript under review. No competing interests declared by the authors. Add a LICENSE file before making the repository public if the venue requires one.
