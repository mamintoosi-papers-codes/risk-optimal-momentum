# Main grid report — SR-Adam vs baselines (CIFAR)

**Run folder:** `results/runs/main_20260921-085656/`  
**Date of analysis:** 2026-09-21  
**Prepared for:** M. Arashi and M. Amintoosi (revision of the SR-Adam / risk-optimal momentum paper)  
**Related code:** `risk-optimal-momentum/code/train_cifar.py`  
**Metric:** final-epoch test accuracy (paper protocol), mean ± std over seeds  

---

## Executive summary (read this first)

1. The **experiment is clean and complete**: 180/180 runs, both datasets, all six optimizers, five seeds, 20 epochs, ~5.1 h on RTX 3090. No missing cells, no duplicate rows, training is numerically healthy.
2. The **main empirical claim does not hold** for the current algorithm under this protocol.
3. **SR-Adam loses to Adam, AdamW, and AMSGrad on every cell** (CIFAR-10/100 × noise {0, 0.05, 0.1}), by about **0.9–3.0** percentage points of final test accuracy.
4. These numbers **must not** be inserted into `paper/sn-article.tex` `\torerun{...}` cells as evidence that SR-Adam improves optimization.
5. Recommended next step: **discuss algorithm defaults with Dr. Arashi**, change the implementation if needed, then re-run `main`. Do not spend more GPU hours on `batch`/`scope` until the method beats (or honestly matches) Adam-family baselines on a smoke `main` grid.

**خلاصه فارسی:** اجرای `main` کامل و قابل اعتماد است، ولی SR-Adam در همهٔ سناریوها از Adam/AdamW/AMSGrad ضعیف‌تر است. این نتایج را نباید به‌عنوان برد روش در مقاله وارد کرد؛ اول باید الگوریتم/پیش‌فرض‌ها با دکتر آرشی بازبینی شود و سپس دوباره اجرا شود.

---

## 1. What was run

| Field | Value |
|--------|--------|
| Command | `train_cifar.py --grid main --epochs 20 --seeds 5` |
| Started | 2026-09-21 08:56:56 |
| Device | CUDA — NVIDIA GeForce RTX 3090 |
| Torch | 2.13.0+cu130 |
| Working dir | `.../risk-optimal-momentum/code` |
| CSV | `results.csv` in this folder (3600 rows) |
| Meta | `run_meta.json` in this folder |

**Grid definition (`main`):**

- Datasets: CIFAR-10, CIFAR-100  
- Input noise σ ∈ {0.0, 0.05, 0.1} (Gaussian, after normalisation)  
- Optimizers: `sgd`, `momentum`, `adam`, `adamw`, `amsgrad`, `sradam`  
- Batch size: 512  
- Seeds: 0–4  
- Epochs: 20  
- Backbone: SimpleCNN (as in the manuscript)  
- Only the optimizer changes across arms  

**`sradam` defaults in code** (`sr_adam.SRAdam` via `make_optimizer`):

- `rho=1.0`, `omega=0.1`, `constant="p"`, `peff_margin=1.0`, `warmup=5`, `amsgrad=True`  
- Default eligibility: every block with ≥3 parameters; peff gate; **no** conv-only filter  
- `weight_decay=0` for SGD / Momentum / Adam / AMSGrad / SR-Adam; **AdamW uses `weight_decay=1e-2`** (PyTorch-style default in the driver)

---

## 2. Data quality

| Check | Result |
|--------|--------|
| Expected runs | 2 × 3 × 6 × 5 = 180 |
| Complete runs (epochs 1…20 unique) | **180 / 180** |
| Missing configurations | **None** |
| Duplicate `(dataset, noise, opt, seed, epoch)` rows | **None** |
| Non-finite / exploding loss | **None observed** |
| Wall-clock (sum of `epoch_time`) | **~5.09 hours** |
| Analysis script | `_analyze_main.py` (same folder) |

Baselines behave as expected: SGD ~13–49% (CIFAR-100/10), Momentum in the middle, Adam-family clustered at the top. That is a sanity check that the training loop and evaluation are not broken.

---

## 3. Final-epoch test accuracy (paper metric)

Mean ± std over 5 seeds. Higher is better.

### CIFAR-10

| Noise | SGD | Momentum | Adam | AdamW | AMSGrad | **SR-Adam** |
|------:|----:|---------:|-----:|------:|--------:|------------:|
| 0.0 | 48.45±1.21 | 72.06±0.60 | 74.14±0.22 | 74.00±0.17 | 74.36±0.34 | **71.85±0.53** |
| 0.05 | 48.62±0.89 | 71.59±0.48 | 74.03±0.60 | 73.99±0.63 | 74.09±0.70 | **71.07±1.18** |
| 0.1 | 48.56±0.97 | 71.14±0.31 | 73.39±0.55 | 73.41±0.87 | 73.47±0.47 | **71.07±0.63** |

### CIFAR-100

| Noise | SGD | Momentum | Adam | AdamW | AMSGrad | **SR-Adam** |
|------:|----:|---------:|-----:|------:|--------:|------------:|
| 0.0 | 13.65±0.48 | 35.43±1.08 | 40.74±0.87 | 40.85±0.54 | 40.78±0.53 | **39.71±0.47** |
| 0.05 | 13.63±0.45 | 35.21±1.17 | 40.43±0.85 | 40.31±0.89 | 40.49±0.86 | **39.01±1.02** |
| 0.1 | 13.66±0.41 | 34.71±1.16 | 39.63±0.77 | 39.64±0.66 | 39.65±0.91 | **38.74±1.16** |

### SR-Adam minus Adam (final-epoch accuracy)

| Dataset | Noise | Δ (SR − Adam) | Reading |
|---------|------:|--------------:|---------|
| CIFAR-10 | 0.0 | **−2.29** | Adam better |
| CIFAR-10 | 0.05 | **−2.96** | Adam better |
| CIFAR-10 | 0.1 | **−2.32** | Adam better |
| CIFAR-100 | 0.0 | **−1.03** | Adam better |
| CIFAR-100 | 0.05 | **−1.42** | Adam better |
| CIFAR-100 | 0.1 | **−0.89** | Adam better |

**Ranking (every cell, same order):**

`AMSGrad ≈ Adam ≈ AdamW > Momentum > SR-Adam ≫ SGD`

Best-over-epochs accuracy (not the paper metric) gives the **same** sign: SR-Adam still below Adam by roughly the same margins.

Adam and SR-Adam both use `weight_decay=0` here, so the SR-Adam deficit vs Adam is **not** explained by weight-decay asymmetry. AdamW’s `wd=1e-2` does not change the ranking vs SR-Adam either.

---

## 4. Do these results match the paper?

| Expectation (revision / manuscript) | Observation on this grid |
|-------------------------------------|---------------------------|
| SR-Adam improves stochastic gradient estimation and should help optimization | SR-Adam is **below** Adam-family on all six cells |
| Gains more visible at higher gradient/input noise | No gain at σ=0.05 or 0.1 on CIFAR-10; still clearly below Adam |
| Large-batch regime (B=512) favourable | Method is **behind** at B=512 |
| Fair modern baselines (AdamW, AMSGrad) — Reviewer B5 | Baselines included; SR-Adam loses to both |
| Rejected v0 paper reported +1.5–2 pts vs Adam (best-acc, old algorithm) | New algorithm is **~2–3 pts worse** on CIFAR-10 (final-acc) |
| Monte Carlo studies show lower estimation risk | Risk theory may still hold; **does not transfer to better CIFAR accuracy** with current defaults |
| Ablation: peff gate replaces conv-only rule | Partial `scope` run (separate folder) showed **`sradam_conv` > full `sradam`** |

**Conclusion for coauthors:**  
Under the paper’s own experimental protocol, the **current SR-Adam implementation does not support the central empirical claim**. Theory sections may remain publishable as decision-theoretic analysis, but **image-classification tables cannot be written as “SR-Adam wins.”**

---

## 5. Related evidence (same project, other folders)

### 5.1 Partial `scope` ablation  
Folder: `results/runs/scope_partial_20260921_manual/`  
CIFAR-10 only; final-epoch acc (approx., some cells have re-run duplicates):

| Noise | `sradam` (gate, all blocks) | `sradam_conv` (conv-only) |
|------:|----------------------------:|--------------------------:|
| 0.0 | ~71.6 | **~72.9** |
| 0.05 | ~71.1 | **~72.2** |
| 0.1 | ~69.7 | **~72.0** |

Restricting shrinkage to convolutional layers **beats** the default all-block peff-gate variant — opposite of the manuscript’s narrative that the gate replaces the layer-type rule.

### 5.2 Diagnostics (`diagnostics.json`, same partial run)

- Gate open ≈ **99–100%** on every layer (including 10-parameter `fc2.bias`).  
- `fc1.weight` (largest block): mean shrinkage weight **k ≈ 0.013** → direction is almost entirely the historical average, very little current gradient.  
- Peff is large on big layers (so the gate stays open), while small layers still pass the threshold `2+γ=3`.  
- Interpretation: the default rule **does not turn shrinkage off** where it may be harmful; it applies **strong** Stein weights almost everywhere.

This is consistent with `main` results: over-shrinkage / mis-calibrated `k` can hurt optimization accuracy even if conditional quadratic risk of the gradient estimate improves in a stylized Gaussian model.

---

## 6. What we should tell ourselves (and Dr. Arashi)

### 6.1 What is solid

- Experimental **infrastructure** works: download, training, logging, run folders, metadata.  
- Grid `main` is **paper-grade complete** if the method were winning.  
- Theory verification / Monte Carlo assets in `out/` and `figs/` remain valid as **theory** artifacts.  
- Unit tests (`code/test_sr_adam.py`): gate-shut SR-Adam matches AMSGrad-AdamW exactly; displacement bound holds.

### 6.2 What is not solid

- Empirical claim that SR-Adam **outperforms Adam-family** on CIFAR with this implementation.  
- Claim that the **effective-dimension gate** is the right scalable substitute for conv-only selection (partial ablation + diagnostics disagree).  
- Any plan to fill `\torerun` cells with these `main` numbers as positive results.

### 6.3 Likely technical reasons (hypotheses — not proven)

1. **Over-shrinkage:** `k_t` near 0 on large FC parameters; noisy-batch gradient information is discarded too often.  
2. **Gate too permissive:** threshold `peff > 3` almost never closes the gate on this CNN.  
3. **Algorithm ≠ v0:** old code used Adam-whitened global σ², clip `[0.1,1]`, moments from the **shrunk** direction, conv-only groups. New code is a different estimator; old positive numbers may not transfer.  
4. **Protocol:** 20 epochs, SimpleCNN, B=512 — may be a regime where Adam-family is already strong and extra bias hurts.  
5. **Preconditioner + shrinkage interaction:** AMSGrad max + Stein weight on raw moments is covered by theory’s floor story, but the floor may be large enough to cost 1–3 accuracy points here.  
6. **Not a logging bug:** rankings are internally consistent; SGD/Momentum/Adam look normal.

### 6.4 Decisions needed (for Dr. Arashi)

| # | Decision | Options (examples) |
|---|----------|---------------------|
| D1 | Keep claiming CIFAR wins, or reframe the paper? | (a) Fix algorithm until wins; (b) paper as theory + honest negative/neutral CIFAR; (c) different task where risk reduction helps (high noise, large batch, different arch) |
| D2 | Default shrinkage scope | conv-only; only blocks with `numel ≥ N`; peff threshold much higher; all-blocks as now |
| D3 | Cap / floor on `k_t` | e.g. `k ∈ [k_min, 1]` with `k_min ∈ [0.2, 0.5]` (closer to `1−β1=0.1` or old clip 0.1) |
| D4 | Safety factor `ρ` / constant | `ρ<1` or `constant="peff"` as **default** if that wins |
| D5 | Fairness protocol | Same `weight_decay` for all adaptive methods; report final-acc (keep) + optional best-acc |
| D6 | Re-run cost | After a change, first `main --seeds 3 --epochs 20` smoke on CIFAR-10 only if needed; full grid only if smoke looks good |

### 6.5 Suggested immediate actions

1. **Share this folder** with Dr. Arashi (`REPORT.md` + `results.csv` + `run_meta.json`).  
2. **Do not edit** `paper/sn-article.tex` empirical tables from this run as positive results.  
3. Agree on **one** algorithm change set (D2–D4), implement in `code/sr_adam.py` / `train_cifar.py`, add a unit test if defaults change.  
4. Re-run:  
   `python train_cifar.py --grid main --epochs 20 --seeds 5`  
   (new folder under `results/runs/`).  
5. If still negative, either change research claim (D1b/c) or the method more deeply — do not keep re-running the same code.  
6. `batch` and `scope` grids: **optional after** `main` is acceptable; they are not the bottleneck now.

---

## 7. How to regenerate this analysis

On a machine with the repo:

```bash
cd risk-optimal-momentum/results/runs/main_20260921-085656
python _analyze_main.py
```

Or aggregate yourself from `results.csv`:

- Keep rows with `epoch == 20` for final-acc.  
- Group by `(dataset, noise, optimizer)` → mean, std over seeds.  
- Compare `sradam` to `adam` / `adamw` / `amsgrad`.

---

## 8. File inventory (this run)

| File | Role |
|------|------|
| `results.csv` | Full epoch-level metrics (source of truth) |
| `run_meta.json` | Command, device, torch, timestamps |
| `_analyze_main.py` | Script that produced the tables above |
| `REPORT.md` | This report |

---

## 9. One-line takeaway

**The main grid ran correctly; SR-Adam currently loses to Adam-family on CIFAR-10/100 under the paper protocol. Treat this as an algorithm/calibration problem to solve with Dr. Arashi — not as numbers to paste into the revision tables.**
