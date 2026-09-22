# Empirical findings on the SR-Adam revision
## What was tested, where the method helps, and where it does not

**To:** Prof. M. Arashi  
**From:** Experimental review (with M. Amintoosi)  
**Date:** 21 September 2026  
**Code & results:** `https://github.com/mamintoosi-papers-codes/risk-optimal-momentum`  
**Scope:** Evaluation of the **current** manuscript algorithm (`code/sr_adam.py`) against AdamW / AMSGrad / SGD-M. Theoretical proofs are yours and are not re-derived here.

---

## Abstract (فارسی — کوتاه)

این گزارش **یافته‌های تجربی** الگوریتم فعلی مقاله (`sr_adam.py`) را بدون وابستگی به برچسب‌های داخلی آزمایش‌ها جمع‌بندی می‌کند.

**نقاط قوت:**  
(۱) در **ابعاد بالا** (به‌ویژه $p$ از حدود ۵۰ تا ۵۰۰) و زیر **نویز گرادیان/برچسب**، روش معمولاً از AdamW بهتر است و **مزیت با $p$ و با نویز بیشتر می‌شود** — هم‌سو با مفهوم **High Dimensions** در نظریه.  
(۲) روی **رگرسیون خطیِ داده‌های واقعی** تقریباً همیشه برنده یا هم‌تراز است؛ روی **نمونه‌های کم‌حجم** یا **$d$ نسبتاً بزرگ‌تر** (مثلاً protein با $d\approx100$) برد روشن دارد.

**نقاط ضعف / مرزها:**  
(۱) روی **CNN کوچک CIFAR** از Adam/AdamW/AMSGrad عقب‌تر است؛  
(۲) روی **ریج مینی‌بچِ مصنوعی** آشکارا بدتر است؛  
(۳) روی **MLPهای جدولی میان‌اندازه** گاهی اندکی بدتر است.

**جمع‌بندی:** روش یک **ابزار با رژیم برتری مشخص** (پُربعد / پُر‌نویز / برآوردی) است، نه بهینه‌ساز غالب همه‌جا. جزئیات در ادامه.

---

## 1. Why this note

You have written the revised theory and Algorithm 1. Before the empirical section is filled, this note records:

1. What was run with the **current** implementation  
2. **Where SR-Adam is stronger** than well-known optimizers  
3. **Where it is not** (so the paper can state boundaries honestly)  
4. Practical next steps for the manuscript  

Paired comparison throughout: $\Delta = \mathrm{AdamW} - \mathrm{SR\text{-}Adam}$ on the **minimized** loss. **Positive $\Delta$ = SR-Adam better.**

---

## 2. Implementation

`code/sr_adam.py` matches the revised estimator:

- Search direction $d_t = \hat m_{t-1} + k_t (g_t - \hat m_{t-1})$ with positive-part Stein weight $k_t = [1 - c_t/Q_t]^+$  
- Coordinatewise **debiased** noise scale $(\hat v - \hat m^2) / (1 - \phi_t)$ and optional pooling  
- Moments formed from the **raw** gradient (as in the theory)  
- Effective-dimension gate; AMSGrad preconditioner  
- Unit tests (`test_sr_adam.py`): with the gate shut, updates coincide with AMSGrad-AdamW; displacement bound holds  

---

## 3. Strengths (what works)

### 3.1 High dimensions: the gain grows with $p$

On a noisy quadratic objective (diagonal curvature, additive gradient noise) and on multinomial logistic regression with label noise, dimension was swept over $p \in \{10, 20, 50, 100, 200, 500\}$ (3 seeds).

**Quadratic, Gaussian noise $\tau=1$ (example):**

| $p$ | AdamW | SR-Adam | $\Delta$ |
|----:|------:|--------:|---------:|
| 10 | 0.041 | 0.028 | +0.012 |
| 50 | 0.258 | 0.211 | +0.046 |
| 100 | 0.454 | 0.343 | +0.110 |
| 200 | 0.955 | 0.733 | +0.222 |
| 500 | 2.312 | 1.722 | **+0.590** |

- Student-$t$ noise: $\Delta$ rises from about +0.01 at $p=10$ to about **+0.38** at $p=500$.  
- Logistic with 20% label flips: almost a tie at $p=10$, then $\Delta$ grows to about **+1.44** at $p=500$.  
- **23 of 24** settings in this sweep favour SR-Adam.

This is direct support for the **high-dimensional** motivation of the paper: the Stein-type advantage is small near $p \approx 3$–$10$ and large in high dimensions.

### 3.2 Controlled optimisation under gradient noise

On synthetic problems with controlled noise (noisy quadratic at several $\tau$, Student-$t$ gradients, logistic with label flips, small teacher–student MLP), **24 of 28** settings favour SR-Adam over AdamW.

### 3.3 Real regression data

Several standard sets (diabetes, california housing, wine quality, linnerud, energy efficiency, concrete, year, protein):

| Pattern | Outcome |
|---------|---------|
| **Linear model** on these sets | Essentially **always** SR-Adam $\ge$ AdamW (including **protein**, $n\approx2000$, $d\approx100$) |
| **Small sample** or **moderately higher $d$** (diabetes MLP $n=442$; linnerud; protein MLP) | **Clear wins** (e.g. diabetes MLP $\Delta \approx +0.24$) |
| **Label noise** added to train targets (wine, california, diabetes) | **8 of 9** settings favour SR-Adam; margin **increases with noise** |

### 3.4 Monte Carlo checks of the theory

Risk vs signal-to-noise, bias factor $1-\phi_t(\beta_1)$, heteroscedasticity / effective dimension, Student window: numerics agree with the theorems in the manuscript (already in the simulation section).

---

## 4. Weaknesses and boundaries (what does not work as well)

### 4.1 Small CNN on CIFAR-10/100

Protocol: SimpleCNN, batch 512, 20 epochs, 5 seeds, noise levels 0 / 0.05 / 0.1. **Final-epoch accuracy** (mean over seeds):

| Dataset | Noise | Adam | AdamW | AMSGrad | SR-Adam | SR − Adam |
|---------|------:|-----:|------:|--------:|--------:|----------:|
| CIFAR-10 | 0.0 | 74.1 | 74.0 | 74.4 | 71.9 | −2.3 |
| CIFAR-10 | 0.05 | 74.0 | 74.0 | 74.1 | 71.1 | −3.0 |
| CIFAR-10 | 0.1 | 73.4 | 73.4 | 73.5 | 71.1 | −2.3 |
| CIFAR-100 | 0.0 | 40.7 | 40.9 | 40.8 | 39.7 | −1.0 |
| CIFAR-100 | 0.05 | 40.4 | 40.3 | 40.5 | 39.0 | −1.4 |
| CIFAR-100 | 0.1 | 39.6 | 39.6 | 39.7 | 38.7 | −0.9 |

SR-Adam is **behind** the Adam family on every cell. Diagnostics show the peff gate almost always open and a **very small** Stein weight on the large fully connected layer ($k \approx 0.01$), i.e. the direction is almost purely the historical average.

**Interpretation:** either the defaults **over-shrink** on this architecture, or small-CNN training sits **outside** the favourable regime. This does not contradict the risk theory; it marks a **boundary** of the tool.

### 4.2 Mini-batch ridge (synthetic least squares)

Clear losses (e.g. $p=150$, batch 32: MSE 0.13 (AdamW) vs 0.68 (SR-Adam)).  
Likely reason: when residual noise near the optimum is already small, shrinkage bias costs more than the variance it removes.

### 4.3 Mid-size tabular MLPs

On energy, concrete, and year-like sets, the MLP version is sometimes **slightly** worse than AdamW (near-tie), while the **linear** model on the **same** data favours SR-Adam.

### 4.4 Statistical caveats

- Some gaps are small (near-ties) with 3–5 seeds — treat as “match”, not strong wins.  
- Large-set loaders used subsets (CPU time).  
- All losses here are **optimisation** metrics (MSE / log-loss / terminal objective), not a full generalisation study at scale.

---

## 5. Regime map (summary table)

| Setting | SR-Adam vs AdamW |
|---------|------------------|
| High dimension $p$ (≈50–500), controlled noise | **Better**, gap ↑ with $p$ |
| Heavy-tailed gradient noise | **Better** |
| Label noise (synthetic and on real targets) | **Better**, gap ↑ with noise |
| Real **linear** regression | **Better or equal** |
| Small-$n$ / moderate-$d$ (e.g. protein $d\approx100$, diabetes MLP) | **Clearly better** |
| Mid-size tabular MLP | Near-tie / slight loss |
| Mini-batch ridge (synthetic) | **Worse** |
| Small CNN (CIFAR) | **Worse** (~1–3 accuracy points) |

**Suggested claim for the paper:**

> Replacing the fixed momentum weight by a positive-part Stein weight helps when gradient estimation is noisy and the parameter block is high-dimensional; it is not claimed to improve every architecture. Outside that regime (low-noise ridge-type fits, small CNNs) it can be neutral or worse.

---

## 6. Title and “High Dimensions”

“High Dimensions” is part of the core contribution and matches both the theory ($p \ge 3$ blocks, Stein risk) and the **dimension sweep** in §3.1 (gain grows with $p$). No need to drop it from the title.

Optional wording variants that **keep** the phrase:

1. **Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions** *(current)*  
2. **Risk-Optimal Shrinkage Weights for Noisy Gradients in High Dimensions**  
3. **Decision-Theoretic Momentum Weights for High-Dimensional Stochastic Gradients**

---

## 7. Recommendations for the paper and next work

| # | Action | Why |
|---|--------|-----|
| 1 | Put the **dimension sweep** ($p$ vs $\Delta$) in the main text | Anchors **High Dimensions** with data |
| 2 | Order the experiments: theory Monte Carlo → synthetic noise / dimension → real regression + label noise → short **limitations** (ridge, CIFAR) | Strengths first, honest limits |
| 3 | Report CIFAR as a **negative control / limitation**, not as a success table | Integrity with reviewers |
| 4 | Optional algorithm variants on the **same** code: floor $k_{\min}$, conv-only filter | If a more CNN-friendly default is desired (label as variants) |
| 5 | Extend $p$ to $10^3$–$5\times 10^3$ on logistic/quadratic | Even stronger high-$p$ evidence |
| 6 | Plot mean $k_t$, $\hat\sigma^2$, peff vs $p$ and vs noise | Explains the mechanism |
| 7 | Short theoretical remark on **ridge** (bias of shrinkage when residual noise is small) | Turns a failure into insight |
| 8 | Optional: a few more large public regression sets | External validity |

**Decisions we would like from you:**

| ID | Decision |
|----|----------|
| D1 | Confirm that the **dimension sweep** goes into the paper as main evidence for high dimensions |
| D2 | CIFAR: short limitations subsection vs omit |
| D3 | Keep the current title wording or one of the variants in §6 (all keep “High Dimensions”) |
| D4 | Whether to add a conservative variant ($k_{\min}$ / scope filter) as an ablation |

---

## 8. One-line summary

**In high dimensions and under substantial noise the current SR-Adam beats Adam-type methods (23/24 on the dimension sweep; 24/28 on controlled noise; all linear real-regression cells, including protein $d\approx100$), while mini-batch ridge, some mid-size tabular MLPs, and a small CNN are boundaries — the paper should claim a high-dimensional, noise-robust advantage, not universal dominance.**

---

## Appendix A — How to reproduce (this repo)

```bash
cd code
python -u test_sr_adam.py
python -u phase_diagram.py --profile cpu       # synthetic optimisation grid
python -u path4_dim_sweep.py                   # dimension sweep
python -u path3_uci.py --seeds 5 --steps 800   # real regression (first batch)
python -u path5_more_datasets.py --seeds 3     # real regression (extra sets)
python -u path3b_label_noise.py                # real data + label noise
# CIFAR (GPU): python train_cifar.py --grid main --epochs 20 --seeds 5
```

## Appendix B — Where the numbers live

| Content | Location |
|---------|----------|
| This report | `reports/REPORT_for_Arashi/REPORT.md` |
| Synthetic grid results | `out/phase_diagram.json`, `figs/phase_diagram_*` |
| Dimension sweep | `out/path4_dim_sweep.json`, `figs/path4_dim_sweep.*` |
| Real regression (batch 1) | `out/path3_uci.json`, `figs/path3_uci.*` |
| Real regression (extra sets) | `out/path5_more_datasets.json` |
| Real label noise | `out/path3b_label_noise.json` |
| CIFAR grid (note) | `results/runs/main_20260921-085656/` |
| Algorithm | `code/sr_adam.py` |

Shorter write-ups (EN + FA) for individual study folders remain under `reports/PATH2_phase_diagram/` and `reports/PATH3_uci_regression/`; they can be read independently if useful.
