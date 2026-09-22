# Path 2 report — phase diagram (synthetic optimization)

**Date:** 2026-09-21  
**Code:** `code/phase_diagram.py --profile cpu`  
**Env:** `C:\programs\anaconda3\envs\pth` (torch 2.3.1, CPU)  
**Data:** **No named datasets.** Synthetic DGPs only (quadratic, ridge, logistic, teacher–student MLP).  
**Related:** Path-2 plan in session notes; CIFAR `main` grid REPORT is separate (`results/runs/main_20260921-085656/REPORT.md`).

---

## Verdict

**Path 2 is good.** On theory-aligned synthetic problems, **SR-Adam beats AdamW in 24/28 settings**.

| Block | Wins (SR-Adam better terminal objective / test loss) |
|-------|-----------------------------------------------------|
| Noisy quadratic, Gaussian, $p\in\{50,150,300\}$, $\tau\in\{0.1..4\}$ | **15/15** |
| Noisy quadratic, Student-$t$, $p=150$ | **5/5** |
| Logistic + label flips $\{0,0.1,0.3\}$ | **3/3** |
| Small MLP regressor | **1/1** |
| **Ridge / linear regression, mini-batch** | **0/4** ← only failure family |

Paired metric: $d = \mathrm{AdamW} - \mathrm{SR-Adam}$ on the minimized loss (positive = SR better).

---

## What this says about the theory

| Claim | Evidence here |
|-------|----------------|
| Risk-optimal Stein weight can help optimization under controlled gradient noise | **Supported** on quadratic (all $p,\tau$) |
| Helps under heavy-tailed / elliptical noise | **Supported** (Student-$t$ all $\tau$) |
| Helps when labels are noisy (estimation-heavy classification) | **Supported** (logistic flips) |
| Helps on mild non-convex regression | **Supported** (small MLP) |
| Always helps on every regression problem | **False** — **ridge with mini-batch gradients loses clearly** |
| Explains CIFAR CNN wins | **Not supported** (see main-grid REPORT); consistent with “depends on problem structure” |

So the right empirical story is **regime-dependent**, exactly as you proposed: good on some problem families, bad on others (here: mini-batch linear regression).

---

## Key numbers (CPU suite, 3 seeds quadratic / 3 seeds others)

### Quadratic (Gaussian), terminal objective (lower better)

| p | τ=0.1 | τ=0.5 | τ=1 | τ=2 | τ=4 |
|--:|------:|------:|----:|----:|----:|
| 50 | SR wins (0.006 vs 0.02) | SR | SR | SR | SR (0.90 vs 1.45) |
| 150 | SR | SR | SR | SR | SR |
| 300 | SR (strong $t$) | SR | SR | SR | SR (near tie at τ=4) |

### Student-$t$, $p=150$: SR wins at every τ.

### Logistic test log-loss: SR wins at flip 0.0, 0.1, 0.3.

### MLP: SR 0.0159 vs AdamW 0.0197.

### Ridge (failure family)

| setting | AdamW MSE | SR-Adam MSE | Δ |
|---------|----------:|------------:|--:|
| p=50, B=32 | 0.036 | 0.114 | **−0.078** |
| p=50, B=256 | 0.033 | 0.052 | **−0.019** |
| p=150, B=32 | 0.129 | 0.682 | **−0.553** |
| p=150, B=256 | 0.084 | 0.192 | **−0.108** |

Hypothesis for later theory work: ridge has **small residual variance near the optimum** (estimation problem gets easy) while Stein shrinkage still injects bias / slow adaptation; or step-size / curvature scale mismatch. Mini-batch linear regression is a **known hard case** for this method under current defaults.

---

## Why this is different from the CIFAR `main` grid

| | CIFAR CNN (`main`) | Path 2 synthetic |
|--|-------------------|------------------|
| SR-Adam vs Adam-family | **Loses 6/6** | **Wins 24/28 vs AdamW** |
| Problem | over-parameterized CNN, huge FC block | controlled $p$, known noise |
| Story | method not a drop-in for CNN training | method is a **candidate in noise-dominated / estimation-like regimes** |

---

## Artifacts

| File | Role |
|------|------|
| `code/phase_diagram.py` | Suite (`--profile cpu` / `full` / `smoke`) |
| `out/phase_diagram.json` | All rows + paired $t$ |
| `figs/phase_diagram_slices.pdf` | Slice + win/lose bars (if PDF locked, timestamped copy) |
| `figs/phase_diagram_heatmap.pdf` | $p\times\tau$ map + mean $k_t$ |

Reproduce:

```bash
cd code
python -u phase_diagram.py --profile cpu
```

---

## Recommendation (Path 2 → next)

1. **Theory paper narrative is viable:** risk-optimal weight helps on a **map of synthetic regimes** (noise level, heavy tails, label noise, small MLP); it is **not** universal (ridge, and previously CIFAR CNN).
2. **Path 3 (real UCI/OpenML regression)** is still worth running **to see whether real datasets look like the quadratic/win side or the ridge/lose side.** Especially include **linear / ridge-like** tabular sets — if they all lose, we document “linear least-squares mini-batch is a negative regime.”
3. Do **not** claim CIFAR superiority from Path 2.
4. For the manuscript: add Path-2 phase diagram as the **primary optimization evidence**; keep CIFAR as limitation or drop it; add a short paragraph on **when not to use SR-Adam** (ridge-type losses with low label noise).

---

## One-line summary for Dr. Arashi

**Path 2 (synthetic, no CIFAR): SR-Adam beats AdamW in 24/28 controlled settings (quadratic all noise levels, Student-t, noisy logistic, small MLP) and loses only on mini-batch ridge regression — a regime map consistent with a decision-theoretic shrinkage story, not with universal optimizer dominance.**
