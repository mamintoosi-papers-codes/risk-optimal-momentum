# Empirical report and manuscript alignment notes
## For Prof. M. Arashi — SR-Adam revision

**To:** Prof. M. Arashi  
**From:** Experimental review (with M. Amintoosi)  
**Date:** 22 September 2026  
**Repository:** `https://github.com/mamintoosi-papers-codes/risk-optimal-momentum`  
**Manuscript file in use:** `paper/sn-article.tex` (your draft)  
**Scope:** Empirical findings for Algorithm 1 (`code/sr_adam.py`); list of prose edits still needed outside the new experiments section.

---

## Abstract (فارسی — کوتاه)

**یافته‌ها:** در مسائل **پُربعد** و **پُر‌نویز** (جاروب $p$، نویز گرادیان/برچسب، رگرسیون خطی واقعی، گراف)، SR-Adam معمولاً بهتر از AdamW است؛ مزیت با $p$ و نویز **بیشتر** می‌شود.  
**مرز:** شبکه‌های کانولوشنی کوچک (مانند MNIST-CNN) می‌توانند **بدتر** باشند (بیش‌شرینک روی بلوک‌های متراکم بزرگ).  
**در مقاله:** بخش «Image classification» با «Empirical evaluation» (بدون CIFAR) عوض شده است.  
**نیازمند اصلاح در بخش‌های دیگر (فقط عنوان موارد، بدون دست‌زدن به متن):** چکیده، بحث (گفت‌وگو دربارهٔ backbone)، Limitations، آدرس کد، و یکدست‌سازی قرارداد علامت در جدول رگرسیون.  
**بدون CIFAR در مقاله** تا مشورت بعدی.

---

## 1. What was run (current algorithm)

Paired comparisons of **SR-Adam** (Algorithm 1 defaults) against AdamW / AMSGrad / SGD-M, same code base and seed pairing.

| Family | What | Headline |
|--------|------|----------|
| **Dimension sweep** | Quadratic + Student-$t$ noise + logistic label flips, $p=10\ldots500$ | **23/24** settings favour SR-Adam; gap **grows with $p$** |
| **Real regression** | diabetes, California housing, wine quality, linnerud, energy, concrete, protein, BlogFeedback, year (if available) | **All linear cells** favour SR-Adam (or tie); clear MLP wins on small-$n$ / moderate-$p$ |
| **Label noise on real $y$** | wine, California, diabetes | **8/9** favour SR-Adam; margin ↑ with noise |
| **Graph node classification** | Cora, CiteSeer, PubMed (GCN), clean + noisy labels | **≥ AdamW** on all cells; slight edge under noise |
| **Linear image classification** | digits logistic + 30% label noise | **+1.3** accuracy points (SR-Adam) |
| **Small CNN (MNIST)** | SmallCNN | **−1.0** accuracy points (AdamW ahead) |
| **Monte Carlo (paper §simulation)** | risk / variance bias / elliptical window | Consistent with the theorems |

Evidence files: `out/*.json`, figures in `figs/` and `paper/*.pdf`, scripts in `code/`.

---

## 2. Strengths of the idea (as supported by data)

1. **High dimensions** ($p\gtrsim 50$–$500$): systematic, growing advantage over a fixed Adam weight — matches the Stein story in the title.  
2. **Noise-dominated settings**: gradient noise, Student-$t$ tails, **label noise** (synthetic and real targets).  
3. **Estimation-heavy models**: linear regression, shallow MLPs, **graph convolution** node classification.  
4. **Practical**: same AMSGrad-class state; gate and debiasing are cheap scalar reductions.

---

## 3. Weaknesses / boundaries (for honesty; not in the current experiments section)

| Boundary | Evidence |
|----------|----------|
| **Small CNN** (MNIST-CNN; previously also a small CIFAR CNN in lab notebooks) | Stein weight on the large dense block ≈ 0 → over-shrinkage; theory-friendly risk, worse discrimination |
| **Mini-batch ridge** (synthetic least squares) | Clear MSE loss vs AdamW |
| **Mid-size tabular MLP** | Near-tie / occasional slight loss |
| **Huge models / transformers** | Not tested |

**Suggested one-line mechanism for CNN:** when $\hat\sigma^2$ is large and concentrated, $k_t\to 0$ and $d_t\approx\hat m_{t-1}$ (Lemma \ref{lem:displacement}); good for estimation risk, bad if the task needs the current batch to reshuffle features quickly.

---

## 4. What is already in `paper/sn-article.tex`

Section **Image classification** was replaced by **Empirical evaluation** (`sec:experiments`) only:

- Dimension & controlled noise (`tab:dim_sweep`, `fig:dim_sweep` — PDF)  
- Real regression + label noise (`tab:real_reg` with **MNIST CNN** boundary row, `fig:real_gaps`)  
- Graph node classification (`tab:gnn`, `fig:gnn_acc`)  
- Win-rate overview (`fig:regime_win`)  
- **No CIFAR wording** (per your request until further discussion)

Figures are **PDF**. Qualitative digits PNG was **not** kept.

---

## 5. Answer: do other parts of the paper need serious revision?

**Yes — a short, targeted revision of prose outside `sec:experiments`.**  
The theorems and Monte Carlo section do **not** need scientific rewrites. The **empirical framing** elsewhere still assumes the old single-backbone image study. **I did not edit those parts** (as you asked). Below are the **headings of items you (or we) should revise**.

### A. Abstract (`\abstract`)

- **A1.** Last sentence still only says “controlled optimisation experiments show gains that grow with the gradient-noise level.”  
- **A2.** Should briefly state the **empirical scope now in the paper**: gains grow with **dimension** and **noise**, including **real regression and graphs**; optional one clause that advantage is **regime-dependent** (not universal).

### B. Organisation paragraph (Introduction §1.2)

- **B1.** Still says Section `sec:experiments` reports “the empirical evaluation…” (partly updated earlier). Confirm wording matches the **new** section title *Empirical evaluation* and its four families (dimension / regression / graphs / small CNN boundary).

### C. Risk theory remark on the gate (Remark `rem:gate`)

- **C1.** The remark referred to a **backbone table / CNN layer list** (SimpleCNN). That table is gone.  
- **C2.** Should describe the gate using the **problem families in `sec:experiments`** (e.g. blocks of linear models, MLPs, GCN layers), not “the backbone used in the experiments.”

### D. Discussion (`sec:conclusion`) — **most important prose fix**

- **D1.** “the backbone used here a layer that a dimension-based rule would exclude holds most of the parameters” — **stale CIFAR/SimpleCNN claim**. Must be rewritten (e.g. parameter count vs peff story, using the **MNIST CNN / dense block** explanation already in `sec:experiments`).  
- **D2.** “the improvement in gradient-estimation risk… regime-dependent” is **good** and should stay, optionally strengthened with a pointer to `sec:experiments` (dimension sweep + MNIST boundary).  
- **D3.** Optional: one paragraph **synthesis** of the regime map (high $p$ / noise / linear & shallow / graphs vs dense CNN), consistent with `fig:regime_win`.

### E. Limitations (`\paragraph{Limitations}`)

- **E1.** “**The empirical study uses one small backbone and two datasets**” is **false** after the replacement. Must be rewritten to list the **breadth** (dimension sweep, tabular regression, graphs) and **true gaps** (large CNNs / transformers / scale, ridge-type least squares, mid-size MLPs).  
- **E2.** Optional: state explicitly that **small CNNs** are a known weaker regime (MNIST row), so reviewers are not surprised.

### F. Declarations / code URL

- **F1.** URL is still `.../SR-Adam`. Should be `https://github.com/mamintoosi-papers-codes/risk-optimal-momentum` (or whichever repo you want cited).

### G. Cross-checks (mechanical)

- **G1.** **Sign convention** of `tab:real_reg`: MSE rows use AdamW − SR-Adam; MNIST row is accuracy SR-Adam − AdamW. Unify or restate clearly in the caption (already partly done; still easy to misread).  
- **G2.** Bibliography: `kipf2017` is required by the graphs subsection (present in `sample.bib`).  
- **G3.** Unused package/cites left over from the deleted section (e.g. if `krizhevsky2009` is unused, drop from bib or leave).  
- **G4.** Compile once after edits; fix any undefined refs (`tab:arch` etc. should already be gone).

### H. Not required (science)

- **H1.** Proofs / theorems in the appendix: **no change** for this empirical swap.  
- **H2.** Monte Carlo section: **no change** required (still the core validation of constants).

**Severity:** **D + E** are mandatory for consistency (a referee would contradict the text against the new section). **A, C, F, G** are strongly recommended. **B** is cosmetic.

---

## 6. Suggested manuscript claim (if you want one sentence for Discussion)

> The risk-optimal Stein weight replaces a fixed momentum weight and improves optimisation when gradient estimation is noisy and the parameter block is high-dimensional—on controlled objectives as $p$ grows, on real linear and shallow regression, and on graph node classification—while it can fail on convolutional networks whose large dense blocks drive $k_t$ toward zero.

---

## 7. Decisions for you

| ID | Decision |
|----|----------|
| **D1** | Approve the five prose fix groups **A–G** (I can implement them in one pass after your OK). |
| **D2** | Keep MNIST CNN as an explicit boundary row (recommended) or omit. |
| **D3** | Whether any **CIFAR** material ever returns (lab report has the numbers; currently **out of the paper**). |
| **D4** | Code URL / repo name in Declarations. |

---

## 8. One-line summary

**Experiments now support a high-dimensional, noise-robust advantage of SR-Adam on controlled, regression, and graph problems (and show a small-CNN boundary); the new §Empirical evaluation is in place, but Abstract, Remark on the gate, Discussion, Limitations, and the code URL still assume the old single-backbone image study and should be revised before submission.**

---

## Appendix — Reproduce (selected)

```bash
cd code
python -u path4_dim_sweep.py
python -u path3_uci.py --seeds 5 --steps 800
python -u path5_more_datasets.py --seeds 3
python -u path3b_label_noise.py
python -u gnn_pilot.py && python -u gnn_pilot_planetoid.py
python -u mnist_cnn_pilot.py
python -u make_paper_barfigs.py   # PDF bar charts for the paper
```

Figures in the paper: `path4_dim_sweep.pdf`, `real_reg_gaps.pdf`, `gnn_acc.pdf`, `regime_winrate.pdf`.
