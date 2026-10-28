# Response to the pre-submission diagnostic (SR-Adam)
## Status of the revision work — what is done, what is not

**To:** Prof. M. Arashi
**From:** Experimental work (M. Amintoosi, with the coding agent)
**Date:** 18 October 2026 (work continued **20–21 October 2026**; see §4 items 11–13)
**Answers:** the diagnostic *Pre-submission diagnostic: Risk-Optimal Momentum Weights (SR-Adam)* dated 17 Oct 2026 (items **a1–a10**, **b1–b6**, **c**)
**Manuscript:** `paper/sn-article.tex` (rebuilt today, `paper/sn-article.pdf`)
**Evidence:** `out/*.json`, `figs/*.pdf`, `paper/tab_*.tex`, `out/stats_tests.md`

---

## فارسی — خلاصهٔ کوتاه

چهار فازِ کارِ تجربی (A: آمار روی نتایج موجود، B: شبیه‌سازی‌های نظریه، C: پایه‌های tuneشده، D: اجرای نهایی همهٔ اسکریپت‌ها) کامل شد؛ فاز E (ساخت PDF) هم سبز است. پاسخ موارد a3، a4، a5، a7، a8، a9 (بخش تجربی)، b1، b2، b3 و آیتم زمان‌سنجیِ بخش c داده شده و شماره‌های مقاله با داده‌های تازه به‌روز شد. **نتیجهٔ مهم و صادقانه:** با step sizeِ tuneشده برای هر بهینه‌ساز، مزیت SR-Adam همه‌جا ادامه پیدا نمی‌کند — در `tab:dim_sweep` علامت شکاف به سطح نویز بستگی دارد نه به بُعد، و در مطالعهٔ کنترل‌شده در بازهٔ میانیِ نویز AdamW جلو می‌افتد؛ در مقابل در رگرسیون خطی (۵/۵)، نویز برچسب (۱۵/۱۹) و گراف‌ها (۶/۷) SR-Adam جلو است. **به‌روزرسانی ۲۰ اکتبر** دو ستون پُر-نویز به sweep اضافه شد (τ=4 و flip=0.4) که دقیقاً همان نویتهای قوت نظریه‌اند — flip 0.4 برد در **هر ۷ بعد** با شکافِ روبه‌رشد (|t| تا 47.7) و τ=4 برد در سه بعد بزرگ — و حالا `tab:dim_sweep` **۱۷ برد از ۴۲** تنظیم است (قبلاً ۷/۲۸) و دو ستون جدید با شبکهٔ جداگانه در همان سطح نویز tune شده‌اند. **به‌روزرسانی ۲۱ اکتبر:** (۱) ردیف‌های گراف هم tune شدند — هر دو روش روی هر مجموعه‌داده دقیقاً یک lr انتخاب کردند و رکورد بدون تغییر ۶ برد/۱ تساوی/۰ باخت ماند؛ (۲) ساختار §9 با پیشنهاد شما به **رگرسیون (9.1) ← طبقه‌بندی (9.2) ← Ablation study (9.3: نویز برچسب + بُعد)** تغییر کرد و زیربخش بُعد حالا **تلفیقی از دو پروتکل** است: `tab:dim_sweep` (تیون‌شده، ۱۷/۴۲) به‌عنوان تیتر و `tab:dim_sweep_common` (lr مشترکِ پیش‌فرض نسخهٔ قبلی، با همان seedها و گام‌ها دوباره اجرا شده) با **۳۸/۴۲**؛ دو جدول دقیقاً در ۲۱ سلول — همه در باند میانیِ نویز — فرق دارند. (۳) به پیشنهادِ تازهٔ شما شکلِ `fig:dim_sweep` از مقاله حذف شد — اطلاعاتش را هر دو جدول می‌دهند — و ارائهٔ دو جدول جابه‌جا شد: اول بازوی پروتکلِ مشترک (`tab:dim_sweep_common`، ۳۸/۴۲) بعد تیون‌شده (`tab:dim_sweep`، ۱۷/۴۲)، پس آن‌ها حالا به‌ترتیب Table 10 و Table 11 هستند و PDF به ۴۱ صفحه رسید. موارد a1، a2، a10، b4، b5 و b6 (متنی/نظری/هماهنگی) برای شما مانده است.

---

## 1. What was executed

Four phases were run end to end (`scripts/phaseA.bat` … `phaseD.bat`, full mode, 18 Oct 2026), plus the manuscript build (`scripts/phaseE.bat` logic, run manually).

| Phase | Scripts | Status | Main artefacts |
|---|---|---|---|
| **A — statistics on stored results** | `make_paper_barfigs.py`, `stats_tests.py` | OK | `out/winrate_summary.json` (ties counted separately), `out/stats_tests.{md,json}` (paired *t*, Wilcoxon, Holm, SEs per cell) |
| **B — new theory-side simulations** | `verify_plugin_pospart.py`, `sim_a5_covariance.py`, `log_variance_ratio.py`, `sim_window_signal.py`, `conv_constants.py` | OK | five JSONs in `out/`, figures `figs/var_ratio_*.pdf`, `figs/sim_window_signal.pdf` |
| **C — fair-baseline infrastructure** | `tune_helper.py` + the four drivers with `--tune --timing` (plus `--gate-log` on `simulation_noise`/`paper_tables`, `--early-stop 200` on `noise_sweep`) and, since 21 Oct, both GNN pilots with `--tune` | OK | `out/*_tune.json` grid tables (auditable), including `out/gnn_{cora,citeseer,pubmed}_tune.json`; timing on every synthetic/tabular row, gate fractions where `--gate-log` was passed — the batch study records neither timing nor gate fractions |
| **D — ordered final re-run** | `test_sr_adam.py` → theory checks → `simulation_study(3).py` → `simulation_noise.py` → `simulation_batch.py` → `path4_dim_sweep.py` → `paper_tables.py` → `noise_sweep.py` → GNN pilots (+ `_refresh_gcn_rows.py`) → `make_figures.py` → `make_paper_barfigs.py` → `stats_tests.py` | OK after one fix | all `out/*.json` regenerated 18–21 Oct, `paper/tab_*.tex`, all PDFs |
| **E — manuscript** | `pdflatex` + `bibtex` + `pdflatex` ×2 | OK | `paper/sn-article.pdf`; no undefined references, no overfull boxes, `\torerun` appears only in the header comment and the macro definition |

**Bug found and fixed in phase D.** `make_figures.py` crashed with `KeyError: 'closed_form'`: `out/simulation_results.json` in the repository still carried the `closed_form` key written by an older version of `study B`, but the current `simulation_study.py` no longer emitted it. The generator was repaired (`study_B` now writes `closed_form = 1 − φ_t(β₁)` alongside the simulated curves), `simulation_study.py` was re-run (deterministic, seeded), and steps 9a/9b/10 of phase D were then completed. No test was weakened or skipped; every exit status was captured directly, not through a pipeline.

---

## 2. Headline result of the re-run (please read before anything else)

**With per-optimiser tuned step sizes the empirical picture is materially weaker and more structured than the draft claimed.** This is exactly what diagnostic items a7 and a8 asked for, and it changes the abstract claim (a6).

| Study (tuned) | Result |
|---|---|
| `tab:dim_sweep` (42 cells, 20 Oct) | **17 wins / 25 losses.** The wins sit at *both ends of the noise axis*: every p at τ=0.1 (7/7) and at flip 0.4 (7/7), plus p ∈ {100,200,500} at τ=4. A tuned AdamW wins the intermediate band — at τ=1, under Student-t at τ=1 and at flip 0.2 **AdamW wins at every p**. Under heavy label noise the per-coordinate gap **grows with p** (7e-4 → 8e-3, \|t\| 10.5–47.7 from p=50 on), so "advantage grows with p" *is* supported in the corruption-dominated regime; at τ=0.1 the per-coordinate gap stays flat (3e-6…1.2e-4) |
| `tab:opt_noise` (quadratic, 5 seeds × 3000 steps) | SR-Adam 32 % and 9 % *below* AdamW at τ=0.1, 0.25; 27 % and 24 % below at τ=2, 4; **11 % and 30 % *above* at τ=0.5, 1** — a band, not monotone growth |
| `tab:opt_noise` (logistic, 3 seeds × 2000 steps) | ahead (4–5 %) up to flip 0.2, **behind** (≈5 %) at flip 0.3 and 0.4, where tuned AdamW drops its lr to 0.003 |
| `tab:reg` | linear rows **5/5** in favour (largest: BlogFeedback +1.21); MLP rows 3/2 |
| `tab:noise` | classification **9/10** in favour (Digits +3.9 → +13.5 pp as the flip rate grows; only clean Digits is a loss), regression 6/9 (two California-Housing cells negative, one positive but imprecise) |
| GNN pilots | Cora/CiteSeer/PubMed, clean + noisy: **6 wins, 1 tie** at the tuned step sizes (the tie is CiteSeer at clean labels; the accuracy levels rise, e.g. Cora 77.8/78.1 → 78.6/78.9). Step sizes tuned per dataset on 21 Oct, see §4 item 12 |
| `tab:batch` | B=8 (t=2.70) and B=32 (t=1.89) are **below** the 5 % critical value 2.776 (df=4), and B=16 is significant but the margin there is only +0.0032 — the draft's "statistically distinguishable" wording was wrong; B≥64 is significant but favours **AdamW** |
| Gate (a9) | recorded as the fraction of parameter blocks open **at the last step** of each run: **100 %** on every run of `tab:opt_noise` and on Digits, 0–75 % on the remaining tabular rows (a block of dimension ≤ 3 can never open it) — the safeguard does fire on some tabular blocks |
| Timing (item c) | mean per-step wall time 0.55 ms (p=400 quadratic, Gaussian; 0.90 ms under Student-t) and 1.8 ms (200-feature logistic), averaged over the three optimisers of the sweep (re-measured with the 20 Oct sweep re-run) |

The honest summary the data supports: **a noise-robust advantage in the estimation-heavy and corrupted-label regimes, no advantage (and a small deficit) once AdamW is tuned at the intermediate noise scales.** This is consistent with the third finding already in the Discussion, and it is *not* a universal-optimiser claim.

---

## 3. Diagnostic items — status

### (a) Mandatory

| # | Status | Evidence / note |
|---|---|---|
| **a1** "understated → overstated", Sec. 3 | **Pending — yours** (one-word text fix; the agreed split assigns all prose to you) |
| **a2** Remark 5.2 pooling bias, Prop. 10 Jensen bound | **Pending — yours** (theory restatement). The measurement machinery of `sim_a5_covariance.py` can supply the identity numerically if you want it cited |
| **a3** (A5) vs (A3′); ν≤4 no fourth moment | **Done (simulation side).** `sim_a5_covariance.py`: under a shared mixing variable W the off-diagonal correlation of coordinatewise variance estimates is 0.47 (ν=3, p=25) and 0.21 (ν=4), against ≈0.002 for Gaussian; Σⱼ\|Cov\| grows from 638 (p=25) to 6994 (p=100) at ν=3, i.e. the C/p bound of Thm 8 cannot hold there. The ν≤4 fourth-moment failure is confirmed by the excess kurtosis rows. **Text restatement still yours** |
| **a4** positive part with plug-in scale | **Done.** `verify_plugin_pospart.py`, 60 cells (p ∈ {20,60,200} × log-spread of r_j ∈ {0,…,1.2} × homo/hetero × μ ∈ {0, moderate}, 2×10⁵ draws): the rule with c = p−2 dominates in **all 60 cells**, is never worse than the raw gradient, and **no empirical sign change occurs below the Cor. 2(iv) window** — the window is sufficient and conservative, as in the known-scale case. Verdict: the plug-in gap can be closed honestly by reporting this Monte Carlo (a one-paragraph addition is yours) |
| **a5** variance ratio along real runs; matched windows | **Done.** `sr_adam.py` gained `var_window="matched"`; `log_variance_ratio.py` logs σ̂²/σ_true² along the controlled quadratic and logistic runs. With the original β₁-horizon correction the ratio peaks at **2.00** (0.34 % of steps beyond the factor-two margin, at t≈100 on the decaying-mean problem); with **matched windows it stays in [1.00, 1.25]** and never leaves the margin. Figures: `figs/var_ratio_quadratic.pdf`, `figs/var_ratio_logistic.pdf`. This is the experiment your item 4 said was missing |
| **a6** abstract "gains grow with the noise level" | **Done in the manuscript** (see §4) — please review the replacement wording |
| **a7** untuned baselines; regularisation confound | **Done.** `tune_helper.py` + `--tune` on all four drivers: a candidate step-size grid shared across the Adam-family optimisers (SGDM on its own, lower range), scored on two tuning seeds — held-out validation loss on the logistic pilots, terminal loss elsewhere — full grid tables written to `out/*_tune.json`. The controlled study retunes at every noise level; the sweep and tabular drivers tune once per problem family on one representative pilot and reuse it (now disclosed in §9). The graph pilots are tuned per dataset on a five-point grid (1e-3 … 1e-1, identical for both optimisers, scored on validation accuracy at clean labels; `out/gnn_{cora,citeseer,pubmed}_tune.json`) — the 21 Oct pass below closed the last untuned cells of §9. `--early-stop 200` control arm in the noise sweep; metadata records explicitly that only **training** labels are corrupted and test labels stay clean. **Not yet done: the β₁ grid.** `tune_helper --beta1-grid` exists, but its axis is restricted to SR-Adam and no reported run passed it, so AdamW still runs at its default β₁ = 0.999; the diagnostic's "β₁ grid including β₁ = 0" would only strengthen the baseline further (D6 below). Result: see §2 — the untuned advantage partly disappears, which is the point of the item |
| **a8** Tables 5 vs 8 irreconcilable; "23 of 24"; per-p normalisation | **Done.** The sweep was re-run on one unified configuration (τ ∈ {0.1, 1, 4}, flip ∈ {0.2, 0.4}, p ∈ {10,…,500}, 5 seeds, 800 steps, tuned; the two heavy-noise columns retuned at their own level). `stats_tests.py` §A4 emits both counts: **42 cells in the JSON, all 42 printed in the table** (as of the 20 Oct pass) and the per-p normalised gap. The manuscript table has all 7 p rows × 6 columns and the "23 of 24" claim is gone |
| **a9** dangling promises and stale text | **Partly done.** Gate frequencies: recorded (`gate_open_frac`, the fraction of parameter blocks open at the last step of a run) and now reported in §9 of the manuscript. Stale sentences fixed: Limitations ("one small backbone and two datasets") and the Discussion sentence about "the backbone used here". The remark after Cor. 6 promising empirical windows at several signal levels: **data ready** (`sim_window_signal.py` → W̄(μ) = 1.001, 1.046, 1.182, 1.360, 1.578 at ‖μ‖ = 0, 2, 5, 10, 30, ν=5, p=30, with the μ=0 closed form confirmed to 1.4×10⁻³) — the 3–4 sentences in §5 are yours to add, or I can draft them |
| **a10** arXiv:2602.01777 conflict | **Not done — coordination item.** Needs the cover-letter disclosure and an updated arXiv version; both are outside this repository |

### (b) Important

| # | Status | Evidence / note |
|---|---|---|
| **b1** vacuous κ; piecewise B_t; Jensen floor | **Done (numbers).** `conv_constants.py`: κ = 10⁸·G_∞ at ε=10⁻⁸ (confirming your objection), realised floor slope −0.896 with a mean fall of **1.86× per batch doubling** against the predicted 2, and a direct MC of the displacement bound. The piecewise restatement of B_t and the softening of "exactly" are yours |
| **b2** window proved only at μ=0 | **Done (data).** `sim_window_signal.py`: W̄ rises monotonically with ‖μ‖ (table under a9), so μ=0 is the worst case numerically, and the positive-part risk rows at ν ∈ {3,5,10} with ρ ∈ {0.5,1} are in `out/sim_window_signal.json`. The proof is yours |
| **b3** seeds, tests, ties, "distinguishable" | **Done.** `stats_tests.py` reports per-cell SE, paired *t*, Wilcoxon with Holm correction, and win/tie/loss with ties excluded from wins; the win-rate figure is now data-driven; the `tab:batch` wording in §8.6 was corrected (B=8 and B=32 are **not** significant) |
| **b4** one larger-scale run; modern baselines | **Not done — needs your decision.** CIFAR/ResNet was abandoned earlier (bad results, project decision). Substitutes available now: the Planetoid graphs (PubMed, 19 717 nodes) and BlogFeedback (8 000 × 280). If a referee-level scale claim is wanted, we need a decision on hardware and backbone |
| **b5** positioning vs Balles & Hennig (2018) | **Not done — yours** (related-work paragraph + bib entry) |
| **b6** clipping breaks exact unbiasedness | **Not done — yours.** The numbers are already in the paper (Table 2 caption: 0.9492 vs 0.9458 at t=40); it needs a remark, not an experiment |

### (c) Minor

Notation collisions, Acknowledgements, .bib completeness — **yours.** Per-step timing (the missing complexity item) — **done**, see §2 and the new paragraph in §9.

---

## 4. Manuscript edits already applied (please review, not re-derive)

All in `paper/sn-article.tex`; the PDF rebuilt cleanly.

1. **Abstract** — the last sentence no longer claims gains that grow with the noise level; it states the regime-dependence (a6).
2. **Section 9 intro** — new paragraph describing the tuning protocol, the per-step timings and the gate frequency (a7, a9, item c).
3. **`tab:dim_sweep` prose** (originally §9.1, now §9.3.2 inside the *Ablation study* subsection): rewritten twice — on 18 Oct to the tuned-protocol numbers, on **20 Oct** (your option A) to the strengths-first version with `tab:dim_sweep` at 7 p-rows × **6** columns (τ=0.1, τ=1, τ=4, Student τ=1, flip 0.2, flip 0.4); it now states that the wins sit at both ends of the noise axis (17/42), that under heavy label noise the per-coordinate gap grows with p, and why the flip-0.4 result does not contradict `tab:opt_noise`. Figure caption corrected.
4. **Dataset list** — Protein marked as skipped when the OpenML fetch is unavailable (it was absent from the fresh table).
5. **§9.1–§9.2 (regression, classification) and §9.3.1 (label noise)** — all quoted numbers replaced by the fresh ones, including the sentences that were false after the re-run ("the MSE gap stays positive at every noise level" is gone).
6. **§8.5 / `tab:opt_noise`** — full table replaced with the tuned numbers, prose rewritten to describe the band, and the SGD sentence corrected (tuned SGD is competitive on clean logistic: 0.233 vs AdamW 0.244).
7. **§8.6** — "statistically distinguishable" replaced by the correct statement (a3 of phase A / b3).
8. **`fig:regime_win` caption and the closing synthesis paragraph** — new fractions 17/42 (20 Oct), 5/5, 15/19, 6/7 (+1 tie), 3/5, and the per-column normalisation statements.
9. **Discussion and Limitations** — the stale backbone sentence and "one small backbone and two datasets" removed.
10. **19 Oct figure/consistency pass** — three defects found and fixed after your PDF comparison: (a) `paper/` still embedded the *pre*-re-run drawings for `fig:dim_sweep` (`path4_dim_sweep.pdf`, 22 Sep) and all six `sim_*.pdf`, so the manuscript showed untuned curves next to tuned tables (the figure literally showed positive τ=1 gaps while `tab:dim_sweep` says negative); the fresh `figs/` copies are now in `paper/`, and `phaseE.bat` gained an **E2b freshness check** that fails the build when the two diverge (since 21 Oct the `path4_dim_sweep` figure is no longer part of the manuscript — item 14 — so E2b guards the six `sim_` figures only). (b) `fig:regime_win`'s bar colour rule was wrong (`wins >= total - losses` ⇔ "no ties"): the GCN bar was red despite 6 wins / 1 tie / 0 losses, while the 7/28 dimension sweep stayed green; colours are now green iff wins/total ≥ 0.5, the pre-diagnostic rule (on 20 Oct the sweep fraction rose to 17/42 = 0.405, so its bar is honestly red again while GCN stays green). (c) section references in this report that said §8 for manuscript items that live in §9 (timing, gate, tuning protocol, `tab:dim_sweep`, regression/classification/label-noise prose) are corrected above.
11. **20 Oct heavy-noise pass (your option A).** Two noise levels were added to the dimension sweep as strength columns: quadratic τ=4 and logistic flip 0.4, each **tuned at its own noise level** on the same grid as every other optimiser (`out/path4_quad_t4.0_tune.json`, `out/path4_log_f0.4_tune.json`; lr = 0.02/0.02/0.02 and 0.1/0.1/0.1). Re-run: 42 settings, **17 wins** — all 7 at τ=0.1, all 7 at flip 0.4 (\|t\| up to 47.7, per-coordinate gap growing 7e-4 → 8e-3 with p), and p ∈ {100,200,500} at τ=4. Manuscript, `fig:dim_sweep` (now 6 panels), `fig:regime_win` caption, `stats_tests` §A4 and all three verifiers updated accordingly; timings re-measured with the re-run (0.55 / 0.90 / 1.8 ms) and both the text and the verifiers follow the fresh JSON.

12. **21 Oct graph tuning pass (your decision).** The graph pilots were the last §9 cells without a step-size grid — they ran at a common fixed lr = 0.01, disclosed in §9 as the sole exception. Both pilots now accept `--tune` and grid-search **five** step sizes (1e-3 … 1e-1, identical for AdamW and SR-Adam, extended down after the first run selected the 3e-3 boundary) on the two tuning seeds, scored on **validation** accuracy at clean labels, and reuse the chosen value at both label-noise levels. `paper_tables.gnn_rows()` reads those choices from `out/gnn_{cora,citeseer,pubmed}_tune.json`, so the GCN rows of `tab:clf` are now tuned too; `scripts/phaseD.bat` steps [8a–8c] run both pilots with `--tune` and then `code/_refresh_gcn_rows.py`. Outcome: **both optimisers pick the same step size on every dataset** (0.001 on Cora and CiteSeer, 0.003 on PubMed) and the record is **unchanged — 6 wins, 1 tie, 0 losses**, with the tie moving from PubMed at 0.3 noise to CiteSeer on clean labels; the accuracy levels rise (Cora 77.8/78.1 → 78.6/78.9, CiteSeer 64.7/65.0 at the table's 150 epochs). The §9 intro no longer announces a fixed-lr exception, §9.3's prose follows the new margins, `fig:regime_win` still shows Graphs 6/7 with one tie, and the two verifiers that asserted the fixed-lr wording were replaced.

13. **21 Oct ablation restructure (your proposal).** §9 is now ordered **Real regression (9.1) → Classification (9.2) → Ablation study (9.3)**, with 9.3.1 *Label noise* and 9.3.2 *Dimension and controlled noise*, so the section opens on the real-data evidence and the qualification lands last. The dimension subsection fuses both step-size protocols: `tab:dim_sweep` (tuned, **17/42**) remains the headline, and a new `tab:dim_sweep_common` reports the **shared default step sizes of the pre-diagnostic draft** (0.02 quadratic, 0.03 logistic) re-run under the *same* seeds, step count and grid — the old numbers themselves were not reused, because the earlier draft used 3 seeds and 500 steps. Under that protocol **38 of 42** settings favour SR-Adam, and the gap grows with $p$ on every column (quadratic τ=1: +0.010 → +0.621 from p=10 to 500). The two tables differ in exactly **21 cells, all in the intermediate band** (τ=1, Student-t at τ=1, flip 0.2): there AdamW's tuned step size is half the shared default (0.01 vs 0.02 quadratic, 0.01 vs 0.03 logistic) while SR-Adam keeps the value the default already gave it — the earlier draft's common step size was in effect SR-Adam's own — whereas both ends of the noise axis give identical outcomes under the two protocols. New code: `path4_dim_sweep.py --common-lr` → `out/path4_dim_sweep_common.json` (that arm writes no figure); `verify_fresh_outputs.py` gained section 5 (both tables, 38/42, the 21-cell claim); `stats_tests` §A4 now reports both populations. Rebuild: 42 pages, 0 overfull, 0 undefined references; all three verifiers exit 0 (that pass's float numbers: `tab:dim_sweep` = Table 10, `tab:dim_sweep_common` = Table 11 — arrangement superseded by item 14 below; every cross-reference is `\ref`, nothing hardcodes numbers).

14. **21 Oct figure reduction and table order (your decision).** With both step-size protocols printed as tables in §9.3.2, the six-panel `fig:dim_sweep` (Fig. 8) was removed from the manuscript: the two tables carry the same information, and the subsection is an ablation by construction. The tables were also reordered so the reference protocol is read first — `tab:dim_sweep_common` (shared defaults, **38/42**) is now **Table 10** and the tuned `tab:dim_sweep` (**17/42**) **Table 11** — with the prose following that order: configuration and shared-default arm, then the tuned arm with its regime analysis, then the 21-cell reconciliation; the closing sentence still names the tuned table as the fair comparison and the headline. `fig:regime_win` is now Fig. 8, `tab:noise` stays Table 9. Supporting changes: `paper/path4_dim_sweep.pdf` deleted and `path4_dim_sweep.py` no longer copies the figure into `paper/` (the driver still writes the `figs/` and `reports/` copies; the `--common-lr` arm never wrote a figure), the provenance comment in `sn-article.tex` dropped the `fig:dim_sweep` line, and `phaseE.bat`'s E2b freshness list no longer contains the figure while still guarding the six `sim_` figures. Captions follow the new order: Table 10 states the full sweep specification itself, and Table 11 refers to it and adds the per-optimiser tuning detail. Highlighting the cells where SR-Adam wins was considered and skipped --- the gap is signed, the win counts are in the captions, and in Table 10 it would mark 38 of the 42 cells. No verifier asserted anything about the figure. Rebuild: **41 pages, 0 overfull, 0 undefined references**; `verify_manuscript_numbers.py` and `verify_fresh_outputs.py` re-run after the change **exit 0**; float numbers confirmed in `sn-article.aux`; `verify_tables2.py` re-run after this report update **exit 0**.

**Not touched, as agreed:** every proof, every theorem statement, (A5), Remark 5.2, Sec. 3 wording, the Cor. 6 remark, notation, Acknowledgements, .bib.

---

## 5. Decisions we need from you

| ID | Decision |
|---|---|
| **D1** | Approve the new abstract/empirical wording (it is more conservative than the previous draft). |
| **D2** | a10 — how to handle the arXiv preprint disclosure in the cover letter. |
| **D3** | b4 — do we attempt one larger-scale run (needs a backbone and hardware decision), or do we defend the current scope (graphs + tabular + controlled synthetic)? |
| **D4** | Whether I should draft the 3–4 sentences in §5 that report the empirical window W̄(μ) at several signal levels (data already produced). |
| **D5** | Whether the Cor. 2(iv) plug-in Monte Carlo (item a4, 60/60 cells) should become a small table in §3 or stay in the supplementary material. |
| **D6** | **Decided 20 Oct (you):** the diagnostic's β₁ grid for AdamW (including β₁ = 0) is *not* run now; the gap stays disclosed here in a7. The step-size grid the diagnostic asks for is in place, and if a referee requests it we run it then. |
| **D7** | **Decided 21 Oct (you):** §9 restructured to Real regression → Classification → **Ablation study** (Label noise + Dimension and controlled noise). The dimension subsection now reports *both* step-size protocols: tuned `tab:dim_sweep` (17/42) as the headline and `tab:dim_sweep_common` with the shared default step sizes of the earlier draft (38/42), re-run under the current seeds/steps. Executed. |
| **D8** | **Decided 21 Oct (you):** per-noise-level retuning of the remaining `tab:dim_sweep` columns (τ=0.1, Student-t reuse the τ=1 grid) is *not* done for now; disclosed in the §9 intro. ~5 min of compute if a referee asks. |
| **D9** | **Decided 21 Oct (you):** the GNN pilots are tuned per dataset on a five-point grid for both optimisers (executed; record unchanged 6/1/0). |
| **D10** | **Decided 21 Oct (you):** drop `fig:dim_sweep` from the manuscript — §9.3.2 now prints both step-size protocols as tables — and present the ablation table (`tab:dim_sweep_common`, 38/42) *before* the tuned table (`tab:dim_sweep`, 17/42). Executed: float numbers follow the order of appearance (Tables 10 and 11), `fig:regime_win` is Fig. 8, PDF 41 pages. |

---

## 6. Reproduce

```bat
scripts\phaseA.bat          & rem statistics on stored results
scripts\phaseB.bat          & rem the five new MC scripts
scripts\phaseC.bat full     & rem tuned baselines, timings, early-stop, gate
scripts\phaseD.bat full     & rem the ordered final re-run (writes tables + figures)
```
then `pdflatex + bibtex + pdflatex ×2` in `paper/`. Two extra regeneration steps now exist inside `phaseD.bat` (steps [8c]): `gnn_pilot.py --tune`, `gnn_pilot_planetoid.py --tune`, then `code/_refresh_gcn_rows.py` (replaces only the GCN rows of `out/paper_tables.json` and rewrites the two `tab_*.tex` files). The ablation table of §9.3.2 is produced by the same driver without tuning:

```bat
cd code
python -u path4_dim_sweep.py --ps 10,20,50,100,200,400,500 --tau 1.0 --steps 800 --seeds 5 --common-lr
rem writes out\path4_dim_sweep_common.json (that arm writes no figure)
```

`phaseD.bat` with no argument is identical to `phaseD.bat full` — the default mode *is* full (the argument only overrides the default). All drivers take `--quick/--smoke` for a minutes-long pipeline check; the full phases ran on 18 Oct 2026 and every artefact in `out/` and `paper/tab_*.tex` carries its creation timestamp.

**One-line summary:** the diagnostic's experimental items are answered, the numbers in the manuscript now come from tuned, seeded, auditable runs, and the headline is a *regime-dependent* advantage — **won at both ends of the noise axis** (light noise, and heavy corruption of the signal where it grows with p), strong under label noise, real linear regression and graphs, absent against a tuned AdamW only in the intermediate gradient-noise band. The one comparison that flips under baseline tuning is quantified rather than hidden: §9.3.2 reports the shared-default-step-size arm alongside the tuned one (38/42 vs 17/42, differing in exactly the 21 intermediate-band cells), so a referee can see that both ends of the noise axis behave identically under either protocol.
