# Pre-submission diagnostic: Risk-Optimal Momentum Weights (SR-Adam)

Oct 17, 2026 · @Mohammad Arashi

*Paper: Arashi & Amintoosi, Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions (sn-article.tex). Mode B, pre-submission. No target venue was given, so I assumed Springer journals because the file uses sn-jnl.cls. I assumed no prior rejections. I could not assess the three \\input tables (tab\_regression, tab\_classification, tab\_noise), the figures, the .bib file or the GitHub repository, because none were provided.*

## Part One: Scientific and specialized review

The core estimation theory is correct and publishable. Several side claims are wrong or overstated, and the optimization evidence is the weak point.

**What I verified and found correct**

- The master identity (Theorem 1) matches a 4,000,000-draw simulation in a heteroscedastic p=6 case: -1.930 simulated against -1.929 from the formula. I also re-derived the window by hand.
- The two-fold identity (Corollary 3) has its sign change at sigma-hat squared = 2 sigma squared. At p=10 the risk difference is -1.22 at 1.9 and +1.34 at 2.1.
- The proofs of the positive-part theorem, the elliptical theorem, the displacement lemma and the convergence theorem are correct as written, including the optimal step-size algebra.
- The closed form phi\_t(0.9) matches Table 2 at t=2, 3, 5, 10 and 20.
- Table 1 ties out: the fixed-weight risk is 0.01 + 0.81 snr, and the oracle risk is snr/(1+snr).
- The paired differences, percentage claims and the 1.86-per-doubling figure in Tables 5 and 6 all recompute.

**Scientific novelty**

- The core idea is that the Adam direction is a deterministic-weight shrinkage estimator, and the paper replaces that weight with a positive-part Stein weight. Lemma 1 is correct and the framing is clear.
- The distinctive pieces are the closed-form bias of v-hat minus m-hat squared (Theorem 7), the loss-weighting-decides-the-dimension result (Corollary 2(iii)), and the contraction factor (nu-2)/nu.
- Related prior work is visible: your own arXiv preprint 2602.01777 (Feb 2026) has the same title stem, the same SR-Adam name and the same idea. In the version I could read, it claims the estimator is minimax-optimal and reports large-batch gains on CIFAR. This manuscript concludes that no large-batch advantage is supported (Section 8.6) and makes no minimax claim. A referee who finds the preprint will see the contradiction.
- I also found Balles and Hennig (ICML 2018), Dissecting Adam, which studies Adam through gradient variance. I did not read it, so check whether it anticipates the v-hat minus m-hat squared statistic. I could not verify AdaBelief-type squared-deviation second moments in this session. I name them only as something to check.
- The comparison set is only AdamW, AMSGrad and SGD with momentum, which is thin for the claim.

**Mathematical and theoretical soundness**

1. **Proven error.** Section 3, third paragraph after Corollary 2: "tolerates the variance being understated by close to a factor of two" is backwards. By Corollaries 2(iv) and 3, dominance requires sigma-hat squared below 2 sigma squared, so the tolerance is for the variance being overstated. Section 5 and Remark 5.1 already have it the right way round.
2. **Proven error.** Remark 5.2 says pooling "leaves the mean unchanged", and the proof of Proposition 10 uses it coordinatewise. It holds only for the coordinate average. In fact E\[sigma-hat squared\_j\] = (1-omega) s\_j squared + omega times the mean variance, which I confirmed by calculation (0.567 against a true 0.5 at omega=0.1). The Jensen bound therefore gives E\[lambda\_max of Sigma-hat\] at least (1-omega) lambda\_max(Sigma) + omega times the mean variance + epsilon, not at least lambda\_max(Sigma) as stated. The 1/B order survives.
3. **Proven inconsistency.** Assumption (A5) conflicts with (A3') of Section 4. Under a shared mixing variable W, Cov(g\_j squared, g\_k squared) = sigma\_j squared sigma\_k squared Var(W) > 0 for every pair, so the sum of absolute covariances grows like p and the C/p variance bound in Theorem 8 fails. Simulated correlation between coordinatewise variance estimates is 0.21 for Student nu=6 and 0.002 for Gaussian. For nu at most 4 the fourth moment in (A5) does not exist, yet Section 8.4 and Table 4 go down to nu=3. Separately, (A4) (bounded gradients) cannot coexist with Gaussian or Student noise, so no single object satisfies (A3) and (A4).
4. **Ambiguity requiring clarification.** (A5) assumes a stationary mean, which a training run violates. v-hat uses beta2=0.999 and remembers about 1,000 steps, while m-hat squared uses about 10. In a toy run with the gradient mean decaying as exp(-t/100) and unit noise variance, the corrected estimator overstates the variance by 2.4 times at t=20, 7.8 times at t=100 and 4.8 times at t=300. That is outside the factor-two margin of Corollary 3. This is my construction, not your setting, but Sections 8.1 and 8.6 contain no experiment that measures sigma-hat squared over sigma squared during optimization.
5. **Ambiguity (claim stronger than the proof).** Theorem 4 (positive part) needs A = Sigma and D = Sigma inverse. Algorithm 1 uses A = D inverse = Sigma-hat with the positive part. Then the standardized observation has covariance diag(r\_j), spherical symmetry is lost, and the pairing argument no longer gives a nonnegative conditional mean. The statement in Section 6 that the risk calculation "applies verbatim at each step" is not established.
6. **Ambiguity.** Corollary 6 evaluates the window only at mu=0. Numerically the weighted mean of W rises with the norm of mu (nu=5, p=30: 1.00, 1.05, 1.18, 1.36, 1.58 at norms 0, 2, 5, 10, 30), so mu=0 looks like the worst case. That is not proved, and the abstract says "dominance window".
7. **Improvement suggestion (convergence).** First, p\_max = 1/epsilon gives kappa = (G\_inf + epsilon)/epsilon, about 10^8 times G\_inf for epsilon = 10^-8, so the bound is numerically vacuous. Second, B\_t must be defined piecewise, because t=1, the warm-up steps and gated blocks use other weights; the theorem says "run Algorithm 1" and the Remark patches this afterward. Third, the floor comes from the norm of an expectation being at most the expectation of the norm, so "exactly the quantity Lemma 2 bounds" overstates: the bias can be much smaller than the displacement. Finally, the theorem never uses the risk reduction, so it cannot show a benefit of shrinkage.

**Method and robustness.** The gate and pooling are heuristics, and the paper labels them as such. Robustness to correlated coordinates is disclosed in the Limitations but not tested.

**Decision function and classification.** Does not apply.

**Experimental design**

- Section 8.5 compares optimizers at a common step size with no tuning. SGD with momentum scores 0.233 on clean logistic regression, against AdamW's 0.366, which suggests AdamW is mis-set.
- The fixed-weight baseline is exactly beta1. The fair comparison is AdamW with a beta1 grid (including beta1=0) and a step-size grid per noise level.
- Under label noise, SR-Adam's gain may be implicit regularization (a lower effective step) rather than better gradient estimation. Early-stopping and weight-decay controls are missing, and it is unclear whether the test labels are noisy.
- The abstract says "gains that grow with the gradient-noise level", and Table 5 contradicts it. On the quadratic the SR-Adam/AdamW terminal ratios are 0.34, 0.47, 0.57, 0.66, 0.73, 0.77 as tau rises, so the relative gain falls. On the logistic problem the relative gain falls from 36% to 7% as the flip rate rises. Your own Section 8.5 text says the advantage is largest where the noise is smallest relative to the curvature.
- Table 8 gaps divided by p are 0.0012, 0.0018, 0.0009, 0.0011, 0.0011, 0.0012. The quadratic gap grows with p only because the objective scales with p, so "advantage grows with p" is not shown.
- Numbers that do not reconcile: quadratic tau=1 at p=400 gives 0.686 in Table 5, but Table 8 gives 0.222 at p=200 and 0.590 at p=500. Logistic with 200 features at flip 0.2 gives 0.123 in Table 5 and 0.202 in Table 8. "23 of 24 settings" does not match the 18 cells in Table 8.
- Stale or dangling statements: Remark 3.1 promises gate-opening frequencies in Section 9, and none appear. The Remark after Corollary 6 promises empirical windows at several signal levels in Section 8, and Table 4 has no signal level. The Limitations still say "one small backbone and two datasets", and Section 10 refers to "the backbone used here". Neither matches this version.
- The header comment says to search for the macro torerun before submission. The macro is defined but unused in the text I have, so the three \\input tables must be checked.

**Statistical analysis**

- Monte Carlo standard errors are not reported.
- Section 8.5 uses 3 or 5 seeds, with t as large as 382.
- "Statistically distinguishable" in Section 8.6 is wrong for B=8 (t=2.7) and B=32 (t=1.9). Both fall below the 5% two-sided critical value of 2.78 at 4 degrees of freedom.
- The win-rate figure counts ties as wins and uses no test across datasets.

**Complexity.** The three extra reductions per block are stated; no wall-clock timing is reported.

**Writing.** The paper is clear and its limitations are candid. Notation collisions to fix: a vs a\_j; d\_t vs d\_j; gamma (gate margin vs the proof of Theorem 4); nu (degrees of freedom vs the sphere measure); lambda (weight decay vs lambda\_max); B (batch size vs B\_t); p vs p\_min and p\_max; k (weight vs index). The Acknowledgements are empty.

## Part Two: Revisions

Ten revisions are mandatory because each one either contradicts the paper's own results or leaves a claim unsupported.

**(a) Mandatory**

| # | Location | Problem | Label | Fix | New analysis | Effort / priority |
| --- | --- | --- | --- | --- | --- | --- |
| a1 | Sec. 3, third paragraph | Understated vs overstated | Proven error | Say "overstated" | No | Low / Critical |
| a2 | Remark 5.2, Prop. 10 proof | Pooling is biased coordinatewise | Proven error | Use E\[sigma-hat squared\_j\] = (1-omega) s\_j squared + omega times the mean variance; restate the Jensen bound | No | Low / Critical |
| a3 | (A5), Thm 8, Sec. 4-5 | Incompatible with (A3'); nu at most 4 has no fourth moment; (A4) vs Gaussian | Proven error | Replace (A5) with a condition allowing a common W, or restrict Thm 8 to Gaussian and give the variance under W; separate the bounded-gradient setting | Yes (simulation) | Medium / Critical |
| a4 | Thm 4 vs Alg. 1 | Positive part with plug-in scale is uncovered | Ambiguity | Prove it, or call it heuristic and add a Monte Carlo over the spread of r\_j | Yes | Medium-High / Critical |
| a5 | (A5), Thm 7 | Mean drift inflates the variance estimate beyond the factor-two margin | Ambiguity | Log the variance ratio along real runs; consider matched windows for m and v; state the failure regime | Yes | High / Critical |
| a6 | Abstract, Table 5 | "Gains grow with noise" is contradicted | Proven error | Reword to absolute vs relative; align with the Sec. 8.5 text | No | Low / Critical |
| a7 | Sec. 8.5, Sec. 9 | Untuned baselines; regularization confound | Ambiguity | beta1 and step-size grids; early-stopping and weight-decay controls; state test-label noise | Yes | High / Critical |
| a8 | Tables 5 and 8, "23/24" | Irreconcilable numbers; no per-p normalization | Ambiguity | Reconcile configs; report gap relative to the loss; fix the count | Yes (rerun) | Medium / Critical |
| a9 | Remark 3.1, Remark after Cor. 6, Sec. 10, Limitations | Dangling and stale statements | Proven error | Delete or supply the missing results | Partly | Low-Medium / Critical |
| a10 | arXiv:2602.01777 | Preprint claims (large-batch gains, minimax) conflict with this paper | Ambiguity | Cite and disclose in the cover letter; update the arXiv version | No | Low / Critical |

**(b) Important**

| # | Location | Problem | Label | Fix | New analysis | Effort / priority |
| --- | --- | --- | --- | --- | --- | --- |
| b1 | Thm 9, Lemma 2 | kappa of order 1/epsilon is vacuous; B\_t piecewise; floor from Jensen | Improvement | Define B\_t piecewise; report numeric constants; bound the bias directly; drop "exactly" | Some | Medium-High / High |
| b2 | Thm 5, Cor. 6, Table 4 | Window proved only at mu=0; Table 4 has no signal level | Ambiguity | Prove the weighted mean of W is at least 1/E\[1/W\] for all mu; add a signal and rho sweep | Yes | Medium / High |
| b3 | Sec. 8.5, 8.6, win-rate figure | Seeds, tests, ties, "distinguishable" | Improvement | Wilcoxon with Holm across datasets; report SEs; fix the wording | Light | Low-Medium / High |
| b4 | Sec. 9 | Scale and baselines | Improvement | One residual-network or CIFAR run (the preprint had one); tuned and modern baselines | Yes | High / High |
| b5 | Introduction | Novelty positioning | Improvement | Position against Balles and Hennig and similar work | No | Low / High |
| b6 | Alg. 1, Table 2 | Clipping at zero breaks exact unbiasedness (0.9492 vs 0.9458 at t=40) | Improvement | State it; report the bias | No | Low / Medium |

**(c) Minor.** Resolve the notation collisions listed in Part One. Fill the Acknowledgements. Check .bib completeness. Add per-step timing.

**Phased action plan**

1. Fix a1, a2, a6 and a9 (text only, about a day).
2. Do a3, a4, b1 and b2 (theory).
3. Do a5, a7, a8, b3 and b4. A corrected estimator or tuned baselines change every downstream table, so these come after the theory.
4. Do a10, b5, b6, the notation fixes, and a final search for the torerun macro.

## Part Three: Scores

The paper needs major revisions plus new experiments. Scores are diagnostic, not predictions, and each maps to the revisions in Part Two.

| Criterion | Score (1-10) | Basis |
| --- | --- | --- |
| Novelty | 6 | Clean identity and exact bias; adjacent work unchecked |
| Mathematical correctness | 7 | Core results correct; items 1-3 and 5 are wrong or overstated |
| Methodological soundness | 6 | Plug-in and drift gaps |
| Experimental validation | 4 | Untuned baselines, irreconcilable numbers |
| Statistical reliability | 4 | 3-5 seeds, no SEs, ties counted as wins |
| Comparison with state of the art | 4 | Three baselines |
| Writing quality | 7 | Clear; stale remnants |
| Reproducibility | 7 | Code declared; not verified by me |
| Overall scientific contribution | 5.5 |  |

**Publication potential:** solid mid-tier. A top-tier statistics or ML venue is not realistic without a larger-scale study.

**Primary obstacle:** the theory covers the idealized estimator while the experiments use the plug-in, positive-part, drifting-mean one, and the optimization evidence is thin and untuned.

## Part Four: Five journals

Statistics and Computing is the closest fit among five Springer journals. Metrics come from each journal's own Springer page, fetched on 2026-10-17. Quartiles need Clarivate or Scimago and are not verified for any of the five. I did not name example recent articles, since I did not confirm any.

| Journal (Springer) | Impact factor (2025, journal page) | Indexing | Fit and trade-offs | Revision needed |
| --- | --- | --- | --- | --- |
| [Statistics and Computing](https://link.springer.com/journal/11222) | 1.8; 5-year 2.1 | SCIE, Scopus, zbMATH | Closest fit (statistical computation, ML). The page lists a median of 3 days to first decision and a move to fully open access on 1 Jan 2027 | Major |
| Machine Learning | Not verified (Springer page refused the fetch, HTTP 429) | Not verified | Good theory-plus-optimization fit; demands strong empirical rigor | Major+ |
| [Computational Optimization and Applications](https://link.springer.com/journal/10589) | 2.5; 5-year 2.6; CiteScore 4.0 | SCIE, Scopus, EI | Wants convergence plus computational experiments; the floor-type theorem is weak for it | Major+ |
| [Annals of the Institute of Statistical Mathematics](https://link.springer.com/journal/10463) | 1.4; 5-year 1.1 | SCIE, Scopus, zbMATH | Shrinkage theory fits; an optimizer paper is unusual there | Major (theory-heavy rewrite) |
| [Neural Computing and Applications](https://link.springer.com/journal/521) | Not on the page | Scopus, EI, INSPEC, others | Lower bar, broad scope; median 14 days to first decision | Moderate |

## Part Five: Acceptance estimates

Neural Computing and Applications has the highest estimated chance, and the realistic range everywhere else is low until the revisions are done. These are conditional expert judgments, not official statistics or guarantees. I did not find published acceptance rates, so the ranges are wide.

| Journal | Quartile and year | Topic alignment | Acceptance (current) | Acceptance (after revision) | Primary barrier |
| --- | --- | --- | --- | --- | --- |
| Statistics and Computing | Not verified | High | 8-18% | 30-45% | Estimator-versus-theory gap; untuned baselines |
| Machine Learning | Not verified | High | 3-10% | 15-30% | Empirical scale and rigor |
| Computational Optimization and Applications | Not verified | Medium | 3-8% | 12-25% | Weak convergence guarantee |
| Annals of the Institute of Statistical Mathematics | Not verified | Medium | 8-15% | 25-40% | Topic fit |
| Neural Computing and Applications | Not verified | Medium | 20-35% | 45-65% | Experimental rigor |

The after-revision column assumes revisions a1 to a10 are done; the current column assumes none are. They weigh scope, rigor and desk-rejection risk. The preprint overlap adds some risk everywhere unless it is disclosed.

## Part Six: Summary and sources

The paper is worth pursuing, but it needs theory fixes and a rebuilt optimization study before submission.

1. **Worth pursuing?** Yes. The identity, the exact variance bias and the loss-weighting result are correct and publishable.
2. **Three highest-impact revisions.** Make the theory match the algorithm (a3 to a5). Redo the optimization experiments with tuned baselines, honest scaling and reconciled numbers (a7, a8). Fix the abstract and the stale text (a6, a9).
3. **To reproduce or validate.** The variance ratio along real training runs, Table 8 against Table 5, and the three \\input tables (not seen).
4. **Is the innovation claim defensible?** Yes for the estimation-theory contribution. For "better optimizer", only in the regimes shown after the baseline fix.
5. **New result or experiment needed?** Yes: positive-part dominance under plug-in scale (or an honest downgrade), and one larger-scale run.
6. **Likely rejection factors.** The preprint contradiction, untuned baselines, the vacuous floor, and the (A5) versus (A3') inconsistency.
7. **Plan.** The four phases in Part Two.

**Sources**

- [Springer: Statistics and Computing](https://link.springer.com/journal/11222)
- [Springer: Annals of the Institute of Statistical Mathematics](https://link.springer.com/journal/10463)
- [Springer: Computational Optimization and Applications](https://link.springer.com/journal/10589)
- [Springer: Neural Computing and Applications](https://link.springer.com/journal/521)
- [arXiv:2602.01777, Stein-Rule Shrinkage for Stochastic Gradient Estimation in High Dimensions](https://arxiv.org/html/2602.01777v2)
- [Balles and Hennig, Dissecting Adam, PMLR v80](https://proceedings.mlr.press/v80/balles18a.html) (found, not read)
