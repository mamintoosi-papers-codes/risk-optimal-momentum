# Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions

This repository contains the implementation, experiment outputs, and manuscript for **SR-Adam**, a stochastic-gradient direction estimator that replaces Adam's fixed first-moment weight with a positive-part Stein shrinkage rule.

**Manuscript:** *Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions*  
**Authors:** Mohammad Arashi and Mahmood Amintoosi  
**Manuscript source:** `paper/sn-article.tex`  
**Bibliography:** `paper/sample.bib`

## What the paper studies

Adam's bias-corrected first moment can be written as a convex combination of the current mini-batch gradient and the historical average. SR-Adam treats that direction as an estimator of the population gradient and chooses its shrinkage weight using a conditional quadratic-risk analysis.

The method and paper cover:

- A conditional risk-dominance result for heteroscedastic gradient noise, estimated scale matrices, and diagonal quadratic loss weights.
- Positive-part shrinkage and the role of nominal versus effective dimension under different loss metrics.
- Elliptically contoured noise, including multivariate Student noise.
- The finite-iteration bias in an exponential-moving-average variance estimate, its debiasing correction under a stationary gradient mean, and analysis of bias under a drifting mean.
- A convergence analysis for the deliberately biased direction, with an explicit residual floor and qualifications on when it scales with batch size.
- Monte Carlo checks of theoretical constants and optimisation experiments on real tabular regression/classification tasks, citation-graph node classification, and controlled-noise settings.

The empirical findings are **regime-dependent**, not a claim that SR-Adam universally outperforms AdamW. The paper reports settings where it helps and others—particularly some intermediate-noise and clean-classification settings—where tuned AdamW matches or performs better. The experiments do not establish performance on large-scale CNNs or transformers.

## Repository layout

| Path | Purpose |
| --- | --- |
| `code/` | SR-Adam implementation, unit tests, theory checks, experiment drivers, and table/figure-generation scripts. |
| `out/` | Machine-readable experiment and Monte Carlo outputs used by the manuscript and analysis scripts. |
| `figs/` | Generated figures from the simulation and experiment pipeline. |
| `paper/` | Springer Nature LaTeX source, bibliography, generated tables, manuscript figures, and compiled PDF. |
| `scripts/phaseD.bat` | Ordered full experiment rerun and refresh of manuscript-facing outputs on Windows. |
| `scripts/phaseE.bat` | Manuscript checks, placeholder audit, figure-freshness checks, and LaTeX/BibTeX build on Windows. |

The repository also contains supporting reports and helper scripts. Manuscript tables are generated from stored outputs; avoid hand-editing generated table cells.

## Environment

The experiment drivers use Python and scientific-computing / machine-learning packages, including NumPy, SciPy, Matplotlib, and PyTorch; graph experiments additionally require the relevant graph-learning dependencies. Install the repository's specified dependencies from `requirements.txt` in an environment compatible with your PyTorch and CUDA installation.

For example, on a Unix-like shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the local hardware and driver when the default package installation does not provide the required accelerator support.

## Checks and manuscript build

From a Windows command prompt at the repository root:

```bat
scripts\phaseE.bat check
```

This runs the manuscript-facing checks without rebuilding the PDF. The checks include generated-table provenance, an audit for unresolved `\\torerun{...}` placeholders in the manuscript body, and a byte-level comparison between generated figures and the copies included in `paper/`.

To run the checks and rebuild the manuscript (requires `pdflatex` and `bibtex` on `PATH`):

```bat
scripts\phaseE.bat
```

The full experiment pipeline is orchestrated by `scripts\phaseD.bat`. It reruns theory checks, simulations, tuned baseline comparisons, tabular tasks, graph pilots, figure/table generation, and statistical tests. It can be computationally expensive; use it when intentionally regenerating experiment outputs, not merely to compile the current manuscript.

The LaTeX source uses the Springer Nature `sn-jnl` class included in `paper/`. The manuscript has no unresolved experimental placeholders in its body; do not follow older instructions that describe the former CIFAR grid or ask for new CIFAR runs.

## Reproducibility notes

- The manuscript's numerical tables and figures are tied to stored machine-readable outputs and the table/figure-generation scripts in `code/`.
- Use the ordered phase-D pipeline when changing experiment code, estimator settings, baseline tuning, or simulation constants; then run phase E to check the refreshed manuscript and build the PDF.
- The manuscript's code-availability statement points to this repository. Before journal submission, ensure it is accessible to editors and reviewers (for example, by making an appropriate public or anonymized review copy available).

## Citation

If you use the method or code, cite the manuscript:

```bibtex
@article{arashi2026sradam,
  title   = {Risk-Optimal Momentum Weights for Stochastic Gradient Estimation in High Dimensions},
  author  = {Arashi, Mohammad and Amintoosi, Mahmood},
  journal = {Manuscript under review},
  year    = {2026},
  note    = {Code repository: mamintoosi-papers-codes/risk-optimal-momentum}
}
```
