"""
SR-Adam: risk-optimal shrinkage weights for the stochastic gradient, with
effective-dimension gating.

Background
----------
Write mhat_{t-1} for the bias-corrected exponential moving average of the raw gradients
up to step t-1.  A short calculation shows that Adam's bias-corrected momentum satisfies,
exactly,

    mhat_t = mhat_{t-1} + w_t ( g_t - mhat_{t-1} ),   w_t = (1 - beta1)/(1 - beta1^t),

so the Adam search direction is a convex combination of the current mini-batch gradient
and the historical average, with a deterministic weight fixed by the hyperparameter
beta1.  SR-Adam keeps the form and replaces the weight by the one that minimises the
conditional quadratic risk of the resulting gradient estimate, namely the positive-part
Stein-rule weight

    k_t = [ 1 - c_t / Q_t ]^+ ,
    Q_t = sum_j ( g_{t,j} - mhat_{t-1,j} )^2 / ( s2_{t,j} + eps ),
    c_t = rho ( p - 2 )        (constant = "p",    the default)
       or rho ( peff_t - 2 )   (constant = "peff", the conservative variant).

Three estimated quantities drive the rule, all built from the raw-gradient moments and
therefore fixed given the first t-1 gradients, which is what makes the conditional risk
calculation in the accompanying paper valid.

  s2_{t,j}  coordinatewise gradient-noise variance, estimated by the exactly unbiased
            statistic ( vhat_j - mhat_j^2 ) / ( 1 - phi_t(beta1) ).  The correction factor
            1 - phi_t(beta1) accounts for the sampling variance of the moving average
            itself; without it the statistic is biased downward, by the factor
            2 beta1/(1 + beta1) even in the limit of many steps.  The coordinatewise
            estimates are then shrunk toward their own pooled mean,

                s2tilde_j = (1 - omega) s2_j + omega * mean_j(s2_j),

            in the manner of a Ledoit-Wolf covariance shrinkage.  An exponential moving
            average carries an effective sample size of only about (1+beta1)/(1-beta1)
            observations, so individual coordinatewise estimates are themselves noisy and
            some land near zero.  Since Q_t divides by them, one such coordinate can
            dominate the statistic on its own and switch the shrinkage off; the pooling
            bounds the spread of the ratios of true to estimated variance, which is
            exactly the quantity the dominance window depends on.

  peff_t    effective dimension, sum_j s2_{t,j} / max_j s2_{t,j}.  Because Q_t whitens by
            the coordinatewise variances, the relevant loss is the standardised one and
            the admissible constant is governed by the nominal dimension p, not by
            peff_t; the effective dimension enters instead as a diagnostic.  A block whose
            estimated noise is concentrated in a handful of coordinates has peff_t close
            to one, which is the signature of a variance estimate one should not trust,
            and shrinkage is then switched off so the block reverts to Adam.

  rho       safety factor in (0, 1].  With exact coordinatewise variances the dominance
            window under the standardised loss is 0 < c < 2(p - 2), so rho = 1 places c at
            its midpoint, which is also the risk-minimising choice.  Writing r_j for the
            ratio of true to estimated variance in coordinate j, the window becomes
            2 ( sum_j r_j - 2 max_j r_j ), so rho = 1 tolerates the variance being
            understated by close to a factor of two on average.  Under Student-t gradient
            noise with nu degrees of freedom the window contracts by the factor
            (nu - 2)/nu, and rho = 1 remains inside it whenever nu > 4.

The preconditioner is the running maximum of the bias-corrected second moment, as in
AMSGrad.  The maximum makes the preconditioner monotone, which the convergence proof uses
to control the drift of the preconditioner between steps, and it removes a failure mode
that a one-step-delayed preconditioner suffers, namely that a coordinate whose gradient
magnitude jumps is divided by a scale the denominator has not yet seen.

Memory cost equals AMSGrad's, three buffers per parameter, or Adam's two when the running
maximum is disabled.  The scalar reductions needed for peff_t and Q_t cost two extra
passes over each gradient block.

Reference implementation accompanying the manuscript.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch.optim import Optimizer

__all__ = ["SRAdam", "ema_variance_correction", "conv_only_filter"]

_TINY = 1e-30


def ema_variance_correction(t: int, beta: float) -> float:
    """
    Return 1 - phi_t(beta), the exact multiplicative bias of vhat_t - mhat_t^2 as an
    estimator of the gradient-noise variance when gradients are serially independent,

        phi_t(beta) = (1 - beta)(1 - beta^{2t}) / [ (1 + beta)(1 - beta^t)^2 ].

    phi_1 = 1, so the correction is defined only for t >= 2.  Its limit is
    1 - (1 - beta)/(1 + beta) = 2 beta/(1 + beta).
    """
    if t < 2:
        return 0.0
    bt = beta ** t
    phi = (1.0 - beta) * (1.0 - bt * bt) / ((1.0 + beta) * (1.0 - bt) ** 2)
    return 1.0 - phi


def conv_only_filter(p: torch.Tensor) -> bool:
    """
    The layer-type rule used in the first version of this work: shrink the weights of
    two-dimensional convolutions only.  Kept so that the ad hoc rule can be compared
    against the effective-dimension gate.
    """
    return p.dim() == 4


class SRAdam(Optimizer):
    """
    Parameters
    ----------
    params : iterable
        Parameters or parameter groups, as for any torch optimiser.
    lr : float
        Step size.
    betas : (float, float)
        beta1 drives the first-moment average supplying the shrinkage target and the
        variance correction; beta2 drives the second-moment average used by the
        preconditioner and by the variance estimate.
    eps : float
        Ridge added to the preconditioner and to the whitening denominator.
    weight_decay : float
        Decoupled weight decay, applied as in AdamW.
    rho : float
        Safety factor in (0, 1] multiplying the shrinkage constant.
    omega : float
        Weight in [0, 1) of the pooled mean in the coordinatewise variance estimate.  Zero
        uses the raw coordinatewise estimates; one would pool them completely.
    constant : {"p", "peff"}
        Which dimension supplies the shrinkage constant.  "p" uses rho (p - 2), admissible
        under the standardised loss whenever the coordinatewise variances are roughly
        right, and is the default.  "peff" uses rho (peff - 2), which is smaller and is
        additionally admissible under the unweighted Euclidean loss, at the cost of
        forfeiting most of the available risk reduction on strongly heteroscedastic
        blocks.
    peff_margin : float
        Shrinkage is applied to a block only when its estimated effective dimension
        exceeds 2 + peff_margin.  The gate guards against variance estimates concentrated
        in a few coordinates.
    warmup : int
        Number of initial steps during which the plain Adam weight is used.  Must be at
        least 3, because the variance correction needs two completed moment updates.
    shrink_filter : callable or None
        Optional predicate on a parameter tensor deciding whether the block is eligible
        for shrinkage at all.  The default admits every block with at least three
        elements and leaves the decision to the effective-dimension gate.  Pass
        conv_only_filter to reproduce the layer-type rule.
    amsgrad : bool
        Use the running maximum of the bias-corrected second moment as preconditioner.
        The default, True, is the variant the convergence theorem covers.
    track_stats : bool
        Record per-block diagnostics for the most recent step.

    Notes
    -----
    When the gate is closed the update coincides with AMSGrad-flavoured AdamW step for
    step, since the weight reverts to w_t = (1 - beta1)/(1 - beta1^t).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rho: float = 1.0,
        omega: float = 0.1,
        constant: str = "p",
        peff_margin: float = 1.0,
        warmup: int = 5,
        amsgrad: bool = True,
        shrink_filter=None,
        track_stats: bool = False,
        grouping: str = "block",
        k_min: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must lie in [0,1), got {betas}")
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must lie in (0,1], got {rho}")
        if warmup < 3:
            raise ValueError("warmup must be at least 3")
        if constant not in ("p", "peff"):
            raise ValueError(f"constant must be 'p' or 'peff', got {constant}")
        if not 0.0 <= omega < 1.0:
            raise ValueError(f"omega must lie in [0,1), got {omega}")
        if grouping not in ("block", "filter", "channel"):
            raise ValueError(f"grouping must be block|filter|channel, got {grouping}")
        if not 0.0 <= k_min <= 1.0:
            raise ValueError(f"k_min must lie in [0,1], got {k_min}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rho=rho,
            omega=omega,
            constant=constant,
            peff_margin=peff_margin,
            warmup=warmup,
            amsgrad=amsgrad,
            grouping=grouping,
            k_min=k_min,
        )
        super().__init__(params, defaults)
        self.shrink_filter = shrink_filter
        self.track_stats = track_stats
        self.stats: List[Dict[str, float]] = []

    def _eligible(self, p: torch.Tensor) -> bool:
        if p.numel() < 3:
            return False
        return True if self.shrink_filter is None else bool(self.shrink_filter(p))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.track_stats:
            self.stats = []

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            rho, margin, warmup = group["rho"], group["peff_margin"], group["warmup"]
            constant = group["constant"]
            omega = group["omega"]
            amsgrad = group["amsgrad"]
            grouping = group.get("grouping", "block")
            k_min = group.get("k_min", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("SR-Adam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["mbar"] = torch.zeros_like(p)
                    state["vbar"] = torch.zeros_like(p)
                    if amsgrad:
                        state["vmax"] = torch.zeros_like(p)

                mbar, vbar = state["mbar"], state["vbar"]
                s = state["step"]          # completed updates, equal to t-1
                t = s + 1

                # ---- direction weight -------------------------------------------------
                # Adam weight, used during warm-up and whenever the gate is closed.
                w_adam = (1.0 - beta1) / (1.0 - beta1 ** t)

                # Grouping for the Stein weight: one k per block (default) or one k
                # per filter/channel (dim 0 of conv/linear weights). Bias vectors stay
                # a single block under channel/filter so that k is defined for p >= 3.
                if grouping == "block" or p.dim() < 2:
                    n_groups, group_size = 1, p.numel()
                else:
                    n_groups = p.shape[0]
                    group_size = p.numel() // max(n_groups, 1)

                if s < 1:
                    direction = g
                    peff_v = float("nan")
                    kfac_v = 1.0
                    sig2_v = float("nan")
                else:
                    bc1 = 1.0 - beta1 ** s
                    bc2 = 1.0 - beta2 ** s
                    m_prev = mbar / bc1
                    v_prev = vbar / bc2
                    y = g - m_prev

                    use_stein = (s >= warmup) and self._eligible(p)
                    if use_stein:
                        corr = ema_variance_correction(s, beta1)
                        s2 = (v_prev - m_prev * m_prev).clamp(min=0.0).div_(corr)
                        if omega > 0.0:
                            pooled = s2.mean() * omega
                            s2.mul_(1.0 - omega).add_(pooled)

                        y_g = y.reshape(n_groups, group_size)
                        s2_g = s2.reshape(n_groups, group_size)
                        s2_max = s2_g.max(dim=1).values.clamp_min(_TINY)   # (G,)
                        s2_sum = s2_g.sum(dim=1)                           # (G,)
                        peff = s2_sum / s2_max                             # (G,)
                        if constant == "peff":
                            dim = peff
                        else:
                            dim = torch.full_like(peff, float(group_size))
                        c = rho * (dim - 2.0).clamp(min=0.0)               # (G,)
                        q = (y_g * y_g / (s2_g + eps)).sum(dim=1).clamp_min(_TINY)
                        k_stein = (1.0 - c / q).clamp(min=max(k_min, 0.0))
                        if k_min > 0.0:
                            k_stein = k_stein.clamp(min=k_min)
                        gate = (peff > 2.0 + margin).to(y_g.dtype)
                        k_g = gate * k_stein + (1.0 - gate) * w_adam       # (G,)
                        direction = m_prev + (k_g.unsqueeze(1) * y_g).reshape_as(p)
                        k_for_stats = k_g
                    else:
                        k_for_stats = torch.as_tensor(w_adam, dtype=g.dtype, device=g.device)
                        direction = m_prev + k_for_stats * y

                    if self.track_stats:
                        if use_stein:
                            kfac_v = float(k_for_stats.mean())
                            peff_v = float(peff.mean())
                            sig2_v = float(s2_sum.mean() / group_size)
                        else:
                            kfac_v = float(k_for_stats)
                            peff_v = float("nan")
                            sig2_v = float("nan")

                # ---- raw-moment updates ----------------------------------------------
                mbar.mul_(beta1).add_(g, alpha=1.0 - beta1)
                vbar.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                state["step"] = t

                # ---- preconditioner ---------------------------------------------------
                vhat = vbar / (1.0 - beta2 ** t)
                if amsgrad:
                    torch.maximum(state["vmax"], vhat, out=state["vmax"])
                    denom = state["vmax"].sqrt().add_(eps)
                else:
                    denom = vhat.sqrt_().add_(eps)

                # ---- parameter update -------------------------------------------------
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(direction, denom, value=-lr)

                if self.track_stats:
                    self.stats.append(
                        dict(numel=p.numel(), step=t, shrink_weight=kfac_v,
                             peff=peff_v, sigma2=sig2_v, adam_weight=w_adam)
                    )

        return loss

    def shrinkage_report(self) -> Dict[str, float]:
        """Summarise the diagnostics recorded at the most recent step."""
        if not self.stats:
            return {}
        gated_on = [d for d in self.stats if d["peff"] == d["peff"]
                    and d["peff"] > 2.0 + self.param_groups[0]["peff_margin"]]
        return dict(
            n_blocks=len(self.stats),
            n_gated_on=len(gated_on),
            mean_weight=sum(d["shrink_weight"] for d in self.stats) / len(self.stats),
            mean_peff=(sum(d["peff"] for d in gated_on) / len(gated_on)) if gated_on else float("nan"),
        )
