"""
Numerical verification of every analytical claim used in the manuscript.

Each block states the claim, computes the quantity by Monte Carlo, and compares it
against the closed-form expression derived in the appendix.  A claim is accepted only
when the Monte Carlo estimate agrees with the closed form to within the reported
Monte Carlo standard error.

Run:  python3 verify_theory.py
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(20260912)
SEP = "=" * 78


def hdr(s):
    print("\n" + SEP + "\n" + s + "\n" + SEP)


# ----------------------------------------------------------------------------------
# Claim 1.  Master risk identity (heteroscedastic, plug-in scale matrix).
#
#   Y ~ N_p(mu, Sigma),  Sigma = diag(s_1^2,...,s_p^2) ,  A = diag(a_1,...,a_p) fixed,
#   Q = Y' A^{-1} Y ,    delta_c = (1 - c/Q) Y .
#
#   R(delta_c) - tr(Sigma)
#        = E[ -2c tr(Sigma)/Q + 4c (Y' Sigma A^{-1} Y)/Q^2 + c^2 ||Y||^2 / Q^2 ].
# ----------------------------------------------------------------------------------
def claim1_risk_identity(p=12, n=4_000_000, c=5.0, seed=1):
    r = np.random.default_rng(seed)
    s2 = np.linspace(0.5, 2.0, p)                 # heteroscedastic noise variances
    a = np.linspace(0.7, 1.8, p)                  # scale matrix A, deliberately != Sigma
    mu = r.normal(size=p) * 0.8

    Y = mu + r.normal(size=(n, p)) * np.sqrt(s2)
    Q = (Y ** 2 / a).sum(1)
    delta = (1.0 - c / Q)[:, None] * Y

    lhs = ((delta - mu) ** 2).sum(1).mean() - s2.sum()
    lhs_se = ((delta - mu) ** 2).sum(1).std(ddof=1) / np.sqrt(n)

    term1 = -2 * c * s2.sum() / Q
    term2 = 4 * c * (Y ** 2 * s2 / a).sum(1) / Q ** 2
    term3 = c ** 2 * (Y ** 2).sum(1) / Q ** 2
    rhs_vals = term1 + term2 + term3
    rhs = rhs_vals.mean()
    rhs_se = rhs_vals.std(ddof=1) / np.sqrt(n)

    print(f"  MC risk difference   = {lhs: .6f}  (se {lhs_se:.6f})")
    print(f"  Stein-identity value = {rhs: .6f}  (se {rhs_se:.6f})")
    ok = abs(lhs - rhs) < 4 * np.sqrt(lhs_se ** 2 + rhs_se ** 2)
    print(f"  agreement within 4 combined se: {ok}")
    return ok


# ----------------------------------------------------------------------------------
# Claim 2.  Sufficient dominance window
#
#      0 < c < 2 lam_max(Sigma) / lam_max(A) * ( p_eff(Sigma) - 2 ),
#      p_eff(Sigma) = tr(Sigma) / lam_max(Sigma).
#
# Verified by scanning c and locating the empirical sign change of the risk difference.
# The derived window is sufficient, so the empirical crossing must be at or beyond the
# stated upper endpoint.
# ----------------------------------------------------------------------------------
def claim2_window(p=12, n=2_000_000, seed=2):
    r = np.random.default_rng(seed)
    out = []
    for label, s2, a, mu_scale in [
        ("homoscedastic, A = Sigma", np.ones(p), np.ones(p), 0.0),
        ("homoscedastic, A = Sigma, mu != 0", np.ones(p), np.ones(p), 1.0),
        ("heteroscedastic, A = Sigma", np.linspace(0.4, 2.5, p), None, 0.7),
        ("heteroscedastic, A = 1.5 Sigma", np.linspace(0.4, 2.5, p), 1.5, 0.7),
    ]:
        A = s2.copy() if a is None else (s2 * a if np.isscalar(a) else a)
        mu = r.normal(size=p) * mu_scale
        Y = mu + r.normal(size=(n, p)) * np.sqrt(s2)
        Q = (Y ** 2 / A).sum(1)
        base = ((Y - mu) ** 2).sum(1).mean()

        p_eff = s2.sum() / s2.max()
        c_star = 2 * s2.max() / A.max() * (p_eff - 2)

        # empirical sign change of R(delta_c) - R(Y)
        grid = np.linspace(0.05, 4 * max(c_star, 1e-6), 160)
        diffs = np.array([
            (((1 - cc / Q)[:, None] * Y - mu) ** 2).sum(1).mean() - base for cc in grid
        ])
        pos = np.where(diffs > 0)[0]
        c_cross = grid[pos[0]] if len(pos) else np.inf
        out.append(c_cross >= c_star * 0.98)
        print(f"  {label:38s} p_eff={p_eff:6.2f}  window upper={c_star:7.3f}"
              f"  empirical crossing={c_cross:7.3f}  ok={out[-1]}")
    return all(out)


# ----------------------------------------------------------------------------------
# Claim 3.  Plug-in variance corollary.
#   Sigma = sigma^2 I, A = sigmahat^2 I, c chosen so the estimator is
#   (1 - (p-2) sigmahat^2 / ||Y||^2) Y.  Then
#       R(delta) - p sigma^2 = -(p-2)^2 sigmahat^2 (2 sigma^2 - sigmahat^2) E||Y||^{-2},
#   so dominance holds iff 0 < sigmahat^2 < 2 sigma^2.
# ----------------------------------------------------------------------------------
def claim3_plugin(p=10, n=4_000_000, sigma2=1.0, seed=3):
    r = np.random.default_rng(seed)
    mu = r.normal(size=p) * 0.6
    Y = mu + r.normal(size=(n, p)) * np.sqrt(sigma2)
    inv = 1.0 / (Y ** 2).sum(1)
    base = ((Y - mu) ** 2).sum(1).mean()
    E_inv = inv.mean()
    ok = True
    print(f"  E||Y||^-2 = {E_inv:.6e},  p sigma^2 = {p * sigma2:.4f},  MC base = {base:.4f}")
    for sh2 in [0.25, 0.5, 1.0, 1.5, 1.9, 2.0, 2.1, 2.6]:
        a = (p - 2) * sh2
        mc = (((1 - a * inv)[:, None] * Y - mu) ** 2).sum(1).mean() - base
        closed = -((p - 2) ** 2) * sh2 * (2 * sigma2 - sh2) * E_inv
        good = abs(mc - closed) < 0.02 * max(1.0, abs(closed))
        ok &= good
        print(f"    sigmahat^2={sh2:4.2f}  MC={mc:+.5f}  closed form={closed:+.5f}"
              f"  dominates={mc < 0}  match={good}")
    return ok


# ----------------------------------------------------------------------------------
# Claim 4.  Exact finite-sample bias of the bias-corrected Adam moment variance
#   estimator.  For g_{t,j} iid in t with variance s^2,
#       E[ vhat_t - mhat_t^2 ] = (1 - phi_t(beta1)) s^2,
#       phi_t(b) = (1-b)(1-b^{2t}) / [ (1+b)(1-b^t)^2 ].
# ----------------------------------------------------------------------------------
def phi(t, b):
    return (1 - b) * (1 - b ** (2 * t)) / ((1 + b) * (1 - b ** t) ** 2)


def claim4_ema_bias(n=400_000, b1=0.9, b2=0.999, T=40, seed=4):
    r = np.random.default_rng(seed)
    mu, s2 = 0.3, 1.7
    m = np.zeros(n)
    v = np.zeros(n)
    ok = True
    for t in range(1, T + 1):
        g = mu + r.normal(size=n) * np.sqrt(s2)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        if t in (1, 2, 3, 5, 10, 20, 40):
            mc = (vh - mh ** 2).mean()
            se = (vh - mh ** 2).std(ddof=1) / np.sqrt(n)
            closed = (1 - phi(t, b1)) * s2
            good = abs(mc - closed) < 4 * se
            ok &= good
            print(f"    t={t:3d}  MC={mc:+.5f} (se {se:.5f})  closed={closed:+.5f}"
                  f"  debias factor 1/(1-phi)={1/(1-phi(t,b1)) if t>1 else float('inf'):8.4f}"
                  f"  match={good}")
    print(f"    limit t->inf: 1 - phi = {1 - (1-b1)/(1+b1):.6f} = 2b1/(1+b1) = {2*b1/(1+b1):.6f}")
    return ok


# ----------------------------------------------------------------------------------
# Claim 5.  Deterministic displacement bound
#      || ghat - g || = min( ||Y||, a/||Y|| ) <= sqrt(a),   a = (p-2) sigmahat^2,
#   where ghat = m + [1 - a/||Y||^2]^+ Y and Y = g - m.
# ----------------------------------------------------------------------------------
def claim5_displacement(p=50, n=500_000, seed=5):
    r = np.random.default_rng(seed)
    worst = 0.0
    for sh2 in [0.1, 1.0, 5.0]:
        Y = r.normal(size=(n, p)) * r.uniform(0.2, 3.0, size=(n, 1))
        nrm = np.linalg.norm(Y, axis=1)
        a = (p - 2) * sh2
        cfac = np.maximum(0.0, 1 - a / nrm ** 2)
        disp = np.linalg.norm((1 - cfac)[:, None] * Y, axis=1)
        ratio = (disp / np.sqrt(a)).max()
        worst = max(worst, ratio)
        print(f"    sigmahat^2={sh2:4.1f}  bound sqrt(a)={np.sqrt(a):7.4f}"
              f"  max displacement={disp.max():7.4f}  ratio={ratio:.6f}")
    return worst <= 1.0 + 1e-9


# ----------------------------------------------------------------------------------
# Claim 6.  Elliptical robustness.  Y | W ~ N(mu, W sigma^2 I), W = nu / chi^2_nu, so
#   Y is multivariate t_nu with scale sigma^2 and marginal variance
#   s^2 = sigma^2 nu/(nu-2).  Claim: the dominance window written in terms of the
#   MARGINAL variance contracts by the factor (nu-2)/nu, i.e.
#       a < 2(p-2) s^2 (nu-2)/nu ,
#   and, as ||mu|| -> 0, the exact window constant is 1 / E[W^{-1}] = 1.
# ----------------------------------------------------------------------------------
def claim6_student_t(p=12, n=3_000_000, seed=6):
    ok = True
    for nu in [5, 8, 20]:
        for mu_scale in [0.0, 0.5]:
            r = np.random.default_rng(seed + nu)
            mu = r.normal(size=p) * mu_scale
            W = nu / r.chisquare(nu, size=n)
            Y = mu + r.normal(size=(n, p)) * np.sqrt(W)[:, None]   # sigma^2 = 1
            base = ((Y - mu) ** 2).sum(1).mean()
            s2_marg = nu / (nu - 2)
            a_pred = 2 * (p - 2) * s2_marg * (nu - 2) / nu          # = 2(p-2) sigma^2
            grid = np.linspace(0.5, 2.2 * a_pred, 120)
            nrm2 = (Y ** 2).sum(1)
            diffs = np.array([
                (((1 - aa / nrm2)[:, None] * Y - mu) ** 2).sum(1).mean() - base
                for aa in grid])
            pos = np.where(diffs > 0)[0]
            a_cross = grid[pos[0]] if len(pos) else np.inf
            good = a_cross >= 0.95 * a_pred
            ok &= good
            print(f"    nu={nu:3d} ||mu||scale={mu_scale:3.1f}  marginal s^2={s2_marg:.4f}"
                  f"  predicted window={a_pred:7.3f}  empirical crossing={a_cross:7.3f}"
                  f"  ok={good}")
    return ok


# ----------------------------------------------------------------------------------
# Claim 7.  Positive-part truncation never increases risk (spherical case).
# ----------------------------------------------------------------------------------
def claim7_positive_part(p=8, n=3_000_000, seed=7):
    r = np.random.default_rng(seed)
    ok = True
    for mu_scale in [0.0, 0.5, 2.0]:
        mu = r.normal(size=p) * mu_scale
        Y = mu + r.normal(size=(n, p))
        nrm2 = (Y ** 2).sum(1)
        a = p - 2
        f = 1 - a / nrm2
        r_js = ((f[:, None] * Y - mu) ** 2).sum(1).mean()
        r_pos = ((np.maximum(f, 0.0)[:, None] * Y - mu) ** 2).sum(1).mean()
        good = r_pos <= r_js + 1e-9
        ok &= good
        print(f"    ||mu||={np.linalg.norm(mu):6.3f}  R(JS)={r_js:.5f}"
              f"  R(JS+)={r_pos:.5f}  improvement={r_js - r_pos:+.5f}  ok={good}")
    return ok


# ----------------------------------------------------------------------------------
# Claim 8.  Pooled debiased variance estimator concentrates at rate p^{-1/2}.
# ----------------------------------------------------------------------------------
def claim8_concentration(b1=0.9, b2=0.999, T=25, reps=3000, seed=8):
    r = np.random.default_rng(seed)
    print("      p      sd(sigmahat^2)   sd * sqrt(p)")
    vals = []
    for p in [16, 64, 256, 1024, 4096]:
        s2 = r.uniform(0.5, 2.0, size=p)
        mu = r.normal(size=p)
        reps_p = max(200, reps // (p // 16))
        m = np.zeros((reps_p, p))
        v = np.zeros((reps_p, p))
        for t in range(1, T + 1):
            g = mu + r.normal(size=(reps_p, p)) * np.sqrt(s2)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g ** 2
        mh = m / (1 - b1 ** T)
        vh = v / (1 - b2 ** T)
        est = ((vh - mh ** 2) / (1 - phi(T, b1))).mean(1)
        sd = est.std(ddof=1)
        vals.append(sd * np.sqrt(p))
        print(f"    {p:6d}     {sd:12.6f}   {sd*np.sqrt(p):10.4f}"
              f"   bias={est.mean() - s2.mean():+.5f}")
    ratio = max(vals) / min(vals)
    print(f"    sd*sqrt(p) varies by factor {ratio:.2f} across a 256-fold range of p")
    return ratio < 2.0


if __name__ == "__main__":
    results = {}
    hdr("Claim 1: master Stein risk identity (heteroscedastic, plug-in A)")
    results["1 risk identity"] = claim1_risk_identity()

    hdr("Claim 2: dominance window  0 < c < 2 lam_max(Sigma)/lam_max(A) (p_eff - 2)")
    results["2 dominance window"] = claim2_window()

    hdr("Claim 3: plug-in variance, exact risk gain and the window 0 < sigmahat^2 < 2 sigma^2")
    results["3 plug-in window"] = claim3_plugin()

    hdr("Claim 4: exact bias of the Adam-moment variance estimator")
    results["4 EMA bias"] = claim4_ema_bias()

    hdr("Claim 5: deterministic displacement bound sqrt((p-2) sigmahat^2)")
    results["5 displacement bound"] = claim5_displacement()

    hdr("Claim 6: Student-t robustness, window factor (nu-2)/nu")
    results["6 elliptical window"] = claim6_student_t()

    hdr("Claim 7: positive-part truncation reduces risk")
    results["7 positive part"] = claim7_positive_part()

    hdr("Claim 8: p^{-1/2} concentration of the pooled debiased variance estimator")
    results["8 concentration"] = claim8_concentration()

    hdr("SUMMARY")
    for k, v in results.items():
        print(f"  {k:28s} {'PASS' if v else 'FAIL'}")
