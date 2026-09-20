"""
Verification of the master dominance theorem in its general weighted-loss form.

    Y ~ N_p(mu, Sigma),  Sigma = diag(sigma_j^2),  A = diag(a_j),  D = diag(d_j),
    Q = Y' A^{-1} Y,     delta_c = (1 - c/Q) Y,    loss = || . ||^2_D .

Claim (exact identity):

    R(delta_c) - tr(D Sigma)
        = E[ -2c tr(D Sigma)/Q + 4c (Y' D Sigma A^{-1} Y)/Q^2 + c^2 (Y' D Y)/Q^2 ].

Claim (sufficient window):

    0 < c < 2 ( tr(D Sigma) - 2 max_j d_j sigma_j^2 ) / max_j (d_j a_j)  implies dominance,

with the four corollaries used in the paper checked separately:
    (a) D = I,        A = Sigma          ->  c < 2 (peff - 2),  peff = tr Sigma / lam_max
    (b) D = A^{-1} = Sigma^{-1}          ->  c < 2 (p - 2)
    (c) D = A^{-1} = Sigmahat^{-1}       ->  c < 2 ( sum_j r_j - 2 max_j r_j ), r_j = sigma_j^2/sigmahat_j^2
    (d) Sigma = sigma^2 I, A = sigmahat^2 I, D = I, c chosen as (p-2)  ->  sigmahat^2 < 2 sigma^2
"""

import numpy as np

rng = np.random.default_rng(4242)


def risk(Y, mu, d, k):
    return float((((k[:, None] if np.ndim(k) else k) * Y - mu) ** 2 * d).sum(1).mean())


def exact_identity(p=10, n=3_000_000, c=4.0, seed=1):
    r = np.random.default_rng(seed)
    s2 = r.uniform(0.4, 2.5, p)
    a = r.uniform(0.5, 2.0, p)
    d = r.uniform(0.3, 3.0, p)
    mu = r.normal(size=p) * 0.7
    Y = mu + r.normal(size=(n, p)) * np.sqrt(s2)
    Q = (Y ** 2 / a).sum(1)
    lhs = risk(Y, mu, d, 1.0 - c / Q) - (d * s2).sum()
    t1 = -2 * c * (d * s2).sum() / Q
    t2 = 4 * c * (Y ** 2 * d * s2 / a).sum(1) / Q ** 2
    t3 = c ** 2 * (Y ** 2 * d).sum(1) / Q ** 2
    vals = t1 + t2 + t3
    rhs, se = vals.mean(), vals.std(ddof=1) / np.sqrt(n)
    lse = ((((1 - c / Q)[:, None] * Y - mu) ** 2 * d).sum(1)).std(ddof=1) / np.sqrt(n)
    print(f"  MC = {lhs:+.6f} (se {lse:.6f})   identity = {rhs:+.6f} (se {se:.6f})")
    return abs(lhs - rhs) < 4 * np.sqrt(se ** 2 + lse ** 2)


def window(label, p, s2, a, d, n=600_000, seed=2, grid_mult=3.0):
    r = np.random.default_rng(seed)
    mu = r.normal(size=p) * np.sqrt(s2) * 0.8
    Y = mu + r.normal(size=(n, p)) * np.sqrt(s2)
    Q = (Y ** 2 / a).sum(1)
    base = risk(Y, mu, d, 1.0)
    c_up = 2 * ((d * s2).sum() - 2 * (d * s2).max()) / (d * a).max()
    grid = np.linspace(0.05 * max(c_up, 1e-9), grid_mult * max(c_up, 1e-9), 90)
    diffs = np.array([risk(Y, mu, d, 1 - cc / Q) - base for cc in grid])
    pos = np.where(diffs > 0)[0]
    cross = grid[pos[0]] if len(pos) else np.inf
    ok = cross >= 0.97 * c_up
    print(f"  {label:52s} window={c_up:9.3f}  empirical crossing={cross:9.3f}  ok={ok}")
    return ok


if __name__ == "__main__":
    print("exact weighted-loss risk identity")
    ok0 = exact_identity()
    print(f"  identity verified: {ok0}\n")

    p = 60
    s2 = np.exp(np.linspace(0, 2.5, p))            # strongly heteroscedastic
    s2 = s2 / s2.mean()
    peff = s2.sum() / s2.max()
    print(f"heteroscedastic block: p={p}, peff={peff:.3f}")

    res = []
    res.append(window("(a) D=I, A=Sigma  [expect 2(peff-2)]", p, s2, s2, np.ones(p)))
    res.append(window("(b) D=A^-1=Sigma^-1  [expect 2(p-2)]", p, s2, s2, 1 / s2))
    sh = s2 * np.exp(rng.normal(size=p) * 0.4)     # noisy variance estimate
    res.append(window("(c) D=A^-1=Sigmahat^-1, mis-specified", p, s2, sh, 1 / sh))
    ho = np.ones(p)
    for f in [0.6, 1.0, 1.6, 2.4]:
        res.append(window(f"(d) homoscedastic, A={f} Sigma, D=I", p, ho, f * ho, ho))

    print("\ncorollary (d) threshold check: with c=(p-2) the rule dominates iff "
          "sigmahat^2 < 2 sigma^2")
    r = np.random.default_rng(9)
    Yh = r.normal(size=(2_000_000, p))
    muh = np.zeros(p)
    nrm2 = (Yh ** 2).sum(1)
    b = risk(Yh, muh, ho, 1.0)
    for sh2 in [0.5, 1.0, 1.5, 1.95, 2.05, 2.5]:
        rr = risk(Yh, muh, ho, 1 - (p - 2) * sh2 / nrm2)
        print(f"    sigmahat^2={sh2:4.2f}  relative risk={rr/b:.4f}  dominates={rr < b}")

    print(f"\nall window checks passed: {all(res) and ok0}")
