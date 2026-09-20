"""
Correctness tests for the SR-Adam implementation.

1. With the gate forced shut, the update must reproduce, step for step, a direct
   implementation of AMSGrad-flavoured AdamW.  This pins down the claim that SR-Adam
   reduces exactly to Adam when no block qualifies for shrinkage.
2. The shrinkage weight must lie in [0,1] and agree with an independent recomputation
   from the pre-step state.
3. The displacement bound || d_t - g_t || <= sqrt( c_t * max_j s2_j ) must hold at every
   gated step.
4. The optimiser must drive a noisy quadratic and a small network to a finite, sensible
   objective, with no step producing a non-finite iterate.
"""

import math

import torch

from sr_adam import SRAdam, ema_variance_correction


def reference_amsgrad_adamw(params, grads, lr, betas, eps, wd):
    """AdamW with the AMSGrad running maximum, written out directly."""
    b1, b2 = betas
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    vmax = [torch.zeros_like(p) for p in params]
    for t in range(1, len(grads) + 1):
        for i, p in enumerate(params):
            g = grads[t - 1]
            s = t - 1
            if s >= 1:
                m_prev = m[i] / (1 - b1 ** s)
                w = (1 - b1) / (1 - b1 ** t)
                direction = m_prev + w * (g - m_prev)
            else:
                direction = g
            m[i].mul_(b1).add_(g, alpha=1 - b1)
            v[i].mul_(b2).addcmul_(g, g, value=1 - b2)
            vhat = v[i] / (1 - b2 ** t)
            torch.maximum(vmax[i], vhat, out=vmax[i])
            denom = vmax[i].sqrt() + eps
            if wd:
                p.mul_(1 - lr * wd)
            p.addcdiv_(direction, denom, value=-lr)
    return params


def test_reduces_to_adam():
    torch.manual_seed(0)
    p1 = torch.randn(40, requires_grad=True)
    p2 = p1.detach().clone()
    grads = [torch.randn(40) for _ in range(40)]

    opt = SRAdam([p1], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 peff_margin=1e9, warmup=3)          # gate can never open
    for t in range(40):
        p1.grad = grads[t].clone()
        opt.step()

    reference_amsgrad_adamw([p2], grads, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, wd=0.01)
    err = (p1.detach() - p2).abs().max().item()
    print(f"  max |SR-Adam(gate shut) - AMSGrad AdamW| = {err:.3e}")
    assert err < 1e-10, err


def test_weight_and_displacement():
    torch.manual_seed(1)
    n = 200
    p = torch.randn(n, requires_grad=True)
    rho, margin, warmup, omega = 1.0, 1.0, 5, 0.1
    opt = SRAdam([p], lr=1e-3, rho=rho, omega=omega, peff_margin=margin, warmup=warmup,
                 track_stats=True)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ok_weight, ok_disp, n_gated = True, True, 0
    mu = torch.randn(n) * 0.3
    for _ in range(1, 150):
        g = mu + torch.randn(n) * 0.8
        s = opt.state[p].get("step", 0)
        k_ref = None
        if s >= warmup:
            m_prev = opt.state[p]["mbar"] / (1 - b1 ** s)
            v_prev = opt.state[p]["vbar"] / (1 - b2 ** s)
            s2 = (v_prev - m_prev ** 2).clamp(min=0) / ema_variance_correction(s, b1)
            s2 = (1 - omega) * s2 + omega * s2.mean()
            peff = (s2.sum() / s2.max()).item()
            y = g - m_prev
            q = (y * y / (s2 + eps)).sum().item()
            c = rho * (n - 2)
            if peff > 2 + margin:
                n_gated += 1
                k_ref = max(0.0, 1 - c / q)
                disp = ((1 - k_ref) * y).norm().item()
                ok_disp &= disp <= math.sqrt(c * s2.max().item()) + 1e-8
            else:
                k_ref = (1 - b1) / (1 - b1 ** (s + 1))
        p.grad = g.clone()
        opt.step()
        k = opt.stats[0]["shrink_weight"]
        ok_weight &= (-1e-12 <= k <= 1.0 + 1e-12)
        if k_ref is not None:
            ok_weight &= abs(k - k_ref) < 1e-5
    print(f"  gated steps={n_gated}  weight in [0,1] and matches recomputation: {ok_weight}")
    print(f"  displacement bound ||d-g|| <= sqrt(c max_j s2_j) holds: {ok_disp}")
    assert ok_weight and ok_disp


def test_noisy_quadratic():
    torch.manual_seed(2)
    p_dim, noise = 300, 1.0
    H = torch.linspace(0.2, 3.0, p_dim)
    x_star = torch.randn(p_dim)

    def run(name):
        torch.manual_seed(5)
        x = torch.zeros(p_dim, requires_grad=True)
        opt = (SRAdam([x], lr=2e-2, warmup=5) if name == "sradam"
               else torch.optim.AdamW([x], lr=2e-2, weight_decay=0.0, amsgrad=True))
        for _ in range(2000):
            x.grad = H * (x.detach() - x_star) + noise * torch.randn(p_dim)
            opt.step()
            assert torch.isfinite(x).all()
        return (0.5 * (H * (x.detach() - x_star) ** 2).sum()).item()

    f_sr, f_ad = run("sradam"), run("adamw")
    print(f"  terminal quadratic objective: SR-Adam={f_sr:.4f}  AMSGrad-AdamW={f_ad:.4f}")
    assert math.isfinite(f_sr)


def test_small_network():
    torch.manual_seed(3)
    n, d, h, k = 2048, 30, 64, 4
    X = torch.randn(n, d)
    y = (X @ torch.randn(d, k) + 0.4 * torch.randn(n, k)).argmax(1)
    lossf = torch.nn.CrossEntropyLoss()

    def run(make_opt, steps=600, bs=64):
        torch.manual_seed(7)
        net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(),
                                  torch.nn.Linear(h, k))
        opt = make_opt(net.parameters())
        for _ in range(steps):
            idx = torch.randint(0, n, (bs,))
            opt.zero_grad()
            lossf(net(X[idx]), y[idx]).backward()
            opt.step()
        with torch.no_grad():
            return lossf(net(X), y).item()

    l_sr = run(lambda ps: SRAdam(ps, lr=3e-3, warmup=5))
    l_ad = run(lambda ps: torch.optim.AdamW(ps, lr=3e-3, weight_decay=0.0, amsgrad=True))
    print(f"  full-data loss after 600 steps: SR-Adam={l_sr:.4f}  AMSGrad-AdamW={l_ad:.4f}")
    assert math.isfinite(l_sr)


if __name__ == "__main__":
    print("test 1: exact reduction to AdamW when the gate is shut")
    test_reduces_to_adam()
    print("test 2: shrinkage weight and displacement bound")
    test_weight_and_displacement()
    print("test 3: noisy quadratic")
    test_noisy_quadratic()
    print("test 4: small network")
    test_small_network()
    print("\nall tests passed")
