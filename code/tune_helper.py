"""
tune_helper.py — Phase C shared infrastructure (no experiment logic of its own).

Diagnostic item a7: the baselines in Sec. 8.5 / 8.6 run at a common step size
with no tuning, which the diagnostic calls unfair.  None of the paper drivers
exposes any hyper-parameter grid today (only --seeds/--steps/--quick), so this
module adds, in one place, the arguments and helpers every driver needs:

    --tune              run the grid below on a couple of tuning seeds and use
                         the chosen settings for the reported run
    --tune-seeds N      seeds used for tuning only (default 2)
    --lr-grid LIST      comma-separated step sizes, tried for every optimiser
                         (so the grid is identical across methods)
    --beta1-grid LIST   comma-separated beta1 values for the first moment
                         (beta1 = 0 is included to satisfy the diagnostic;
                          applies to SR-Adam and is recorded for the others)
    --wd-grid LIST      comma-separated weight-decay values (regularisation
                         control arm of a7)
    --early-stop N      validation patience in steps, 0 = off (regularisation
                        control arm of a7; needs the driver to pass a val split)
    --rwd / --no-rwd ... (not needed; defaults unchanged)
    --timing            per-step wall-clock timing column
    --gate-log          log SR-Adam gate-open statistics per run

Typical use inside a driver:

    from tune_helper import add_tuning_args, parse_grid, run_tuning, Stopwatch

    ap = argparse.ArgumentParser()
    add_tuning_args(ap)
    args = ap.parse_args()
    if args.tune:
        report, chosen = run_tuning(
            args,
            options=OPTS,                       # optimisers to tune
            pilot=lambda cfg, seed: score(...)  # lower is better by default
        )
    ...

`run_tuning` writes ../out/<tag>_tune.json with the full grid table so the
fairness protocol is auditable.

This module executes nothing on import.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"


def add_tuning_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = ap.add_argument_group("phase C — fair-baseline controls (item a7)")
    g.add_argument("--tune", action="store_true",
                   help="tune lr (and grids below) on tuning seeds, then run")
    g.add_argument("--tune-seeds", type=int, default=2)
    g.add_argument("--lr-grid", default=None,
                   help="comma list; default: the driver's own grid")
    g.add_argument("--beta1-grid", default=None,
                   help="comma list, e.g. 0,0.5,0,9,0.99 — SR-Adam only")
    g.add_argument("--wd-grid", default=None,
                   help="comma list of weight decay (regularisation control)")
    g.add_argument("--early-stop", type=int, default=0, metavar="PATIENCE",
                   help="validation patience in steps; 0 = off")
    g.add_argument("--timing", action="store_true",
                   help="record per-step wall-clock time for every optimiser")
    g.add_argument("--gate-log", action="store_true",
                   help="record SR-Adam gate-open statistics per run")
    return ap


def parse_grid(s, default=None):
    """'a,b,c' -> list of floats (empty -> default)."""
    if not s:
        return default
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]


class Stopwatch:
    """Per-step accumulator; enabled=False makes tick() cost nothing."""

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.acc = 0.0
        self.n = 0
        self._last = None

    def tick(self):
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._last is not None:
            self.acc += now - self._last
            self.n += 1
        self._last = now

    def reset(self):
        self.acc, self.n, self._last = 0.0, 0, None

    @property
    def ms_per_step(self):
        return (self.acc / self.n * 1000.0) if self.n else None


def gate_stats(opt, name):
    """Gate-open diagnostics for SR-Adam runs; {} for other optimisers."""
    try:
        from sr_adam import SRAdam
    except Exception:
        return {}
    if not isinstance(opt, SRAdam) or not getattr(opt, "stats", None):
        return {}
    rep = opt.shrinkage_report()
    return dict(n_blocks=rep.get("n_blocks"),
                gate_open_frac=(rep["n_gated_on"] / rep["n_blocks"]
                                if rep.get("n_blocks") else None),
                mean_weight=rep.get("mean_weight"),
                mean_peff=rep.get("mean_peff"))


def enable_track_stats(opt_name, opt, flag):
    """Drivers construct optimisers themselves; after construction, if the
    flag is set and the optimiser is SR-Adam, ask it to record statistics.
    SRAdam reads track_stats at construction time in most drivers, so drivers
    should instead pass track_stats=flag; this helper covers wrappers that
    cannot (returns True when the option was already satisfied)."""
    if not flag:
        return True
    try:
        from sr_adam import SRAdam
    except Exception:
        return True
    if isinstance(opt, SRAdam):
        opt.track_stats = True
        return True
    return False


def run_tuning(args, options, pilot, tag, lower_is_better=True,
               default_grids=None, extra_grids=None):
    """Grid-search `options` x lr (per-option default grid if none given).

    args       : parsed args containing tune*, lr_grid, beta1_grid, wd_grid
    options    : list of optimiser names
    pilot      : callable(cfg: dict, seed: int) -> float  (score to minimise
                 when lower_is_better, else maximise)
    tag        : output stem, e.g. "simulation_noise"
    default_grids: {opt: [lr, ...]} used when --lr-grid is absent
    extra_grids  : {"beta1": [...], "wd": [...]} optional extra axes

    Returns (report_rows, chosen) where chosen[opt] = dict(lr=..., beta1=...,
    wd=...)  and additionally appends _tune.json under ../out.
    """
    lrs = parse_grid(getattr(args, "lr_grid", None))
    betas = parse_grid(getattr(args, "beta1_grid", None))
    wds = parse_grid(getattr(args, "wd_grid", None))
    tseeds = list(range(1, int(getattr(args, "tune_seeds", 2)) + 1))

    report, chosen = [], {}
    for o in options:
        grid = lrs or (default_grids or {}).get(o) or [None]
        best = None
        for lr in grid:
            for b in (betas or [None]):
                for wd in (wds or [None]):
                    cfg = dict(opt=o, lr=lr, beta1=b, wd=wd)
                    if b is not None and o not in ("sradam",):
                        continue        # beta1 axis only applies to SR-Adam
                    scores = []
                    for s in tseeds:
                        try:
                            scores.append(float(pilot(cfg, s)))
                        except Exception as e:      # keep the grid honest
                            scores.append(float("nan"))
                    score = _nanmean(scores)
                    row = dict(cfg, scores=scores, score=score)
                    report.append(row)
                    if score == score and (best is None or
                                           (score < best if lower_is_better
                                            else score > best)):
                        best, best_cfg = score, dict(cfg)
        chosen[o] = {"lr": (best_cfg or {}).get("lr"),
                     "beta1": (best_cfg or {}).get("beta1"),
                     "wd": (best_cfg or {}).get("wd")}
        print(f"  [tune] {o:8s} -> {chosen[o]}  score={best}")

    payload = dict(meta=dict(tag=tag, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                             tune_seeds=tseeds, lower_is_better=lower_is_better,
                             note="grid identical across optimisers unless a "
                                  "driver default grid differs (recorded in rows)"),
                   chosen=chosen, rows=report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}_tune.json").write_text(json.dumps(payload, indent=2,
                                                     default=str))
    print("  [tune] wrote", OUT / f"{tag}_tune.json")
    return report, chosen


def _nanmean(xs):
    good = [x for x in xs if x == x]
    if not good:
        return float("nan")
    return sum(good) / len(good)


def merge_lrs(default_lrs, chosen):
    """default_lrs {opt: lr}; chosen from run_tuning -> final {opt: lr}."""
    out = dict(default_lrs)
    for o, cfg in (chosen or {}).items():
        if cfg.get("lr") is not None:
            out[o] = cfg["lr"]
    return out


class EarlyStopper:
    """Validation early stopping (control arm of item a7).

    patience is expressed in *steps*; validation runs every `interval` steps,
    so the number of consecutive non-improving validations allowed is
    max(1, patience // interval).  mode="min" for losses, "max" for accuracies.

        es = EarlyStopper(patience, interval=25, mode="min")
        ...
        if es.due():
            improved = es.update(val_score)   # caller snapshots parameters
            if es.should_stop:
                break
    """

    def __init__(self, patience, interval=25, mode="min"):
        self.patience = max(0, int(patience))
        self.interval = max(1, int(interval))
        self.mode = mode
        self.best = None
        self.since = 0
        self.should_stop = False
        self.evals = 0

    @property
    def active(self):
        return bool(self.patience)

    def due(self, step):
        """step is 0-based; True when this step ends an evaluation window."""
        return self.active and (step + 1) % self.interval == 0

    def update(self, score):
        self.evals += 1
        better = (self.best is None or
                  (score < self.best if self.mode == "min" else score > self.best))
        if better:
            self.best, self.since = score, 0
        else:
            self.since += 1
            if self.since >= max(1, self.patience // self.interval):
                self.should_stop = True
        return better

    @property
    def best_score(self):
        return self.best