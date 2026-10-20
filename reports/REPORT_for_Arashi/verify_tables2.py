"""Verify every number the RESPONSE REPORT quotes against fresh phase-D outputs.

Checks (against out/*.json regenerated 18 Oct 2026):
  1. tab:dim_sweep headline: 17/42, wins only at the ends of the noise axis,
     flip 0.4 wins everywhere with a per-coordinate gap growing with p
     at tau=1/flip 0.2 at every p, per-coordinate -0.0007..-0.0002
  2. tab:opt_noise quadratic percentages + logistic percentages + the
     "AdamW drops its lr to 0.003" claim
  3. tab:reg linear 5/5 (largest BlogFeedback +1.21) and MLP 3/2
  4. tab:noise classification 9/10 (Digits +3.9 -> +13.5 pp) and
     regression 6/9 (California: two negative, one positive but imprecise)
  5. GNN pilots: 6 wins, 1 tie (winrate_summary graphs family)
  6. winrate family totals behind fig:regime_win (5/5, 15/19, 3/5, ...)
  7. tab:batch: B=8/B=32 below crit 2.776, B=16 significant but +0.0032,
     B>=64 significant and favours AdamW
  8. gate: 100 % on tab:opt_noise + Digits, 0-0.75 elsewhere
  9. timings 0.55 / 0.90 / 1.8 ms, three optimisers
 10. report wording: new claims present, stale claims gone
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "out"
report = (ROOT / "reports" / "REPORT_for_Arashi" /
          "REPORT_diagnostic_response_2026-10-18.md").read_text(encoding="utf-8")
fails = []


def check(label, want, got, tol):
    ok = abs(want - got) <= tol
    if not ok:
        fails.append(f"{label}: report={want} fresh={got}")
    print(f"   {label:<46} report={want:>10.4f} fresh={got:>10.4f} "
          f"{'OK' if ok else '*** MISMATCH ***'}")


def claim(label, cond, detail=""):
    if not cond:
        fails.append(f"{label} {detail}")
    print(f"   {label:<52} {detail}  {'OK' if cond else '*** MISMATCH ***'}")


# ---------------- 1. dim sweep ----------------
print("1. tab:dim_sweep (path4_dim_sweep.json)")
d4 = json.loads((out / "path4_dim_sweep.json").read_text(encoding="utf-8"))
rows = d4["rows"]
wins = [r for r in rows if r.get("paired_diff", 0) > 0]
losses = [r for r in rows if r.get("paired_diff", 0) < 0]
claim(f"{len(wins)} wins / {len(losses)} losses out of {len(rows)}",
      len(rows) == 42 and len(wins) == 17 and len(losses) == 25)
def key(r):
    return (r["problem"], round(r.get("tau", r.get("flip", -1)), 1))
wkeys = sorted({key(r) for r in wins})
claim("wins only in {gauss tau=0.1 (7), gauss tau=4 (3), flip 0.4 (7)}",
      wkeys == [("logistic_labelnoise", 0.4), ("quadratic_gaussian", 0.1),
                ("quadratic_gaussian", 4.0)], f"keys={wkeys}")
g4 = {r["p"]: r["paired_diff"] for r in rows
      if key(r) == ("quadratic_gaussian", 4.0)}
claim("tau=4: positive exactly at p in {100,200,500}",
      sorted(p for p, v in g4.items() if v > 0) == [100, 200, 500]
      and len(g4) == 7, f"gaps={{{', '.join(f'{p}:{g4[p]:+.3f}' for p in sorted(g4))}}}")
f4 = sorted([r for r in rows if key(r) == ("logistic_labelnoise", 0.4)],
            key=lambda r: r["p"])
claim("flip 0.4: SR wins at every p, all |t|>10 from p=50 on",
      all(r["paired_diff"] > 0 for r in f4)
      and all(abs(r["t_stat"]) > 10 for r in f4 if r["p"] >= 50))
pc = [r["paired_diff"] / r["p"] for r in f4]
claim(f"flip 0.4 per-coordinate grows: {pc[1]:.1e} (p=20) -> {pc[-1]:.1e} "
      f"(p=500), text claims 7e-4 -> 8e-3",
      abs(pc[1] - 7e-4) <= 2e-4 and abs(pc[-1] - 8e-3) <= 1e-3
      and pc[-1] > 8 * pc[1])
low = [r["paired_diff"] / r["p"] for r in rows if key(r) == ("quadratic_gaussian", 0.1)]
claim(f"tau=0.1 per-coordinate flat {min(low):.1e}..{max(low):.1e} "
      f"(text 3e-6..1.2e-4)",
      abs(min(low) - 3.0e-6) <= 1.5e-6 and abs(max(low) - 1.2e-4) <= 3e-5)
mid = [r for r in rows if key(r) in (("quadratic_gaussian", 1.0),
                                     ("quadratic_student", 1.0),
                                     ("logistic_labelnoise", 0.2))]
claim("intermediate band: AdamW wins at every p (tau=1, student, flip 0.2)",
      all(r["paired_diff"] < 0 for r in mid) and len(mid) == 21)

# ---------------- 2. opt_noise percentages ----------------
print("2. tab:opt_noise (simulation_noise.json + tune files)")
sn = json.loads((out / "simulation_noise.json").read_text(encoding="utf-8"))
quad = {round(float(r["tau"]), 2): r for r in sn["quadratic"]}
logi = {round(float(r.get("flip", r.get("label_flip"))), 2): r
        for r in sn["logistic"]}


def rel_pct(r):
    return 100 * (r["sradam"] - r["adamw"]) / r["adamw"]


for tau, want in ((0.1, -32), (0.25, -9), (0.5, +11), (1.0, +30),
                  (2.0, -27), (4.0, -24)):
    check(f"quad tau={tau:<5} pct vs {want:+d}", want, rel_pct(quad[tau]), 1.6)
for flip, want in ((0.0, -4), (0.1, -5), (0.2, -4), (0.3, +5), (0.4, +5)):
    check(f"log flip={flip:<5} pct vs {want:+d}", want, rel_pct(logi[flip]), 1.1)
adam_lrs = {}
for flip in ("0", "0.1", "0.2", "0.3", "0.4"):
    p = out / f"simulation_noise_log_f{flip}_tune.json"
    if p.exists():
        tj = json.loads(p.read_text(encoding="utf-8"))
        adam_lrs[flip] = (tj.get("chosen", {}).get("adamw") or {}).get("lr")
claim("tuned AdamW lr = 0.003 at flip 0.3/0.4 (0.01 below)",
      adam_lrs.get("0.3") == 0.003 and adam_lrs.get("0.4") == 0.003
      and adam_lrs.get("0.2") == 0.01, f"lrs={adam_lrs}")
claim("quadratic 5 seeds x 3000 steps",
      sn["meta"].get("seeds") == [1, 2, 3, 4, 5] and sn["meta"].get("steps") == 3000)
src = (ROOT / "code" / "simulation_noise.py").read_text(encoding="utf-8")
m = re.search(r"def logistic\(.*?steps=(\d+),.*?seeds=\(([^)]*)\)", src, re.S)
claim("logistic 3 seeds x 2000 steps (source defaults)",
      m and m.group(1) == "2000" and m.group(2).strip() == "1, 2, 3",
      f"steps={m.group(1) if m else '?'} seeds=({m.group(2) if m else '?'})")

# ---------------- 3. tab:reg ----------------
print("3. tab:reg (paper_tables.json)")
pt = json.loads((out / "paper_tables.json").read_text(encoding="utf-8"))
reg = [r for r in pt["regression"] if r.get("status") != "skipped"]
lin = [r for r in reg if r["model"] == "linear"]
mlp = [r for r in reg if r["model"] == "mlp"]
claim(f"linear rows {sum(r['paired_diff'] > 0 for r in lin)}/{len(lin)} in favour",
      len(lin) == 5 and all(r["paired_diff"] > 0 for r in lin))
claim(f"MLP rows {sum(r['paired_diff'] > 0 for r in mlp)}/{len(mlp)} in favour",
      len(mlp) == 5 and sum(r["paired_diff"] > 0 for r in mlp) == 3)
big = max(lin, key=lambda r: r["adamw"] - r["sradam"])
big_gap = big["adamw"] - big["sradam"]
check(f"largest linear gain ({big['dataset']}) vs +1.21", 1.21, big_gap, 0.01)
other = sorted(r["adamw"] - r["sradam"] for r in lin if r is not big)
claim(f"other four linear rows in +0.003..+0.025 "
      f"({', '.join(f'{v:+.4f}' for v in other)})",
      min(other) >= 0.0025 and max(other) <= 0.0255)
diab_mlp = next(r for r in mlp if r["dataset"] == "diabetes")
check("largest MLP gain (diabetes) vs +0.47", 0.47,
      diab_mlp["adamw"] - diab_mlp["sradam"], 0.005)

# ---------------- 4. tab:noise ----------------
print("4. tab:noise (noise_sweep.json)")
ns = json.loads((out / "noise_sweep.json").read_text(encoding="utf-8"))
nrows = ns["rows"]
clf = [r for r in nrows if r["metric"] == "acc_frac"]
reg_ = [r for r in nrows if r["metric"] != "acc_frac"]
claim(f"classification {sum(r['paired_diff'] > 0 for r in clf)}/"
      f"{len(clf)} in favour",
      len(clf) == 10 and sum(r["paired_diff"] > 0 for r in clf) == 9)
dig = sorted((r for r in clf if r["problem"].startswith("digits")),
             key=lambda r: r["noise"])
dig_pp = [100 * r["paired_diff"] for r in dig]   # acc_frac stored as fraction
claim(f"Digits pp = {[round(x, 1) for x in dig_pp]} (-1.3 clean; +3.9..+13.5 noisy)",
      abs(dig_pp[0] - (-1.3)) <= 0.1
      and abs(dig_pp[1] - 3.9) <= 0.1
      and abs(dig_pp[-1] - 13.5) <= 0.1
      and all(x > 0 for x in dig_pp[1:]))
claim(f"regression {sum(r['paired_diff'] > 0 for r in reg_)}/{len(reg_)} in favour",
      len(reg_) == 9 and sum(r["paired_diff"] > 0 for r in reg_) == 6)
cal = sorted((r for r in reg_ if r["problem"].startswith("california")),
             key=lambda r: r["noise"])
cal_t = []
for r in cal:
    a = r["paired_all"]
    cal_t.append(statistics.mean(a) / (statistics.stdev(a) / len(a) ** 0.5))
claim(f"California: two negative, one positive but imprecise "
      f"(diffs={[round(r['paired_diff'], 4) for r in cal]}, "
      f"t={[round(t, 2) for t in cal_t]})",
      cal[0]["paired_diff"] < 0 and cal[1]["paired_diff"] < 0
      and cal[2]["paired_diff"] > 0 and abs(cal_t[2]) < 2.0)

# ---------------- 5-6. GNN + winrate families ----------------
print("5-6. GNN pilots and winrate families (winrate_summary.json)")
w = json.loads((out / "winrate_summary.json").read_text(encoding="utf-8"))["summary"]
fam = {f["name"].replace("\n", " "): f for f in w}
graphs = next(v for k, v in fam.items() if "Graphs" in k)
claim(f"graphs {graphs['wins']} wins, {graphs['ties']} tie",
      graphs["wins"] == 6 and graphs["ties"] == 1 and graphs["total"] == 7)
for name, wins_, total_ in (("Dimension sweep", 17, 42),
                            ("Real linear", 5, 5),
                            ("Label / target noise", 15, 19),
                            ("Tabular classification", 3, 5)):
    f = next(v for k, v in fam.items() if k.startswith(name))
    claim(f"winrate {name}: {f['wins']}/{f['total']}",
          f["wins"] == wins_ and f["total"] == total_)

# ---------------- 7. tab:batch ----------------
print("7. tab:batch (simulation_batch.json)")
sb = json.loads((out / "simulation_batch.json").read_text(encoding="utf-8"))
byB = {int(r["B"]): r for r in (sb if isinstance(sb, list) else sb["rows"])}
crit = 2.776
for B, want in ((8, 2.70), (32, 1.89)):
    check(f"B={B} t vs report {want}", want, byB[B]["t_stat"], 0.02)
    claim(f"B={B} |t| < {crit}", abs(byB[B]["t_stat"]) < crit)
claim(f"B=16 significant (t={byB[16]['t_stat']:.2f}) but margin +0.0032",
      abs(byB[16]["t_stat"]) > crit and abs(byB[16]["paired_diff"] - 0.0032) <= 5e-4)
claim("B>=64 significant and favours AdamW",
      all(abs(byB[B]["t_stat"]) > crit and byB[B]["paired_diff"] < 0
          for B in (64, 128, 256, 512)))
trace_ratio = statistics.mean([byB[B]["floor"] / byB[b]["floor"]
                               for B, b in zip((8, 16, 32, 64, 128, 256),
                                               (16, 32, 64, 128, 256, 512))])
claim(f"trace falls {trace_ratio:.3f}x per doubling (manuscript says 1.86)",
      abs(trace_ratio - 1.86) <= 0.02)

# ---------------- 8. gate ----------------
print("8. gate_open_frac")
sn_fr = [float(x) for x in re.findall(r'"gate_open_frac":\s*([0-9.eE+-]+)',
                                     (out / "simulation_noise.json").read_text(encoding="utf-8"))]
claim("tab:opt_noise: every recorded fraction is 1.0",
      bool(sn_fr) and min(sn_fr) >= 1.0, f"n={len(sn_fr)}")
g_dig, g_other = [], []
for k in ("regression", "classification"):
    for r in pt.get(k, []):
        if "gate_open_frac" in r:
            (g_dig if "digit" in str(r.get("dataset", "")) else g_other).append(
                r["gate_open_frac"])
claim(f"Digits always 1.0 {g_dig}; others in [0, 0.75] "
      f"[{min(g_other)}, {max(g_other)}]",
      g_dig and min(g_dig) >= 1.0 and max(g_other) <= 0.75 + 1e-9
      and min(g_other) >= 0.0)
for name in ("path4_dim_sweep", "noise_sweep", "simulation_batch"):
    p = out / f"{name}.json"
    claim(f"{name} does not record gate", not (p.exists() and
                                               "gate_open_frac" in p.read_text(encoding="utf-8")))

# ---------------- 9. timings ----------------
print("9. timings")
ms = lambda prob, p: statistics.mean(
    [r["ms_per_step"] for r in d4["rows"] if r["problem"] == prob and r["p"] == p]
    or [float("nan")])
check("ms gaussian p=400 vs 0.55", 0.55, ms("quadratic_gaussian", 400), 0.006)
check("ms student p=400 vs 0.90", 0.90, ms("quadratic_student", 400), 0.006)
check("ms logistic p=200 vs 1.8", 1.8, ms("logistic_labelnoise", 200), 0.06)

# ---------------- 10. report wording ----------------
print("10. report wording (new claims present, stale claims gone)")
must_have = [
    "classification **9/10** in favour",
    "0.55 ms (p=400 quadratic, Gaussian; 0.90 ms under Student-t) and 1.8 ms",
    "averaged over the three optimisers of the sweep",
    "open **at the last step** of each run",
    "B=16 is significant but the margin there is only +0.0032",
    "the agreed split assigns all prose to you",
    "with no argument is identical to `phaseD.bat full`",
    "The wins sit at *both ends of the noise axis*",
    "42 cells, 20 Oct",
    "The graph pilots keep a common fixed lr (0.01)",
    "the fraction of parameter blocks open at the last step of a run",
]
must_not = [
    "11/12",
    "0.27 ms",
    "1.4 ms",
    "0.29 ms",
    "1.3 ms (200-feature",
    "median over the three high-noise columns",
    "tab:dim_sweep` (28 cells)",
    "open in 100 % of steps",
    "`TODO.md`",
    "averaged over the four optimisers",
    "identical candidate step-size grids",
    "timing and gate columns in every row",
    "chosen on two held-out tuning seeds by validation loss",
    "`--tune --timing --early-stop --gate-log`",
]
for s in must_have:
    ok = s in report
    if not ok:
        fails.append(f"report missing: {s[:60]}")
    print(f"   {'have  ' if ok else 'MISSING'}  {s[:72]}")
for s in must_not:
    ok = s not in report
    if not ok:
        fails.append(f"report stale: {s[:60]}")
    print(f"   {'absent' if ok else 'STALE '}  {s[:72]}")

# ---------------- summary ----------------
print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("ALL REPORT CHECKS PASS")
