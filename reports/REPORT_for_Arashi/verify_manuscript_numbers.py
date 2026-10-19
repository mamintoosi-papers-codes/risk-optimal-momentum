"""Cross-check every number the manuscript/report quotes against the FRESH
phase-D outputs (regenerated 18 Oct 16:31-17:08 by phaseD.bat, full mode).

Checks:
  A. simulation_noise.json: per-section seeds/steps, SGD/AMSGrad/AdamW/SR-Adam
     cells of tab:opt_noise, t-stats, tuned-lr ratio at flip 0.3/0.4
  B. simulation_batch.json: tab:batch cells, the 1.86 trace factor,
     Stein-weight range 0.13-0.16, |diff| < 0.01
  C. gate_open_frac: 1.0 on every tab:opt_noise run and every Digits row,
     0-0.75 on the remaining tabular rows, not recorded elsewhere
  D. ms_per_step: 0.29 ms (gaussian p=400), 0.55 ms (student p=400),
     1.3 ms (logistic p=200)
  E. per-p normalisation: -0.0007..-0.0002 (median over the three
     high-noise columns) and 3e-6..1.2e-4 at tau=0.1
  F. new wording present, stale wording absent
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "out"
tex = (ROOT / "paper" / "sn-article.tex").read_text(encoding="utf-8")
fails = []


def check(label, tex_val, fresh_val, tol, fmt="{:+.4f}"):
    ok = abs(tex_val - fresh_val) <= tol
    if not ok:
        fails.append(f"{label}: tex={tex_val} fresh={fresh_val}")
    print(f"   {label:<38} tex={fmt.format(tex_val):>12} fresh={fmt.format(fresh_val):>12}"
          f"  {'OK' if ok else '*** MISMATCH ***'}")


# ================= A. simulation_noise =================
print("A. simulation_noise.json (tab:opt_noise)")
sn = json.loads((out / "simulation_noise.json").read_text(encoding="utf-8"))
print("   meta:", {k: sn["meta"].get(k) for k in ("seeds", "steps", "tuned", "timing", "gate_log")})
lists = {k: v for k, v in sn.items() if isinstance(v, list) and v and isinstance(v[0], dict)}
for k, v in lists.items():
    kinds = sorted({("quad" if "tau" in r else "log" if ("flip" in r or "label_flip" in r) else "?")
                    for r in v})
    seeds = sorted({str(r.get("seeds")) for r in v if "seeds" in r})
    steps = sorted({str(r.get("steps")) for r in v if "steps" in r})
    print(f"   list '{k}': {len(v)} rows, kinds={kinds}, seeds={seeds}, steps={steps}")
    print(f"      keys: {list(v[0].keys())}")

# collect fresh means by (kind, level)
fresh = {}
for k, v in lists.items():
    for r in v:
        kind = "quad" if "tau" in r else "log"
        lvl = r.get("tau", r.get("flip", r.get("label_flip")))
        fresh[(kind, round(float(lvl), 2))] = r

# parse the inline table from the tex (robust: split cells, strip $ and \tau=)
m = re.search(r"\\label\{tab:opt_noise\}.*?\\begin\{tabular\}\{lccccr\}(.*?)\\bottomrule",
              tex, re.S)
rows_tex = []
for line in m.group(1).split("\\\\"):
    line = line.strip()
    if "&" not in line or "pm" in line or "multicolumn" in line:
        continue
    cells = [c.replace("$", "").strip().replace("\\tau=", "") for c in line.split("&")]
    if len(cells) != 6:
        continue
    try:
        lvl = float(cells[0])
        nums = [float(c) for c in cells[1:5]]
        dt = re.match(r"([+-][0-9.]+)\s*\(([+-]?[0-9.]+)\)$", cells[5].replace(" ", ""))
        if not dt:
            continue
        rows_tex.append((lvl, *nums, float(dt.group(1)), float(dt.group(2))))
    except ValueError:
        continue
print(f"   parsed {len(rows_tex)} data rows from the tex table")
quad_tex = rows_tex[:6]
log_tex = rows_tex[6:]

for lvl, sgd, adamw, ams, sr, diff, t in quad_tex:
    f = fresh.get(("quad", round(lvl, 2)))
    if not f:
        fails.append(f"quad tau={lvl} missing in JSON")
        continue
    for name, tv, jv, tol in (
        ("SGD", sgd, f.get("sgdm"), 5e-4), ("AdamW", adamw, f.get("adamw"), 5e-4),
        ("AMSGrad", ams, f.get("amsgrad"), 5e-4), ("SRAdam", sr, f.get("sradam"), 5e-4),
        ("diff", diff, f.get("paired_diff"), 5e-4), ("t", t, f.get("t_stat"), 0.05),
    ):
        if jv is None:
            fails.append(f"quad tau={lvl} {name}: key missing in JSON")
            continue
        check(f"quad tau={lvl:<5} {name}", tv, jv, tol)
for lvl, sgd, adamw, ams, sr, diff, t in log_tex:
    f = fresh.get(("log", round(lvl, 2)))
    if not f:
        fails.append(f"log flip={lvl} missing in JSON")
        continue
    for name, tv, jv, tol in (
        ("SGD", sgd, f.get("sgdm"), 5e-4), ("AdamW", adamw, f.get("adamw"), 5e-4),
        ("AMSGrad", ams, f.get("amsgrad"), 5e-4), ("SRAdam", sr, f.get("sradam"), 5e-4),
        ("diff", diff, f.get("paired_diff"), 5e-4), ("t", t, f.get("t_stat"), 0.05),
    ):
        if jv is None:
            fails.append(f"log flip={lvl} {name}: key missing in JSON")
            continue
        check(f"log flip={lvl:<5} {name}", tv, jv, tol)

# percentage sentences
q = {round(float(r[0]), 2): r for r in quad_tex}
l = {round(float(r[0]), 2): r for r in log_tex}
print("   percent below/above AdamW (text: 32/9 below at 0.1/0.25; 11/30 above at 0.5/1;"
      " 27/24 below at 2/4):")
for lvl, want in ((0.10, -32), (0.25, -9), (0.50, +11), (1.00, +30), (2.00, -27), (4.00, -24)):
    adamw, sr = q[lvl][2], q[lvl][4]
    pct = 100 * (sr - adamw) / adamw
    ok = abs(pct - want) <= 1.6
    if not ok:
        fails.append(f"quad tau={lvl} pct {pct:.1f} vs text {want}")
    print(f"      tau={lvl:<5} {pct:+6.1f}%  (text {want:+d}%)  {'OK' if ok else '*** MISMATCH ***'}")
for lvl, want in ((0.0, -4), (0.1, -5), (0.2, -4), (0.3, +5), (0.4, +5)):
    adamw, sr = l[lvl][2], l[lvl][4]
    pct = 100 * (sr - adamw) / adamw
    ok = abs(pct - want) <= 1.1
    if not ok:
        fails.append(f"log flip={lvl} pct {pct:.1f} vs text {want}")
    print(f"      flip={lvl:<5} {pct:+6.1f}%  (text {want:+d}%)  {'OK' if ok else '*** MISMATCH ***'}")

# tuned lr ratio at flip 0.3/0.4 vs 0.1/0.2  (from out/simulation_noise_log_f*_tune.json)
print("   tuned-lr sentence ('AdamW lr three times smaller at flip 0.3/0.4'):")
adam_lrs = {}
for flip in ("0", "0.1", "0.2", "0.3", "0.4"):
    p = out / f"simulation_noise_log_f{flip}_tune.json"
    if not p.exists():
        continue
    tj = json.loads(p.read_text(encoding="utf-8"))
    lr = (tj.get("chosen", {}).get("adamw") or {}).get("lr")
    adam_lrs[flip] = lr
    print(f"      f{flip}: chosen adamw lr = {lr}")
lo = [adam_lrs[f] for f in ("0", "0.1", "0.2") if adam_lrs.get(f)]
hi = [adam_lrs[f] for f in ("0.3", "0.4") if adam_lrs.get(f)]
if lo and hi:
    ratio = statistics.mean(lo) / statistics.mean(hi)
    ok = abs(ratio - 3.0) <= 0.6
    if not ok:
        fails.append(f"adamw lr ratio {ratio:.2f} vs text 'three times smaller'")
    print(f"      ratio low/high flip = {ratio:.2f} (text: three times) "
          f"{'OK' if ok else '*** MISMATCH ***'}")
else:
    fails.append("could not read chosen adamw lr from the tune files")

# ================= B. simulation_batch =================
print("\nB. simulation_batch.json (tab:batch)")
sb = json.loads((out / "simulation_batch.json").read_text(encoding="utf-8"))
rows_b = sb if isinstance(sb, list) else sb.get("rows") or sb.get("batch") or []
if not rows_b:
    for k, v in sb.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            rows_b = v
            print(f"   using list '{k}'")
            break
print(f"   {len(rows_b)} rows; keys: {list(rows_b[0].keys())}")
m = re.search(r"\\label\{tab:batch\}.*?\\begin\{tabular\}\{lccrcc\}(.*?)\\bottomrule",
              tex, re.S)
btex = []
BS = chr(92) * 2
for line in m.group(1).split(BS):
    line = line.strip()
    if "&" not in line or "pm" not in line:
        continue
    cells = [c.replace("$", "").strip().replace(chr(92) + "pm", "+/-")
             for c in line.split("&")]
    if len(cells) != 6 or not cells[0].isdigit():
        continue
    mm = re.match(r"([0-9.]+)\+/-([0-9.]+)", cells[1].replace(" ", ""))
    nn = re.match(r"([0-9.]+)\+/-([0-9.]+)", cells[2].replace(" ", ""))
    dd = re.match(r"([+-][0-9.]+)\(([+-]?[0-9.]+)\)", cells[3].replace(" ", ""))
    if not (mm and nn and dd):
        print("   SKIP batch line:", line[:80], cells[1:4])
        continue
    btex.append((int(cells[0]), float(mm.group(1)), float(mm.group(2)),
                 float(nn.group(1)), float(nn.group(2)), float(dd.group(1)),
                 float(dd.group(2)), float(cells[4]), float(cells[5])))
print(f"   parsed {len(btex)} tex rows")
byB = {int(r["B"]): r for r in rows_b}
for B, m1, sd1, m2, sd2, diff, t, trace, weight in btex:
    f = byB.get(B)
    if not f:
        fails.append(f"batch B={B} missing in JSON")
        continue
    check(f"B={B:<4} adam_mean", m1, f["adam_mean"], 5e-4)
    check(f"B={B:<4} adam_sd", sd1, f["adam_sd"], 5e-4)
    check(f"B={B:<4} sr_mean", m2, f["sr_mean"], 5e-4)
    check(f"B={B:<4} sr_sd", sd2, f["sr_sd"], 5e-4)
    check(f"B={B:<4} diff", diff, f["paired_diff"], 5e-4)
    check(f"B={B:<4} t", t, f["t_stat"], 0.05)
    # trace: the table column is tr(Sigmahat), stored as 'floor' (simulation_batch.py)
    check(f"B={B:<4} trace(tr)", trace, f.get("floor"), 5e-4)
    check(f"B={B:<4} weight", weight, f["weight"], 5e-3)

traces = [byB[int(B)].get("floor") for B, *_ in btex]
traces = [t for t in traces if t]
if len(traces) >= 2:
    ratios = [traces[i] / traces[i + 1] for i in range(len(traces) - 1)]
    mean_ratio = statistics.mean(ratios)
    ok = abs(mean_ratio - 1.86) <= 0.02
    if not ok:
        fails.append(f"trace ratio {mean_ratio:.3f} vs text 1.86")
    print(f"   mean trace ratio per doubling = {mean_ratio:.3f} (text 1.86) "
          f"{'OK' if ok else '*** MISMATCH ***'}")

# ================= C. gate =================
print("\nC. gate_open_frac (text: fraction of blocks open at the last step; 1.0 on")
print("   every tab:opt_noise run and every Digits row; 0-0.75 elsewhere)")
for name in ("simulation_noise", "paper_tables", "path4_dim_sweep", "noise_sweep",
             "simulation_batch"):
    p = out / f"{name}.json"
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8")
    fr = sorted(set(re.findall(r'"gate_open_frac":\s*([0-9.eE+-]+)', txt)))
    has = "gate_open_frac" in txt
    print(f"   {name:<20} recorded={has}  distinct fractions={fr}")

sn_fr = [float(x) for x in re.findall(
    r'"gate_open_frac":\s*([0-9.eE+-]+)',
    (out / "simulation_noise.json").read_text(encoding="utf-8"))]
ok_sn = bool(sn_fr) and min(sn_fr) >= 1.0
if not ok_sn:
    fails.append(f"simulation_noise gate {sorted(set(sn_fr))} vs text 'every block ... tab:opt_noise'")
print(f"   tab:opt_noise: all {len(sn_fr)} recorded fractions are 1.0  "
      f"{'OK' if ok_sn else '*** MISMATCH ***'}")

ptj = json.loads((out / "paper_tables.json").read_text(encoding="utf-8"))
g_dig, g_other = [], []
for k in ("regression", "classification"):
    for r in ptj.get(k, []):
        if "gate_open_frac" in r:
            (g_dig if "digit" in str(r.get("dataset", "")) else g_other).append(
                r["gate_open_frac"])
ok_pt = (g_dig and min(g_dig) >= 1.0 and g_other
         and max(g_other) <= 0.75 + 1e-9 and min(g_other) >= 0.0)
if not ok_pt:
    fails.append(f"paper_tables gate digits={g_dig} other={sorted(set(g_other))} vs "
                 "text 'Digits open everywhere, remaining 0 to three quarters'")
print(f"   paper_tables: digits={g_dig}  other=[{min(g_other)}, {max(g_other)}]  "
      f"{'OK' if ok_pt else '*** MISMATCH ***'}")

for name in ("path4_dim_sweep", "noise_sweep", "simulation_batch"):
    p = out / f"{name}.json"
    if p.exists() and "gate_open_frac" in p.read_text(encoding="utf-8"):
        fails.append(f"{name} now records gate_open_frac; update the gate sentence")
    else:
        print(f"   {name:<20} not recorded (as the text implies)  OK")

# ================= D. timings =================
print("\nD. ms_per_step (text: 0.29 ms gauss p=400, 0.55 ms student p=400, 1.3 ms logistic p=200)")
d = json.loads((out / "path4_dim_sweep.json").read_text(encoding="utf-8"))
g400 = [r["ms_per_step"] for r in d["rows"]
        if r["problem"] == "quadratic_gaussian" and r["p"] == 400]
s400 = [r["ms_per_step"] for r in d["rows"]
        if r["problem"] == "quadratic_student" and r["p"] == 400]
l200 = [r["ms_per_step"] for r in d["rows"]
        if r["problem"] == "logistic_labelnoise" and r["p"] == 200]
check("ms gauss quadratic p=400", 0.29, statistics.mean(g400), 0.006, "{:.3f}")
check("ms student quadratic p=400", 0.55, statistics.mean(s400), 0.006, "{:.3f}")
check("ms logistic p=200", 1.3, statistics.mean(l200), 0.06, "{:.3f}")

# ================= E. per-p normalisation =================
print("\nE. per-p normalisation (text: -0.0007..-0.0002 as the median over the")
print("   three high-noise columns; 3e-6..1.2e-4 at tau=0.1)")
rows = d["rows"]
ps = sorted({r["p"] for r in rows})
gap_over_p = []
for p in ps:
    nov01 = [r["paired_diff"] for r in rows if r["p"] == p
             and not (r["problem"] == "quadratic_gaussian" and r["tau"] == 0.1)]
    v = statistics.median(nov01) / p
    gap_over_p.append(v)
    print(f"   p={p:>3}  gap/p (median of 3 high-noise cols)={v:+.6f}")
ok_e = (abs(round(min(gap_over_p), 4) + 0.0007) <= 1e-9
        and abs(round(max(gap_over_p), 4) + 0.0002) <= 1e-9)
if not ok_e:
    fails.append(f"gap/p range {min(gap_over_p):+.6f}..{max(gap_over_p):+.6f} "
                 "vs text -0.0007..-0.0002")
print(f"   rounded range = {round(min(gap_over_p), 4):+.4f}..{round(max(gap_over_p), 4):+.4f}"
      f" (text -0.0007..-0.0002)  {'OK' if ok_e else '*** MISMATCH ***'}")
tau01 = [r["paired_diff"] / r["p"] for r in rows
         if r["problem"] == "quadratic_gaussian" and r["tau"] == 0.1]
ok_t = abs(round(min(tau01), 6) - 3e-6) <= 1e-9 and max(tau01) <= 1.2e-4
if not ok_t:
    fails.append(f"tau=0.1 gap/p {min(tau01):.2e}..{max(tau01):.2e} vs text 3e-6..1.2e-4")
print(f"   tau=0.1 gap/p = {min(tau01):.2e}..{max(tau01):.2e} (text 3e-6..1.2e-4) "
      f"{'OK' if ok_t else '*** MISMATCH ***'}")

# ================= F. wording present / stale wording absent =================
print("\nF. manuscript wording (new claims present, stale claims gone)")
must_have = [
    "All eleven paired differences",
    "At $B=8$ and at $B=32$ the difference is not statistically significant",
    "at $B=16$ it is significant, but the margin is only $0.0032$",
    "$0.29$\\,ms",
    "$0.55$\\,ms",
    "$1.3$\\,ms",
    "at the last step of a run",
    "fixed step size of $0.01$",
    "median over the three high-noise columns of Table",
    "between $3\\times10^{-6}$ and $1.2\\times10^{-4}$",
    "scored on two tuning seeds",
    "the quadratic grid is chosen at $\\tau=1$ and reused",
]
must_not = [
    "All twelve paired",
    "$0.27$\\,ms",
    "$1.4$\\,ms",
    "every step of every run",
    "below $10^{-4}$",
    "held-out tuning seeds",
    "fresh seeds",
    "per noise level",
    "the three smallest batch sizes are not statistically",
]
for s in must_have:
    ok = s in tex
    if not ok:
        fails.append(f"missing wording: {s[:60]}")
    print(f"   {'have  ' if ok else 'MISSING'}  {s[:72]}")
for s in must_not:
    ok = s not in tex
    if not ok:
        fails.append(f"stale wording still present: {s[:60]}")
    print(f"   {'absent' if ok else 'STALE '}  {s[:72]}")

# ================= summary =================
print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  -", f)
else:
    print("ALL CHECKS PASS")
