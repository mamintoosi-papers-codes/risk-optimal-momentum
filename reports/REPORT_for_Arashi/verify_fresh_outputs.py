"""Cross-check paper/report numbers against the FRESH phase-D outputs (18-20 Oct).

Checks:
  1. tab:dim_sweep inline table in sn-article.tex vs out/path4_dim_sweep.json
  2. timing claim (0.55 ms quad p=400 gaussian, 0.90 ms student, 1.8 ms logistic p=200)
  3. gate_open_frac across every fresh JSON that records it
  4. dim sweep win count (17/42) and per-p normalisation range quoted in the text
Exits non-zero if any check fails.
"""
import json
import re
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tex = (ROOT / "paper" / "sn-article.tex").read_text(encoding="utf-8")
d = json.loads((ROOT / "out" / "path4_dim_sweep.json").read_text(encoding="utf-8"))
rows = d["rows"]

fails = []


def cell(prob, key, val):
    return {r["p"]: r["paired_diff"] for r in rows
            if r["problem"] == prob and abs(r[key] - val) < 1e-9}


gq_lo = cell("quadratic_gaussian", "tau", 0.1)
gq_hi = cell("quadratic_gaussian", "tau", 1.0)
gq_vhi = cell("quadratic_gaussian", "tau", 4.0)
stu = cell("quadratic_student", "tau", 1.0)
log = cell("logistic_labelnoise", "flip", 0.2)
log4 = cell("logistic_labelnoise", "flip", 0.4)

m = re.search(r"\\begin\{tabular\}\{ccccccc\}(.*?)\\end\{tabular\}", tex, re.S)
if not m:
    print("1. *** tab:dim_sweep table (ccccc c) not found in sn-article.tex ***")
    sys.exit(1)
tab = m.group(1)
print("1. MANUSCRIPT tab:dim_sweep vs FRESH out/path4_dim_sweep.json:")
ok = True
seen = set()
row_re = re.compile(r"(\d+)((?:\s*&\s*\$[+-][\d.]+\$)+)")
for line in tab.split("\\\\"):
    line = line.strip()
    mm = row_re.search(line)
    if not mm:
        continue
    p = int(mm.group(1))
    seen.add(p)
    vals = [float(x) for x in re.findall(r"\$([+-][\d.]+)\$", mm.group(2))]
    fresh = [round(gq_lo[p], 3), round(gq_hi[p], 3), round(gq_vhi[p], 3),
             round(stu[p], 3), round(log[p], 3), round(log4[p], 3)]
    match = len(vals) == 6 and all(abs(a - b) < 0.00061
                                   for a, b in zip(vals, fresh))
    ok &= match
    print(f"   p={p:>3}  tex={vals}  json={fresh}  {'OK' if match else '*** MISMATCH ***'}")
if seen != {10, 20, 50, 100, 200, 400, 500}:
    ok = False
    print(f"   *** row set is {sorted(seen)}, expected 7 rows ***")
print("   RESULT:", "ALL MATCH" if ok else "*** MISMATCH ***")
if not ok:
    fails.append("section 1: tex table vs JSON")

print("\n2. Timing claim vs fresh ms_per_step (text: 0.55 / 0.90 / 1.8 ms):")
means = {}
for label, sel in (
        ("gauss p=400", lambda r: r["problem"] == "quadratic_gaussian" and r["p"] == 400),
        ("student p=400", lambda r: r["problem"] == "quadratic_student" and r["p"] == 400),
        ("logistic p=200", lambda r: r["problem"] == "logistic_labelnoise" and r["p"] == 200)):
    xs = [r["ms_per_step"] for r in rows if sel(r)]
    means[label] = sum(xs) / len(xs)
    print(f"   {label:<16} rows={[round(x, 3) for x in xs]}  mean: {means[label]:.3f}")
for label, want, tol in (("gauss p=400", 0.55, 0.006),
                         ("student p=400", 0.90, 0.006),
                         ("logistic p=200", 1.8, 0.06)):
    good = abs(means[label] - want) <= tol
    if not good:
        fails.append(f"section 2: {label} timing {means[label]:.3f} vs text {want}")
    print(f"   {label:<16} vs {want}  {'OK' if good else '*** MISMATCH ***'}")

print("\n3. gate_open_frac across fresh JSONs:")
found = False
for f in sorted(glob.glob(str(ROOT / "out" / "*.json"))):
    txt = Path(f).read_text(encoding="utf-8", errors="replace")
    if "gate_open_frac" in txt:
        fracs = sorted(set(re.findall(r'"gate_open_frac":\s*([0-9.eE+-]+)', txt)))
        print(f"   {Path(f).name}: {fracs}")
        found = True
if not found:
    fails.append("section 3: no JSON records gate_open_frac")
    print("   *** no JSON records gate_open_frac ***")

print("\n4. Dim-sweep win count and per-p normalisation (text claims 17/42, "
      "flip 0.4 per-coordinate 7e-4..8e-3 growing, tau=0.1 flat "
      "3e-6..1.2e-4):")
wins = sum(1 for r in rows if r["win_sradam"])
good = wins == 17 and len(rows) == 42
if not good:
    fails.append(f"section 4: wins {wins}/{len(rows)} vs text 17/42")
print(f"   wins = {wins}/{len(rows)}  (text: 17/42)  "
      f"{'OK' if good else '*** MISMATCH ***'}")
norms_hi = [r["paired_diff"] / r["p"] for r in rows
            if r["problem"] == "logistic_labelnoise" and r["flip"] == 0.4]
norms_lo = [r["paired_diff"] / r["p"] for r in rows
            if r["problem"] == "quadratic_gaussian" and r["tau"] == 0.1]
good_hi = (abs(norms_hi[1] - 7e-4) <= 1e-4 and abs(norms_hi[-1] - 8e-3) <= 1e-3
           and norms_hi[-1] > 6 * norms_hi[1])
good_lo = abs(round(min(norms_lo), 6) - 3e-6) <= 1e-9 and max(norms_lo) <= 1.2e-4
if not good_hi:
    fails.append("section 4: flip 0.4 gap/p range vs text 7e-4..8e-3")
if not good_lo:
    fails.append("section 4: tau=0.1 gap/p range vs text 3e-6..1.2e-4")
print(f"   flip 0.4 gap/p: {min(norms_hi):.1e}..{max(norms_hi):.1e} "
      f"(text 7e-4..8e-3)  {'OK' if good_hi else '*** MISMATCH ***'}")
print(f"   tau=0.1 gap/p: {min(norms_lo):.1e}..{max(norms_lo):.1e} "
      f"(text 3e-6..1.2e-4)  {'OK' if good_lo else '*** MISMATCH ***'}")

print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL FRESH-OUTPUT CHECKS PASS")
