"""Cross-check paper/report numbers against the FRESH phase-D outputs (18 Oct).

Checks:
  1. tab:dim_sweep inline table in sn-article.tex vs out/path4_dim_sweep.json
  2. timing claim (0.29 ms quad p=400 gaussian, 0.55 ms student, 1.3 ms logistic p=200)
  3. gate_open_frac across every fresh JSON that records it
  4. dim sweep win count (7/28) and per-p normalisation range quoted in the text
"""
import json
import re
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tex = (ROOT / "paper" / "sn-article.tex").read_text(encoding="utf-8")
d = json.loads((ROOT / "out" / "path4_dim_sweep.json").read_text(encoding="utf-8"))
rows = d["rows"]


def cell(prob, key, val):
    return {r["p"]: r["paired_diff"] for r in rows
            if r["problem"] == prob and abs(r[key] - val) < 1e-9}


gq_lo = cell("quadratic_gaussian", "tau", 0.1)
gq_hi = cell("quadratic_gaussian", "tau", 1.0)
stu = cell("quadratic_student", "tau", 1.0)
log = cell("logistic_labelnoise", "flip", 0.2)

m = re.search(r"\\begin\{tabular\}\{ccccc\}(.*?)\\end\{tabular\}", tex, re.S)
tab = m.group(1)
print("1. MANUSCRIPT tab:dim_sweep vs FRESH out/path4_dim_sweep.json:")
ok = True
for line in tab.split("\\\\"):
    line = line.strip()
    mm = re.match(r"^(\d+)\s*&\s*\$([+-][\d.]+)\$\s*&\s*\$([+-][\d.]+)\$"
                  r"\s*&\s*\$([+-][\d.]+)\$\s*&\s*\$([+-][\d.]+)\$", line)
    if not mm:
        continue
    p = int(mm.group(1))
    vals = [float(mm.group(i)) for i in (2, 3, 4, 5)]
    fresh = [round(gq_lo[p], 3), round(gq_hi[p], 3), round(stu[p], 3), round(log[p], 3)]
    match = all(abs(a - b) < 0.00061 for a, b in zip(vals, fresh))
    ok &= match
    print(f"   p={p:>3}  tex={vals}  json={fresh}  {'OK' if match else '*** MISMATCH ***'}")
print("   RESULT:", "ALL MATCH" if ok else "*** MISMATCH ***")

print("\n2. Timing claim vs fresh ms_per_step:")
t400 = [r["ms_per_step"] for r in rows
        if r["problem"] == "quadratic_gaussian" and r["p"] == 400 and r["tau"] == 1.0]
l200 = [r["ms_per_step"] for r in rows
        if r["problem"] == "logistic_labelnoise" and r["p"] == 200]
print("   quadratic p=400 tau=1 rows:", [round(x, 3) for x in t400],
      " mean:", round(sum(t400) / len(t400), 3))
print("   logistic p=200 rows:       ", [round(x, 3) for x in l200],
      " mean:", round(sum(l200) / len(l200), 3))

print("\n3. gate_open_frac across fresh JSONs:")
found = False
for f in sorted(glob.glob(str(ROOT / "out" / "*.json"))):
    txt = Path(f).read_text(encoding="utf-8", errors="replace")
    if "gate_open_frac" in txt:
        fracs = sorted(set(re.findall(r'"gate_open_frac":\s*([0-9.eE+-]+)', txt)))
        print(f"   {Path(f).name}: {fracs}")
        found = True
if not found:
    print("   *** no JSON records gate_open_frac ***")

print("\n4. Dim-sweep win count and per-p normalisation (text claims 7/28, "
      "-0.0007..-0.0002 per coordinate as the median over the three "
      "high-noise columns, 3e-6..1.2e-4 at tau=0.1):")
wins = sum(1 for r in rows if r["win_sradam"])
print(f"   wins = {wins}/28")
import statistics as _st
med_hi = []
for p in sorted({r["p"] for r in rows}):
    three = [r["paired_diff"] for r in rows if r["p"] == p
             and not (r["problem"] == "quadratic_gaussian" and r["tau"] == 0.1)]
    med_hi.append(round(_st.median(three) / p, 4))
norms_lo = [r["paired_diff"] / r["p"] for r in rows
            if r["problem"] == "quadratic_gaussian" and r["tau"] == 0.1]
print("   median-of-3 gap/p:", med_hi, "(text -0.0007..-0.0002)")
print(f"   tau=0.1 gap/p: {min(norms_lo):.1e}..{max(norms_lo):.1e} "
      f"(text 3e-6..1.2e-4)")
