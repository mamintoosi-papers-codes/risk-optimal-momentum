import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "out"

sb = json.loads((out / "simulation_batch.json").read_text(encoding="utf-8"))
rows = sb if isinstance(sb, list) else sb.get("rows", sb)
print("== simulation_batch.json ==")
if isinstance(sb, dict):
    print("meta:", json.dumps(sb.get("meta", {}))[:300])
for r in rows:
    print("  ", json.dumps(r))

cc = json.loads((out / "conv_constants.json").read_text(encoding="utf-8"))
print("\n== conv_constants.json ==")
print(json.dumps(cc, indent=1)[:2500])

d = json.loads((out / "path4_dim_sweep.json").read_text(encoding="utf-8"))
print("\n== path4 ms_per_step ==")
for r in d["rows"]:
    if r["p"] in (400, 200) and r["problem"] in ("quadratic_gaussian", "logistic_labelnoise"):
        print(f"   {r['problem']} p={r['p']} tau={r.get('tau')} flip={r.get('flip')} "
              f"ms={r['ms_per_step']:.3f}")

ns = json.loads((out / "noise_sweep.json").read_text(encoding="utf-8"))
print("\n== noise_sweep ms_per_step (unique by problem) ==")
seen = set()
for r in ns.get("rows", []):
    key = r["problem"]
    if key in seen:
        continue
    seen.add(key)
    if "ms_per_step" in r:
        print(f"   {key}: {r['ms_per_step']:.3f} ms")

print("\n== paper_tables gate_open_frac by row ==")
pt = json.loads((out / "paper_tables.json").read_text(encoding="utf-8"))
for split in ("regression", "classification"):
    for r in pt[split]:
        print(f"   {split[:3]} {r.get('dataset','?'):<20} {r.get('model','?'):<9} "
              f"gate={r.get('gate_open_frac')} ms={r.get('ms_per_step')}")
