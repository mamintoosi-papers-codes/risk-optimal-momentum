"""Inspect the fresh phase-D JSONs for gate fractions, tuned-vs-untuned status,
and every number the manuscript/report quotes, so stale claims can be repaired."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "out"

pt = json.loads((out / "paper_tables.json").read_text(encoding="utf-8"))
print("== paper_tables meta ==")
print(json.dumps(pt["meta"], indent=1)[:800])

print("\n== regression rows (gate/timing if present) ==")
for r in pt["regression"]:
    keep = {k: r[k] for k in r if k in
            ("dataset", "model", "paired_diff", "t_stat", "win_sradam",
             "gate_open_frac", "ms_per_step", "adamw", "sradam")}
    print("  ", keep)

print("\n== classification rows ==")
for r in pt["classification"]:
    keep = {k: r[k] for k in r if k in
            ("dataset", "model", "paired_diff", "t_stat", "win_sradam",
             "gate_open_frac", "ms_per_step", "adamw", "sradam")}
    print("  ", keep)

ns = json.loads((out / "noise_sweep.json").read_text(encoding="utf-8"))
print("\n== noise_sweep meta ==")
print(json.dumps(ns.get("meta", {}), indent=1)[:600])
rows = ns.get("rows", [])
print("n rows:", len(rows))
print("row keys:", list(rows[0].keys()) if rows else None)
