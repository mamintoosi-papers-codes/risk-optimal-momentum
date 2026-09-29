import json
from pathlib import Path

base = Path(r"C:\git\mamintoosi-papers-codes\risk-optimal-momentum\out")
pt = json.loads((base / "paper_tables.json").read_text())
ns = json.loads((base / "noise_sweep.json").read_text())
print("paper_tables meta", pt["meta"])
print("regression", len(pt["regression"]))
for r in pt["regression"]:
    print(" ", r.get("dataset"), r.get("model"), r.get("status"), r.get("paired_diff"))
print("classification", len(pt["classification"]))
for r in pt["classification"]:
    err = (r.get("error") or "")[:100]
    print(" ", r.get("dataset"), r.get("model"), r.get("label_noise"), r.get("status"), err)
print("noise meta", ns["meta"])
reg_win = sum(1 for r in pt["regression"] if r.get("status") == "ok" and r.get("win_sradam"))
reg_ok = sum(1 for r in pt["regression"] if r.get("status") == "ok")
clf_win = sum(1 for r in pt["classification"] if r.get("status") == "ok" and r.get("win_sradam"))
clf_ok = sum(1 for r in pt["classification"] if r.get("status") == "ok")
noise_win = sum(1 for r in ns["rows"] if r.get("win_sradam"))
print(f"reg win {reg_win}/{reg_ok}  clf win {clf_win}/{clf_ok}  noise win {noise_win}/{len(ns['rows'])}")
