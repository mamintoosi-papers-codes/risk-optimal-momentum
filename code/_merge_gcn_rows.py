"""Append clean GCN rows from gnn_pilot_*.json to paper/tab_classification.tex if missing."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "out"
PAPER = Path(__file__).resolve().parent.parent / "paper"

meta = {
    "Cora": dict(n=2708, p=1433, K=7),
    "CiteSeer": dict(n=3327, p=3703, K=6),
    "PubMed": dict(n=19717, p=500, K=3),
}
rows = []
for fn in ("gnn_pilot_cora.json", "gnn_pilot_planetoid.json"):
    path = OUT / fn
    if not path.exists():
        continue
    for r in json.loads(path.read_text())["rows"]:
        if r.get("label_noise"):
            continue
        if r["dataset"] not in meta:
            continue
        rows.append((r, meta[r["dataset"]]))

tex = (PAPER / "tab_classification.tex").read_text()
if "GCN" in tex:
    print("GCN already present")
else:
    block = []
    for r, m in rows:
        aw = 100 * float(r["adamw"]) if float(r["adamw"]) <= 1.5 else float(r["adamw"])
        sr = 100 * float(r["sradam"]) if float(r["sradam"]) <= 1.5 else float(r["sradam"])
        sr_s = f"{sr:.1f}"
        if r.get("win_sradam"):
            sr_s = r"\textbf{" + sr_s + "}"
        block.append(
            f"{r['dataset']} & GCN & {m['n']} & {m['p']} & {m['K']} & "
            f"{aw:.1f} & {sr_s} \\\\"
        )
    tex = tex.replace(r"\bottomrule", "\n".join(block) + "\n" + r"\bottomrule")
    (PAPER / "tab_classification.tex").write_text(tex)
    print("appended GCN rows", len(block))

print((PAPER / "tab_classification.tex").read_text())
