"""Refresh the GCN rows of out/paper_tables.json (and paper/tab_classification.tex).

Run *after* `gnn_pilot.py --tune` and `gnn_pilot_planetoid.py --tune`.  The
tabular rows are left untouched: only the GCN entries are recomputed with
`paper_tables.gnn_rows()`, which reads the tuned step sizes from
out/gnn_<dataset>_tune.json (item a7) instead of the old fixed 0.01, so the
full paper_tables driver does not have to be re-run.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_tables import gnn_rows, write_tex  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"

pt_path = OUT / "paper_tables.json"
pt = json.loads(pt_path.read_text(encoding="utf-8"))

fresh = gnn_rows([1, 2, 3, 4, 5], label_noise=(0.0, 0.3))
ok = [r for r in fresh if r.get("status") == "ok"]
print(f"fresh GCN rows: {len(ok)}/{len(fresh)}")
for r in ok:
    print(f"  {r['dataset']:9s} n={r['label_noise']:.1f}  A={r['adamw']:.4f} "
          f"S={r['sradam']:.4f} d={r['paired_diff']:+.4f} win={r['win_sradam']} "
          f"lr=({r.get('lr_adamw')}, {r.get('lr_sradam')})")

kept = [r for r in pt["classification"]
        if str(r.get("model", "")).lower() != "gcn"]
pt["classification"] = kept + ok
pt.setdefault("meta", {})["gcn_refreshed"] = time.strftime("%Y-%m-%d %H:%M:%S")
pt_path.write_text(json.dumps(pt, indent=2, default=str), encoding="utf-8")
write_tex(pt["regression"], pt["classification"])
print("GCN rows replaced in paper_tables.json; tab_regression / tab_classification rewritten")
