"""Print E5 geometry summary (ERes2Net-en by default): algorithm vs its Euler baseline, both relative to the
teacher reference cloud."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
emb = sys.argv[1] if len(sys.argv) > 1 else "eres2net"
src = sys.argv[2] if len(sys.argv) > 2 else "results/E5_geometry_summary.csv"
O = pd.read_csv(ROOT / src)
if "base_est" in O and "diff_p" not in O:
    O = O.rename(columns={"p": "diff_p"})
pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 300)
for s in ["en", "zh"]:
    t = O[(O.set == s) & (O.emb == emb)].copy()
    t["v"] = [f"{r.alg_est:.3f} (eu {r.base_est:.3f}; d {r.diff:+.4f}{'*' if r.diff_p < 0.05 else ''})"
              for r in t.itertuples()]
    p = t.pivot_table(index="alg", columns="measure", values="v", aggfunc="first")
    print("==", s, emb)
    print(p[["lvr", "ddcond", "ddglob", "lnorm", "dprompt", "sim"]].to_string())
