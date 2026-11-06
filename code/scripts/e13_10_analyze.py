"""E13-10 (v10.6 GLM all-null 10 / mid+R) verdicts from results/E13_seed_sim.csv (8 seeds x en/zh 100) and
results/E13_centre_shift_summary.csv. Paired comparisons 01-10, 10-11, mid-01, mid-T (T = ga_T) per set x embedder:
mean over utterance x seed, utterance-bootstrap CI (from E13_seed_pairs.csv), Wilcoxon on per-utterance seed means,
Holm over the 4 comparisons x 2 sets within each embedder. P1-P5 as registered.
python scripts/e13_10_analyze.py   (env_eval) -> results/E13_10_pairs.csv, results/E13_10_verdicts.csv
"""
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EPS = 0.42
M = "GLM-TTS all-null"
CMP = [("01", "10"), ("10", "11"), ("mid", "01"), ("mid", "T")]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1.0)
    return out

D = pd.read_csv(ROOT / "results/E13_seed_sim.csv")
D = D[D.model == M]
B = pd.read_csv(ROOT / "results/E13_seed_pairs.csv")
B = B[B.model == M]
C = pd.read_csv(ROOT / "results/E13_centre_shift_summary.csv")
C = C[C.model == M]
rows = []
for e, ge in D.groupby("emb"):
    for s, g in ge.groupby("set"):
        W = g.pivot_table(index=["utt", "seed"], columns="cond", values="sim")
        for a, b in CMP:
            dd = (W[a] - W[b]).unstack("seed")
            per_seed = dd.mean(0)
            bb = B[(B.set == s) & (B.emb == e) & (B.comparison == f"{a} - {b}")].iloc[0]
            rows.append(dict(set=s, emb=e, comparison=f"{a} - {b}", d=float(dd.values.mean()), lo=bb.lo, hi=bb.hi,
                             p=float(wilcoxon(dd.mean(1)).pvalue), seeds_positive=int((per_seed > 0).sum()),
                             seed_min=float(per_seed.min()), seed_max=float(per_seed.max())))
P = pd.DataFrame(rows)
P["p_holm"] = 0.0
for e, idx in P.groupby("emb").groups.items():
    P.loc[idx, "p_holm"] = holm(P.loc[idx, "p"])
P.to_csv(ROOT / "results/E13_10_pairs.csv", index=False)

dc = C.set_index(["cond", "set", "emb"]).delta_corr
lab = {"10": "first step only (10)", "11": "baseline", "01": "R (ours)", "mid": "mid+R (ours)"}
V = []
for e in ("eres2net", "wavlm", "ecapa"):
    for s in ("en", "zh"):
        q = P[(P.set == s) & (P.emb == e)].set_index("comparison")
        r01_10, r10_11, rm01 = q.loc["01 - 10"], q.loc["10 - 11"], q.loc["mid - 01"]
        V.append(dict(set=s, emb=e,
                      P1=bool(r01_10.seeds_positive == 8 and r01_10.d >= EPS),
                      P2=bool(r10_11.d < 0),
                      P3=bool(dc[lab["10"], s, e] >= dc[lab["11"], s, e]),
                      P4=bool(rm01.d >= -EPS and dc[lab["mid"], s, e] < dc[lab["01"], s, e]),
                      P4_sim=bool(rm01.d >= -EPS), P4_dcorr=bool(dc[lab["mid"], s, e] < dc[lab["01"], s, e]),
                      dcorr_10=dc[lab["10"], s, e], dcorr_11=dc[lab["11"], s, e], dcorr_01=dc[lab["01"], s, e],
                      dcorr_mid=dc[lab["mid"], s, e]))
V = pd.DataFrame(V)
sign = V.groupby("set")[["P1", "P2", "P3", "P4"]].agg(lambda x: x.nunique() == 1)
V.to_csv(ROOT / "results/E13_10_verdicts.csv", index=False)
pd.set_option("display.width", 250)
print(P.round(4).to_string(index=False))
print(V.round(3).to_string(index=False))
print("P5 (same verdict across the three embedders):", sign.to_dict())
S = pd.read_csv(ROOT / "results/E13_seed_summary.csv")
S = S[(S.model == M) & (S.emb == "eres2net")]
print(S.pivot_table(index="cond", columns="set", values=["mean", "std"]).round(2).to_string())
