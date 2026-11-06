"""E1b diagnostics on the E1 paired comparisons (reads results/E1_long.csv from agg_e1.py).

(a) quality confound: per comparison (setting vs family reference) and embedder, Spearman rho between
    utterance-level dSIM and dUTMOS. Only non-collapsed settings (mean dWER <= 2 points) are pooled.
(b) noise floor: results/E1b_noise.csv (from e1b_aux.py noise) -> eps per embedder.
(c) length: utterance-level (SIM_c6 - SIM_full) vs NFE within each family; Spearman with NFE and the
    mean shift at each NFE relative to the reference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from agg_e1 import EMBS, REF, group  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
pd.set_option("display.width", 220)


def main():
    d = pd.read_csv(ROOT / "results" / "E1_long.csv")
    d["grp"] = d.job.map(group)
    d["nfe"] = d.job.str.extract(r"nfe(\d+)$")[0].astype(float)
    noise = pd.read_csv(ROOT / "results" / "E1b_noise.csv")
    eps = noise.groupby("emb").dsim.quantile(0.95)
    print("eps95 (cosine):", eps.round(4).to_dict())

    recs, lens = [], []
    for (job, s), g in d.groupby(["job", "set"]):
        grp = group(job)
        if grp not in REF or job == REF[grp] or grp in ("f5v0",):
            continue
        ref = d[(d.job == REF[grp]) & (d.set == s)].set_index("utt_id")
        g = g.set_index("utt_id")
        c = g.index.intersection(ref.index)
        dwer = (g.loc[c, "wer"] - ref.loc[c, "wer"]).mean() * 100
        du = g.loc[c, "utmos"] - ref.loc[c, "utmos"]
        for e in EMBS:
            ds = g.loc[c, f"sim_{e}"] - ref.loc[c, f"sim_{e}"]
            rho = spearmanr(ds, du).correlation
            dl = (g.loc[c, f"sim_{e}_c6"] - g.loc[c, f"sim_{e}"]) - (ref.loc[c, f"sim_{e}_c6"] - ref.loc[c, f"sim_{e}"])
            recs.append(dict(job=job, set=s, emb=e, dWER=dwer, dUTMOS=du.mean(), dSIM=ds.mean() * 100, rho=rho,
                             collapsed=dwer > 2, dlen=dl.mean() * 100))
    R = pd.DataFrame(recs)
    R.to_csv(ROOT / "results" / "E1b_confound.csv", index=False)
    ok = R[~R.collapsed]
    print("\n(a) Spearman rho(dSIM, dUTMOS), non-collapsed comparisons")
    print(ok.pivot_table(index=["job", "set"], columns="emb", values="rho").round(2).to_string())
    summ = ok.groupby("emb").agg(rho_median=("rho", "median"), rho_absmax=("rho", lambda x: x.abs().max()),
                                 frac_gt05=("rho", lambda x: (x.abs() > 0.5).mean()))
    print("\nsummary by embedder\n", summ.round(3))
    print("\n(c) length effect: mean (c6 - full) shift vs reference, SIM points x100 (non-collapsed)")
    print(ok.pivot_table(index=["job", "set"], columns="emb", values="dlen").round(2).to_string())
    # length effect trend with NFE within family (en, all settings incl. reference)
    print("\n(c2) Spearman(NFE, c6-full) per family/embedder, en")
    for grp in ["cv2", "zipvoice", "zipvoice_distill", "dmo_teacher", "dmo_student"]:
        g = d[(d.grp == grp) & (d.set == "en")]
        print(grp, {e: round(spearmanr(g.nfe, g[f"sim_{e}_c6"] - g[f"sim_{e}"]).correlation, 3) for e in EMBS})
    # ZipVoice 16 -> 8 discrepancy
    print("\nZipVoice 16->8 per set (dSIM x100, rho with dUTMOS, dUTMOS, dlen)")
    print(R[R.job == "zipvoice_nfe8"].pivot_table(index="set", columns="emb", values=["dSIM", "rho", "dlen"]).round(2))
    print(R[R.job == "zipvoice_nfe8"].groupby("set")[["dUTMOS", "dWER"]].first().round(3))


if __name__ == "__main__":
    main()
