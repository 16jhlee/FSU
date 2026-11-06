"""E14-8: R′+R prosody variance ratio vs teacher (table 5 recipe, e9_tables.table5: CosyVoice2 en, 8 seeds,
geometric mean over utterances of var_k(feat)/var_k(teacher feat)). Re-derives the R / R′ rows as a check.
python scripts/e14_prosody.py   (env_eval)  -> results/E14_rpr_prosody.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e9_tables import FEAT, pvar  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFGS = [("baseline", "e82_cv2_all_n2_def"), ("R (ours)", "e82_cv2_all_n2_R"),
        ("R′ (Sel-CFG)", "e82_cv2_speaker_n2_def"), ("R′+R", "e82_cv2_speaker_n2_R")]


def main():
    T = pvar("e82_cv2_all_n10_def", "en")
    rows = []
    for meth, cfg in CFGS:
        P = pvar(cfg, "en")
        r = {"method": meth, "config": cfg, "n_utt": int(P.notna().all(axis=1).sum())}
        for f in FEAT:
            lr = np.log((P[f] + 1e-9) / (T[f] + 1e-9)).dropna()
            b = np.random.default_rng(0).choice(lr.values, (10000, len(lr))).mean(1)
            r[f] = np.exp(lr.mean())
            r[f + "_lo"], r[f + "_hi"] = np.exp(np.percentile(b, [2.5, 97.5]))
        rows.append(r)
    D = pd.DataFrame(rows)
    lo, hi, x = (D.set_index("method").loc[m, FEAT] for m in ("R′ (Sel-CFG)", "R (ours)", "R′+R"))
    D["P1_between"] = ""
    D.loc[D.method == "R′+R", "P1_between"] = str(bool(((x >= np.minimum(lo, hi)) & (x <= np.maximum(lo, hi))).all()))
    D.to_csv(ROOT / "results/E14_rpr_prosody.csv", index=False)
    print(D.round(3).to_string())


if __name__ == "__main__":
    main()
