"""Compact E7 table: per model x NFE x variant -> mask, ΔSIM(ERes2Net, x100) en/zh/hard with significance,
ΔWER en/zh, recovery en/zh, UTMOS en, hf en."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = pd.read_csv(ROOT / "results/E7_seed0_summary.csv")


def f(r, m="d_sim_eres2net"):
    if r is None or pd.isna(r.get(m)):
        return ""
    return f"{r[m]:+.2f}{'*' if r[m + '_p'] < 0.05 else ''}"


rows = []
for (model, nfe, v), g in S.groupby(["model", "nfe", "variant"], sort=False):
    by = {r.set: r for _, r in g.iterrows()}
    en, zh, hd = by.get("en"), by.get("zh"), by.get("hard")
    rows.append(dict(model=model, nfe=nfe, variant=v, mask=g["mask"].iloc[0],
                     dSIM_en=f(en), dSIM_zh=f(zh), dSIM_hard=f(hd),
                     dWER_en=f(en, "d_wer"), dWER_zh=f(zh, "d_wer"), dWER_hard=f(hd, "d_wer"),
                     rec_en=round(en["recovery"], 2) if en is not None else None,
                     rec_zh=round(zh["recovery"], 2) if zh is not None else None,
                     vsT_en=round(en["vs_teacher"], 2) if en is not None else None,
                     utmos_en=round(en["utmos"], 2) if en is not None else None,
                     hf_en=round(en["hf_ratio"], 3) if en is not None and "hf_ratio" in en else None))
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
print(pd.DataFrame(rows).to_string(index=False))
