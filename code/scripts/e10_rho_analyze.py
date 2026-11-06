"""E10-7 analysis: results/E10_rho_{cv2,cv3,idx,f5,zip}.csv -> results/E10_rho.csv (per model), printed C-1..C-3.
Absolute distances are in each model's own feature space (CosyVoice/IndexTTS 80-bin log-mel, F5 100-bin vocos log-mel,
ZipVoice scaled fbank), so only the scale-free ratios rho/sigma and rho/||g|| are compared across models.
C-3 WER penalty of R (G2): paired d_WER (R - official sampler, % points) at the Table-1 few-step NFE of each model
(CosyVoice2/3 and IndexTTS-2 NFE 2, F5-TTS NFE 8, ZipVoice NFE 4), mean of en and zh (results/E8p_pairs.csv, E1/E8').
python scripts/e10_rho_analyze.py   (env_eval)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("cv2", "CosyVoice2", "token"), ("cv3", "CosyVoice3", "token"), ("idx", "IndexTTS-2", "token"),
          ("f5", "F5-TTS v0", "text"), ("zip", "ZipVoice", "text")]
WER_NFE = {"CosyVoice2": ("G1", 2), "CosyVoice3": ("G1", 2), "IndexTTS-2": ("G1", 2), "F5-TTS": ("G2", 8), "ZipVoice": ("G2", 4)}


def main():
    P = pd.read_csv(ROOT / "results/E8p_pairs.csv")
    rows = []
    for key, name, kind in MODELS:
        f = ROOT / f"results/E10_rho_{key}.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f)
        D["r_sigma"], D["r_g"] = D.rho_nn / D.sigma, D.rho_nn / D.g_rms
        D["rm_sigma"], D["rm_g"] = D.rho_mean / D.sigma, D.rho_mean / D.g_rms
        base = name.split(" ")[0]
        g, n = WER_NFE[base if base != "F5-TTS" else "F5-TTS"]
        q = P[(P.group == g) & (P.model == base) & (P.label == f"NFE {n}") & P.set.isin(["en", "zh"])]
        rows.append(dict(model=name, kind=kind, n_utt=D.utt.nunique(), rho_nn=D.rho_nn.mean(), rho_mean=D.rho_mean.mean(),
                         sigma=D.sigma.mean(), g_rms=D.g_rms.mean(),
                         rho_over_sigma=D.r_sigma.mean(), rho_over_sigma_med=D.r_sigma.median(),
                         rho_over_g=D.r_g.mean(), rho_over_g_med=D.r_g.median(),
                         rhomean_over_sigma=D.rm_sigma.mean(), rhomean_over_g=D.rm_g.mean(),
                         frac_utt_rho_lt_sigma=(D.r_sigma < 1).mean(),
                         wer_penalty_R=q.d_wer.mean() if len(q) else np.nan, wer_nfe=n))
    R = pd.DataFrame(rows)
    R.to_csv(ROOT / "results/E10_rho.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string())
    for col in ("rho_over_g", "rho_over_sigma"):
        rho, p = spearmanr(R[col], R.wer_penalty_R)
        print(f"C-3 Spearman({col}, WER penalty) = {rho:+.2f} (p {p:.3f}, n {len(R)})")


if __name__ == "__main__":
    main()
