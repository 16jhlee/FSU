"""E14-7: one-step closed form for IndexTTS-2 (the Prop 1 check of v8_theory_cosy.py that gave the A1 table's CosyVoice
values), mel space, generated region of the first text segment, en_100, noises hash(utt|noise|k|0) as run_index.
  x1(w)  = NFE 1 output (one Euler step from noise, uniform schedule), w in {0, 0.7, 1.5}; slope = (x1(1.5) - x1(0)) / 1.5
           (= u^c_0 - u^null_0 by construction; collinearity of the three runs reported as lin_resid)
  g_est  = mean(teacher: NFE 25, w 0.7, seeds 0-7) - mean(unconditional: NFE 25, w = -1 -> v = v_u, seeds 0-7)
  cos_slope_gest = cos(slope_k, g_est); rms_slope = RMS(slope_k); rms_gest = RMS(g_est)
  k in {0, 1} = the CosyVoice protocol (seeds shared with g_est); k in {8, 9} = disjoint seeds (independent of g_est)
python scripts/e14_idx_closedform.py [n_utt=100]   (env_index) -> results/E14_idx_closedform.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e10_rho import h, rms  # noqa: E402


def cosv(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    from run_index import REPO, load_model, read_meta, seed_all
    n_utt = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    meta = read_meta(ROOT / "data/lists/en_100.meta.lst")[:n_utt]
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    dev = tts.device
    cfm.cfg_interval, cfm.cfg_ramp, cfm.cfg_branch = None, None, "all"
    if hasattr(cfm, "cfg_steps"):
        cfm.cfg_steps = None

    def out(s, utt, k, n, w):
        seed_all(h(f"{utt}|noise|{k}|0"))
        with torch.inference_mode():
            vc = cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev), None, n,
                               inference_cfg_rate=w)
        return vc[0, :, s["prompt"].size(-1):].double().cpu().numpy()
    rows = []
    for utt, *_ in meta:
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]
        T = [out(s, utt, k, 25, 0.7) for k in range(8)]
        U = [out(s, utt, k, 25, -1.0) for k in range(8)]
        g_est = np.mean(T, 0) - np.mean(U, 0)
        for k in (0, 1, 8, 9):
            x = {w: out(s, utt, k, 1, w) for w in (0.0, 0.7, 1.5)}
            slope = (x[1.5] - x[0.0]) / 1.5
            rows.append(dict(utt=utt, k=k, seeds="shared" if k < 8 else "disjoint",
                             lin_resid=rms(x[0.7] - (x[0.0] + 0.7 * slope)) / rms(1.5 * slope),
                             cos_slope_gest=cosv(slope, g_est), rms_slope=rms(slope), rms_gest=rms(g_est)))
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E14_idx_closedform.csv", index=False)
    print(D.groupby("seeds")[["lin_resid", "cos_slope_gest", "rms_slope", "rms_gest"]].agg(["mean", "min"]).round(4).to_string())


if __name__ == "__main__":
    main()
