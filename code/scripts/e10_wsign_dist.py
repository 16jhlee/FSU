"""E10-3 dump measurement: first-step state z_{t1} = eps + t1 v(w1), v(w1) = (1+w1) v_c - w1 v_u at t=0, distance to the
teacher path set Pi_t1(eps) = {(1-t1) eps + t1 x_k}, x_k = teacher NFE 10 samples (8 seeds) — the THEORY_checks
proposition-2 procedure (v8_theory_cosy.py) with w1 in the E10-3 grid and t1 = the NFE 2 cosine first step (0.2929).
Also ||g_c|| = RMS(v_c - v_u) at t=0 per seed. en first 50 utterances, seeds k = 0, 1 (as proposition 2).
python scripts/e10_wsign_dist.py cv2|cv3 [n_utt=50]   (env_cosy, GPU) -> results/E10_wsign_dist_<model>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v8_theory_cosy import Runner, read_meta, rms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W1 = [-1.0, -0.7, -0.4333, -0.2, 0.0, 0.2, 0.35, 0.7, 1.05]
T1 = float(1 - np.cos(np.pi / 4))


def main():
    key = sys.argv[1]
    n_utt = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    out = ROOT / f"results/E10_wsign_dist_{key}.csv"
    R = Runner(key)
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    rows = []
    for utt, *_ in read_meta(ROOT / "data/lists/en_100.meta.lst")[:n_utt]:
        seg = torch.load(ROOT / R.cache / f"{utt}.pt")[0]
        Tm = np.stack([R.run(seg, utt, k, t10, 0.7)["final"] for k in range(8)])
        for k in range(2):
            o = R.run(seg, utt, k, [0.0, 1.0], 0.0)
            g = rms(o["vc"] - o["vu"])
            pi = (1 - T1) * o["eps"][None] + T1 * Tm
            for w in W1:
                z = o["eps"] + T1 * ((1 + w) * o["vc"] - w * o["vu"])
                d = min(rms(z - p) for p in pi)
                rows.append(dict(utt=utt, k=k, t1=T1, w1=w, dist=d, dist_over_t1=d / T1, g_rms=g))
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(out, index=False)
    print(D.groupby("w1")[["dist_over_t1", "g_rms"]].mean().round(4))


if __name__ == "__main__":
    main()
