"""E12-4 (v10.4): finite-difference Lipschitz estimate of the CosyVoice2 velocity field, en_100 first 50, seed 0.
State on the teacher path z_t = (1 - t) eps + t x_T (x_T = teacher NFE 10 output, same eps, full sequence incl. prompt
region), t in {0.1, 0.3, 0.5, 0.7}. Perturbation delta: Gaussian on the generated region, RMS(delta) = a * RMS(z_t) over the
generated region, a in {0.01, 0.1, 1.0}, 4 directions. Velocities from one official Euler step of length 1 started at the
state (t_span [t, t+1], dumped v_c / v_u / guided v). Ratio = ||u(z+delta) - u(z)|| / ||delta|| on the generated region,
for u = v_c (conditional field, L), v_u, and the guided field v = (1+w) v_c - w v_u (w = 0.7; bound (1+2w) L).
python scripts/e12_lip.py [n_utt=50]  (env_cosy, GPU)  -> results/E12_lip.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v8_theory_cosy import Runner  # noqa: E402
from run_cosy import h, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = 0.7
T10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))


def flow(R, seg, x0, ts, w):
    dec, m = R.dec, R.m
    dec.inference_cfg_rate, dec.t_span_override, dec.cfg_branch, dec.cfg_interval = w, ts, "all", None
    full = torch.zeros([1, 80, 50 * 300])
    full[:, :, :x0.shape[2]] = x0
    dec.rand_noise = full
    dec.dump_states = []
    with torch.inference_mode():
        m.flow.inference(token=seg["token"].to(m.device, dtype=torch.int32),
                         token_len=torch.tensor([seg["token"].shape[1]], dtype=torch.int32).to(m.device),
                         prompt_token=seg["prompt_token"].to(m.device),
                         prompt_token_len=torch.tensor([seg["prompt_token"].shape[1]], dtype=torch.int32).to(m.device),
                         prompt_feat=seg["prompt_feat"].to(m.device),
                         prompt_feat_len=torch.tensor([seg["prompt_feat"].shape[1]], dtype=torch.int32).to(m.device),
                         embedding=seg["embedding"].to(m.device), streaming=False, finalize=True)
    st, dec.dump_states = dec.dump_states, None
    return st


def main():
    n_utt = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    R = Runner("cv2")
    rng = np.random.default_rng(0)
    rows = []
    for utt, *_ in read_meta(ROOT / "data/lists/en_100.meta.lst")[:n_utt]:
        seg = torch.load(ROOT / R.cache / f"{utt}.pt")[0]
        P = seg["prompt_feat"].shape[1]
        eps = torch.randn([1, 80, 50 * 300], generator=torch.Generator().manual_seed(h(f"{utt}|noise|0")))
        st = flow(R, seg, eps, T10, W)
        xT = st[-1]["x"] + (T10[-1] - T10[-2]) * st[-1]["v"]  # full sequence
        Ltot = xT.shape[2]
        eps = eps[:, :, :Ltot]
        for t in (0.1, 0.3, 0.5, 0.7):
            z = (1 - t) * eps + t * xT

            def vel(x):
                s0 = flow(R, seg, x, [t, t + 1.0], W)[0]
                vc, vu = s0["vc"][0, :, P:].double().numpy(), s0["vu"][0, :, P:].double().numpy()
                return dict(vc=vc, vu=vu, v=(1 + W) * vc - W * vu)
            u0 = vel(z)
            zr = float(torch.sqrt(torch.mean(z[0, :, P:].double() ** 2)))
            for a in (0.01, 0.1, 1.0):
                for j in range(4):
                    d = rng.standard_normal((80, Ltot - P))
                    d *= a * zr / np.sqrt(np.mean(d ** 2))
                    zp = z.clone()
                    zp[0, :, P:] += torch.from_numpy(d).float()
                    u1 = vel(zp)
                    nd = np.linalg.norm(d)
                    r = {f"ratio_{f}": float(np.linalg.norm(u1[f] - u0[f]) / nd) for f in ("vc", "vu", "v")}
                    rows.append(dict(utt=utt, t=t, a=a, j=j, z_rms=zr, **r))
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E12_lip.csv", index=False)
    print(D.groupby(["t", "a"])[["ratio_vc", "ratio_vu", "ratio_v"]].agg(["median", lambda x: x.quantile(0.95)]).round(3).to_string())


if __name__ == "__main__":
    main()
