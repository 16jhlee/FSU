"""E8'-9e: direct mel-space check of the 1-step closed form x1(w) = x_c + w (x_c - x_u)  (CosyVoice2, en 100, 8 seeds).
For NFE 1 Euler from noise z at t=0: x1(w) = z + v(w), x_c = z + v_c, x_u = z + v_u (x-predictions, 1 - t = 1).
Reported per utterance/seed (generated region, mel 80 x T, RMS = sqrt(mean square)):
  lin_resid  = RMS(x1(0.7) - [x1(0) + 0.7/1.5 (x1(1.5) - x1(0))]) / RMS(x1(1.5) - x1(0))   (collinearity of 3 runs)
  slope_cos  = cos(x1(1.5) - x1(0), x1(0) - x_u)                                          (slope = conditional - null)
  d_T(w)     = RMS(x1(w) - mean_k teacher_k)      teacher = NFE 10, w 0.7, 8 seeds (same noises)
  d_TT       = mean_k RMS(teacher_k - mean_{k'!=k} teacher)  (teacher's own leave-one-out spread)
  d_seed(w)  = mean over seed pairs RMS(x1(w)_k - x1(w)_k')  (seed dependence of the 1-step output)
env_cosy; CPU is fine. Output results/E8p9e_onestep.csv and a printed summary.
NOTE: collinearity and slope direction are algebraic identities of one Euler step with CFG, so lin_resid / slope_cos
only verify the implementation numerically; the substantive quantities are d_T and d_seed ("1-step ~ conditional mean").
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: F401,E402
from run_cosy import AutoModel, h, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(m, dec, seg, utt, k, ts, w):
    dec.inference_cfg_rate = w
    dec.t_span_override = ts
    g = torch.Generator().manual_seed(h(f"{utt}|noise|{k}"))
    dec.rand_noise = torch.randn([1, 80, 50 * 300], generator=g)
    dec.dump_states = []
    with torch.inference_mode():
        m.flow.inference(token=seg["token"].to(m.device, dtype=torch.int32),
                         token_len=torch.tensor([seg["token"].shape[1]], dtype=torch.int32).to(m.device),
                         prompt_token=seg["prompt_token"].to(m.device),
                         prompt_token_len=torch.tensor([seg["prompt_token"].shape[1]], dtype=torch.int32).to(m.device),
                         prompt_feat=seg["prompt_feat"].to(m.device),
                         prompt_feat_len=torch.tensor([seg["prompt_feat"].shape[1]], dtype=torch.int32).to(m.device),
                         embedding=seg["embedding"].to(m.device), streaming=False, finalize=True)
    st = dec.dump_states
    dec.dump_states = None
    P = seg["prompt_feat"].shape[1]
    last = st[-1]
    x_end = (last["x"] + (ts[-1] - ts[-2]) * last["v"])[0, :, P:].double().numpy()
    first = st[0]
    xc = (first["x"] + first["vc"])[0, :, P:].double().numpy()
    xu = (first["x"] + first["vu"])[0, :, P:].double().numpy()
    return x_end, xc, xu


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def main():
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / "CosyVoice2-0.5B"))
    m, dec = cv.model, cv.model.flow.decoder
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    rows = []
    for utt, *_ in read_meta(ROOT / "data/lists/en_100.meta.lst"):
        seg = torch.load(ROOT / "cache/cosy2_tokens/en" / f"{utt}.pt")[0]
        X = {w: [] for w in (0.0, 0.7, 1.5)}
        XU, T = [], []
        for k in range(8):
            for w in X:
                x1, xc, xu = run(m, dec, seg, utt, k, [0.0, 1.0], w)
                X[w].append(x1)
                if w == 0.0:
                    XU.append(xu)
            T.append(run(m, dec, seg, utt, k, t10, 0.7)[0])
        Tm = np.mean(T, 0)
        d_TT = np.mean([rms(T[k] - np.mean([T[j] for j in range(8) if j != k], 0)) for k in range(8)])
        for k in range(8):
            a, b, c = X[0.0][k], X[0.7][k], X[1.5][k]
            slope = c - a
            rows.append(dict(utt=utt, k=k, lin_resid=rms(b - (a + 0.7 / 1.5 * slope)) / rms(slope),
                             slope_cos=float((slope * (a - XU[k])).sum() / np.linalg.norm(slope) / np.linalg.norm(a - XU[k])),
                             d_T_w0=rms(a - Tm), d_T_w07=rms(b - Tm), d_T_w15=rms(c - Tm), d_TT=d_TT,
                             d_T_xu=rms(XU[k] - Tm)))
        for w in X:
            rows.append(dict(utt=utt, k=-1, w=w, d_seed=np.mean([rms(X[w][i] - X[w][j])
                                                                 for i, j in itertools.combinations(range(8), 2)]),
                             d_seed_T=np.mean([rms(T[i] - T[j]) for i, j in itertools.combinations(range(8), 2)])))
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E8p9e_onestep.csv", index=False)
    A = D[D.k >= 0]
    print(A[["lin_resid", "slope_cos", "d_T_w0", "d_T_w07", "d_T_w15", "d_T_xu", "d_TT"]].describe().round(4).to_string())
    print(D[D.k < 0].groupby("w")[["d_seed", "d_seed_T"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
