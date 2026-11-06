"""E12-2 (v10.4): mel STATE norms (the quantity of Stage-wise CFG's norm-explosion argument), en_100, seed 0, generated
region only. RMS(a) = sqrt(mean a^2) over (80 x L); frame RMS = per-frame sqrt(mean over bins a^2).
  z_t1   : first Euler state of NFE 2 with guidance (baseline, w = 0.7) and without (R, w = 0), and the teacher's state at
           the same t (CosyVoice: cosine NFE 10 grid contains t = 1 - cos(pi/4) exactly (step 5); IndexTTS-2: linear
           NFE 25 grid has t = 0.48 / 0.52 around t1 = 0.5 -> linear interpolation of the two recorded states).
           Geometry check: RMS(z_base) vs the prediction sqrt(RMS(z_R)^2 + 2<z_R, t1 w g>/D + RMS(t1 w g)^2) is exact by
           construction, so we report the measured ratio and the size of t1 w RMS(g) relative to RMS(z_R).
  x1     : final mel of baseline NFE 2, R NFE 2, teacher (official NFE, w 0.7): RMS and frame-RMS percentiles.
python scripts/e12_norm.py cv2|cv3|idx [n_utt=100]  (env_cosy | env_cosy | env_index; idx under the heavy_gen lock)
-> results/E12_norm_<model>.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
W = 0.7


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))


def frame_q(a):
    f = np.sqrt(np.mean(np.asarray(a, np.float64) ** 2, 0))
    return {f"frame_p{q}": float(np.percentile(f, q)) for q in (5, 50, 95)}


def row(utt, model, what, a, t=None):
    r = dict(utt=utt, model=model, what=what, t=t, rms=rms(a))
    r.update(frame_q(a))
    return r


def cosy(key, utts):
    from v8_theory_cosy import Runner
    R = Runner(key)
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    t2 = list(1 - np.cos(np.pi / 2 * np.arange(3) / 2))
    t1 = t2[1]
    assert abs(t10[5] - t1) < 1e-9
    rows = []
    for utt in utts:
        seg = torch.load(ROOT / R.cache / f"{utt}.pt")[0]
        dec = R.dec
        P = seg["prompt_feat"].shape[1]
        cut = lambda a: a[0, :, P:].double().numpy()  # noqa: E731
        res = {}  # as Runner.run, keeping every dumped state
        for name, ts, iv in (("base", t2, None), ("R", t2, [0.1, 1.0]), ("teacher", t10, None)):
            dec.cfg_interval = iv
            g = torch.Generator().manual_seed(R_h(f"{utt}|noise|0"))
            dec.rand_noise = torch.randn([1, 80, 50 * 300], generator=g)
            dec.inference_cfg_rate, dec.t_span_override, dec.cfg_branch = W, ts, "all"
            dec.dump_states = []
            m = R.m
            with torch.inference_mode():
                m.flow.inference(token=seg["token"].to(m.device, dtype=torch.int32),
                                 token_len=torch.tensor([seg["token"].shape[1]], dtype=torch.int32).to(m.device),
                                 prompt_token=seg["prompt_token"].to(m.device),
                                 prompt_token_len=torch.tensor([seg["prompt_token"].shape[1]], dtype=torch.int32).to(m.device),
                                 prompt_feat=seg["prompt_feat"].to(m.device),
                                 prompt_feat_len=torch.tensor([seg["prompt_feat"].shape[1]], dtype=torch.int32).to(m.device),
                                 embedding=seg["embedding"].to(m.device), streaming=False, finalize=True)
            st, dec.dump_states = dec.dump_states, None
            res[name] = dict(states=[cut(s["x"]) for s in st], final=cut(st[-1]["x"] + (ts[-1] - ts[-2]) * st[-1]["v"]),
                             g=cut(st[0]["vc"] - st[0]["vu"]))
        zb, zr, zt = res["base"]["states"][1], res["R"]["states"][1], res["teacher"]["states"][5]
        rows += [row(utt, key, "z_t1_base", zb, t1), row(utt, key, "z_t1_R", zr, t1), row(utt, key, "z_t1_teacher", zt, t1),
                 row(utt, key, "eps", res["base"]["states"][0], 0.0)]
        rows[-4]["step_rms"] = rms(t1 * W * res["base"]["g"])
        for n in ("base", "R", "teacher"):
            rows.append(row(utt, key, f"x1_{n}", res[n]["final"], 1.0))
        print(utt, flush=True)
    return rows


def R_h(s):
    from run_cosy import h
    return h(s)


def index(utts):
    from run_index import REPO, h, load_model, seed_all
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    dev = tts.device
    cfm.cfg_interval, cfm.cfg_ramp, cfm.cfg_branch = None, None, "all"
    est = cfm.estimator
    rec = []

    class Rec(torch.nn.Module):  # read-only: records the state x and t entering every estimator call
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, *a, **kw):
            rec.append((float(a[2][0]), x[:1].detach().float().cpu()))
            return self.inner(x, *a, **kw)
    cfm.estimator = Rec(est)

    def go(s, utt, n, w, iv=None):
        cfm.cfg_interval = iv
        rec.clear()
        seed_all(h(f"{utt}|noise|0|0"))
        with torch.inference_mode():
            out = cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev), None, n,
                                inference_cfg_rate=w)
        P = s["prompt"].size(-1)
        return out[0, :, P:].double().cpu().numpy(), [(t, x[0, :, P:].double().numpy()) for t, x in rec]
    rows = []
    for utt in utts:
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]
        xb, sb = go(s, utt, 2, W)
        xr, sr = go(s, utt, 2, W, [0.1, 1.0])
        xt, stt = go(s, utt, 25, W)
        t1 = sb[1][0]
        lo = max((x for x in stt if x[0] <= t1 + 1e-6), key=lambda x: x[0])
        hi = min((x for x in stt if x[0] >= t1 - 1e-6), key=lambda x: x[0])
        a = 0.0 if hi[0] == lo[0] else (t1 - lo[0]) / (hi[0] - lo[0])
        zt = (1 - a) * lo[1] + a * hi[1]
        rows += [row(utt, "idx", "z_t1_base", sb[1][1], t1), row(utt, "idx", "z_t1_R", sr[1][1], t1),
                 row(utt, "idx", "z_t1_teacher", zt, t1), row(utt, "idx", "eps", sb[0][1], 0.0)]
        rows[-4]["step_rms"] = rms(sb[1][1] - sr[1][1])
        for n, x in (("base", xb), ("R", xr), ("teacher", xt)):
            rows.append(row(utt, "idx", f"x1_{n}", x, 1.0))
        print(utt, flush=True)
    return rows


def main():
    key = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    utts = [l.split("|")[0] for l in (ROOT / "data/lists/en_100.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    rows = index(utts) if key == "idx" else cosy(key, utts)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / f"results/E12_norm_{key}.csv", index=False)
    print(D.groupby("what")[["rms", "frame_p5", "frame_p50", "frame_p95"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
