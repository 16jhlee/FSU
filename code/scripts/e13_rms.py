"""E13-4 (+ E13-2 dumps): first-step deviation of eq. (4) with the RMS over the teacher samples instead of nearest-of-8.
z_{t1}(w1) = eps + t1 [(1+w1) v_c - w1 v_u] at t = 0 (one dumped evaluation), pi_j = (1-t1) eps + t1 x_j, x_j = teacher
outputs (official NFE, official w, 8 seeds), generated region, RMS = sqrt(mean a^2):
  dist_nn  = min_j RMS(z - pi_j) / t1                 (E10-3 / THEORY_checks definition)
  dist_rms = sqrt(mean_j RMS(z - pi_j)^2) / t1         (E13-4; root-mean-square over p(x|c) estimated by 8 samples)
  g_rms    = RMS(v_c - v_u)
w1 in {-1, -0.7, -0.4333, -0.2, 0, 0.2, 0.35, 0.7, 1.05}; seeds k = 0, 1.
  cv2 / cv3 : cosine NFE 2 t1 = 0.2929, teacher NFE 10 (en first 50, as E10-3)
  idx       : uniform NFE 2 t1 = 0.5, teacher NFE 25 (en 100, E13-2)
  glm       : cosine t1 = 0.2929, teacher NFE 10 official; branches official (speaker-null) and all-null (en first 50)
python scripts/e13_rms.py cv2|cv3|idx|glm [n_utt]  (env_cosy | env_cosy | env_index | env_glm, GPU) -> results/E13_rms_<model>.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
W1 = [-1.0, -0.7, -0.4333, -0.2, 0.0, 0.2, 0.35, 0.7, 1.05]
TC = float(1 - np.cos(np.pi / 4))


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))


def utts(n):
    return [l.split("|")[0] for l in (ROOT / "data/lists/en_100.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()][:n]


def measure(rows, tag, utt, k, eps, vc, vu, Tm, t1, branch="official"):
    pi = (1 - t1) * eps[None] + t1 * Tm
    for w in W1:
        z = eps + t1 * ((1 + w) * vc - w * vu)
        d = np.array([rms(z - p) for p in pi])
        rows.append(dict(model=tag, branch=branch, utt=utt, k=k, t1=t1, w1=w, dist_nn=d.min() / t1,
                         dist_rms=float(np.sqrt(np.mean(d ** 2))) / t1, g_rms=rms(vc - vu)))


def cosy(key, n):
    from v8_theory_cosy import Runner
    R = Runner(key)
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    rows = []
    for utt in utts(n):
        seg = torch.load(ROOT / R.cache / f"{utt}.pt")[0]
        Tm = np.stack([R.run(seg, utt, k, t10, 0.7)["final"] for k in range(8)])
        for k in range(2):
            o = R.run(seg, utt, k, [0.0, 1.0], 0.0)
            measure(rows, key, utt, k, o["eps"], o["vc"], o["vu"], Tm, TC)
        print(utt, flush=True)
    return rows


def index(n):
    from run_index import REPO, h, load_model, seed_all
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    dev = tts.device
    cfm.cfg_interval, cfm.cfg_ramp, cfm.cfg_branch, cfm.cfg_steps = None, None, "all", None
    est, rec = cfm.estimator, []

    class Rec(torch.nn.Module):  # read-only: the state entering each estimator call
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, *a, **kw):
            rec.append(x[:1].detach().float().cpu())
            return self.inner(x, *a, **kw)
    cfm.estimator = Rec(est)

    def go(s, utt, k, nfe, w, dump=False):
        rec.clear()
        cfm.dump_states = [] if dump else None
        seed_all(h(f"{utt}|noise|{k}|0"))
        with torch.inference_mode():
            out = cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev), None, nfe,
                                inference_cfg_rate=w)
        st, cfm.dump_states = cfm.dump_states, None
        P = s["prompt"].size(-1)
        return out[0, :, P:].double().cpu().numpy(), st, rec[0][0, :, P:].double().numpy() if rec else None, P
    rows = []
    for utt in utts(n):
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]
        Tm = np.stack([go(s, utt, k, 25, 0.7)[0] for k in range(8)])
        for k in range(2):
            _, st, eps, P = go(s, utt, k, 1, 0.7, dump=True)
            vc, vu = st[0]["vc"][0, :, P:].double().numpy(), st[0]["vu"][0, :, P:].double().numpy()
            measure(rows, "idx", utt, k, eps, vc, vu, Tm, 0.5)
        print(utt, flush=True)
    return rows


def glm(n):
    from run_glm import h
    repo = ROOT / "repos" / "GLM-TTS"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from utils import yaml_util
    flow = yaml_util.load_flow_model(os.path.join("ckpt", "flow", "flow.pt"), os.path.join("ckpt", "flow", "config.yaml"), "cuda")

    def run(seg, utt, k, ts, w, branch="all", dump=False):
        flow.inference_cfg_rate, flow.t_span_override, flow.cfg_branch = w, ts, branch
        flow.cfg_interval, flow.cfg_steps, flow.solver = None, None, "euler"
        g = torch.Generator().manual_seed(h(f"{utt}|noise|{k}"))
        flow.noise_override = torch.randn([1, 6000, flow.mel_dim], generator=g)
        flow.dump_states = [] if dump else None
        with torch.inference_mode():
            mel, _ = flow.inference_with_cache(token=seg["token"].cuda(), prompt_token=seg["prompt_token"].cuda(),
                                               prompt_feat=seg["prompt_feat"].cuda(), embedding=seg["embedding"].cuda(),
                                               n_timesteps=len(ts) - 1)
        st, flow.dump_states = flow.dump_states, None
        return mel[0].double().cpu().numpy(), st, seg["prompt_feat"].shape[1]
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    cut = lambda a, P: a[0, P:, :].double().numpy().T  # noqa: E731  [T, D] -> [D, T_gen]
    rows = []
    for utt in utts(n):
        seg = torch.load(ROOT / "cache/glm_tokens/en" / f"{utt}.pt")[0]
        Tm = np.stack([run(seg, utt, k, t10, 0.7)[0] for k in range(8)])
        for k in range(2):
            for br, lab in (("all", "official (speaker-null)"), ("allnull", "all-null")):
                _, st, P = run(seg, utt, k, [0.0, 1.0], 0.7, br, dump=True)
                eps, vc, vu = cut(st[0]["x"], P), cut(st[0]["vc"], P), cut(st[0]["vu"], P)
                L = min(eps.shape[1], Tm.shape[2])
                measure(rows, "glm", utt, k, eps[:, :L], vc[:, :L], vu[:, :L], Tm[:, :, :L], TC, lab)
        print(utt, flush=True)
    return rows


def main():
    key = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else (100 if key == "idx" else 50)
    rows = {"cv2": lambda: cosy("cv2", n), "cv3": lambda: cosy("cv3", n), "idx": lambda: index(n), "glm": lambda: glm(n)}[key]()
    import csv
    with open(ROOT / f"results/E13_rms_{key}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    D = pd.DataFrame(rows)
    print(D.groupby(["branch", "w1"])[["dist_nn", "dist_rms", "g_rms"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
