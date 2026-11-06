"""E11 block C dumps for GLM-TTS (en first 100 utterances, first text segment, generated region of the flow mel; RMS =
sqrt(mean a^2); noise hash(utt|noise|k)). Procedures follow the CosyVoice measurements:
  g1     t = 0 guidance RMS(v_c - v_u) per unconditional branch: all = official (speaker-null in the released config,
         speech_token_cfg False), allnull (tokens zeroed too), token (tokens only) (one Euler step, w 0.7, k = 0)
         [E8'-8 block D / Table 3]
  gamma  gamma(t) = RMS(v_c - v_u) on the 20-step linspace Euler path (w 0.7, all-null) at t = 0, 0.05, ..., 0.95
         [v8 §1.3 grid]; gamma(t)/gamma(0) reported on {0, .05, .1, .2, .3, .5, .7, .9}
  onestep NFE 1 w = 0 output xbar vs teacher (NFE 10, w 0.7) 8 seeds: d_T = RMS(xbar - mean T), d_TT = teacher
         leave-one-out spread [E8'-9e]; rho_nn / rho_mean / sigma / g_rms exactly as scripts/e10_rho.py [E10-7]
python scripts/e11_dumps.py [n_utt=100]   (env_glm, GPU) -> results/E11_dumps.csv
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "repos" / "GLM-TTS"
sys.path.insert(0, str(ROOT / "scripts"))
from run_glm import h, read_meta  # noqa: E402


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))


def main():
    n_utt = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows_meta = read_meta(ROOT / "data/lists/en_100.meta.lst")[:n_utt]
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
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
        P = seg["prompt_feat"].shape[1]
        return mel[0].double().cpu().numpy(), st, P
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    grid = list(np.linspace(0, 1, 21))
    out = []
    for utt, *_ in rows_meta:
        seg = torch.load(ROOT / "cache/glm_tokens/en" / f"{utt}.pt")[0]
        # g1 per branch
        for br in ("all", "allnull", "token"):  # all = official (= speaker-null here); allnull; token-null
            _, st, P = run(seg, utt, 0, [0.0, 1.0], 0.7, br, dump=True)
            d = (st[0]["vc"] - st[0]["vu"])[0, P:, :].double().numpy()
            out.append(dict(utt=utt, check="g1", branch=br, value=rms(d)))
        # gamma(t)
        _, st, P = run(seg, utt, 0, grid, 0.7, "all", dump=True)
        for s in st:
            out.append(dict(utt=utt, check="gamma", t=round(s["t"], 4), value=rms((s["vc"] - s["vu"])[0, P:, :].double().numpy())))
        # one-step closed form / rho
        T = [run(seg, utt, k, t10, 0.7)[0] for k in range(8)]
        X = [run(seg, utt, k, [0.0, 1.0], 0.0)[0] for k in range(2)]
        G = [run(seg, utt, k, [0.0, 1.0], 1.0)[0] - X[k] for k in range(2)]
        Tm = np.mean(T, 0)
        sig = np.mean([rms(a - b) for a, b in itertools.combinations(T, 2)])
        d_tt = np.mean([rms(T[j] - np.mean([T[i] for i in range(8) if i != j], 0)) for j in range(8)])
        for k in range(2):
            out.append(dict(utt=utt, check="onestep", k=k, rho_nn=min(rms(X[k] - t) for t in T), rho_mean=rms(X[k] - Tm),
                            sigma=sig, d_TT=d_tt, g_rms=rms(G[k]), seed_dep=rms(X[0] - X[1])))
        print(utt, flush=True)
    import csv
    keys = sorted({k for r in out for k in r})
    with open(ROOT / "results/E11_dumps.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print("rows", len(out))


if __name__ == "__main__":
    main()
