"""GLM-TTS probes (environment env_glm, GPU). Flow mel [T, D] of the first text segment, generated region; noise
noise_override = randn([1, 6000, D]) of seed_of(f"{utt}|noise|{k}") as the runner.

    python -m fsu.probes.glmtts dumps [--n_utt 100]

The flow is driven through patches/glmtts.diff: t_span_override, inference_cfg_rate, cfg_branch (all = official =
speaker-null in the released configuration, allnull, token), cfg_interval / cfg_steps / solver and dump_states
(per evaluation: t, x, v, v_c, v_u).
"""
import itertools
import os
import sys

import numpy as np
import torch

from fsu.common import seed_of
from fsu.probes import T1_COS2, cos_grid, deviation_rows, main_for, rms, token_cache, utterance_rows

DECODER = "glmtts"
T10 = cos_grid(10)


class GLMTTSProbe:
    """Direct access to the GLM-TTS flow; one method per measurement."""

    def __init__(self, decoder=DECODER):
        self.cache = token_cache(DECODER)
        from fsu.decoders.glmtts import GLMTTS
        GLMTTS(None)  # working directory and import path of the upstream repository
        from utils import yaml_util
        self.flow = yaml_util.load_flow_model(os.path.join("ckpt", "flow", "flow.pt"),
                                              os.path.join("ckpt", "flow", "config.yaml"), "cuda")

    def segment(self, utt):
        return torch.load(self.cache / f"{utt}.pt")[0]

    def run(self, seg, utt, k, ts, w, branch="all", dump=False):
        """One sampler run on grid ts with weight w and noise draw k. Returns (mel [T, D] float64, dumped states or
        None, number of prompt frames)."""
        flow = self.flow
        flow.inference_cfg_rate, flow.t_span_override, flow.cfg_branch = w, ts, branch
        flow.cfg_interval, flow.cfg_steps, flow.solver = None, None, "euler"
        g = torch.Generator().manual_seed(seed_of(f"{utt}|noise|{k}"))
        flow.noise_override = torch.randn([1, 6000, flow.mel_dim], generator=g)
        flow.dump_states = [] if dump else None
        with torch.inference_mode():
            mel, _ = flow.inference_with_cache(token=seg["token"].cuda(), prompt_token=seg["prompt_token"].cuda(),
                                               prompt_feat=seg["prompt_feat"].cuda(), embedding=seg["embedding"].cuda(),
                                               n_timesteps=len(ts) - 1)
        st, flow.dump_states = flow.dump_states, None
        return mel[0].double().cpu().numpy(), st, seg["prompt_feat"].shape[1]

    # ------------------------------------------------------------ measurements
    def dumps(self, n_utt=100):
        """check g1: RMS(v_c - v_u) at t = 0 per branch (all = official speaker-null, allnull, token; w 0.7, noise 0);
        check gamma: RMS(v_c - v_u) on the 20-step uniform Euler path (w 0.7, official branch);
        check onestep: 1-step w = 0 output vs the teacher (10 steps, w 0.7, 8 seeds): rho_nn, rho_mean, sigma,
        d_TT (teacher leave-one-out spread), g_rms = RMS([1-step w = 1] - [1-step w = 0]), seed_dep."""
        grid = list(np.linspace(0, 1, 21))
        out = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            for br in ("all", "allnull", "token"):
                _, st, P = self.run(seg, utt, 0, [0.0, 1.0], 0.7, br, dump=True)
                d = (st[0]["vc"] - st[0]["vu"])[0, P:, :].double().numpy()
                out.append(dict(utt=utt, check="g1", branch=br, value=rms(d)))
            _, st, P = self.run(seg, utt, 0, grid, 0.7, "all", dump=True)
            for s in st:
                out.append(dict(utt=utt, check="gamma", t=round(s["t"], 4),
                                value=rms((s["vc"] - s["vu"])[0, P:, :].double().numpy())))
            T = [self.run(seg, utt, k, T10, 0.7)[0] for k in range(8)]
            X = [self.run(seg, utt, k, [0.0, 1.0], 0.0)[0] for k in range(2)]
            G = [self.run(seg, utt, k, [0.0, 1.0], 1.0)[0] - X[k] for k in range(2)]
            Tm = np.mean(T, 0)
            sig = np.mean([rms(a - b) for a, b in itertools.combinations(T, 2)])
            d_tt = np.mean([rms(T[j] - np.mean([T[i] for i in range(8) if i != j], 0)) for j in range(8)])
            for k in range(2):
                out.append(dict(utt=utt, check="onestep", k=k, rho_nn=min(rms(X[k] - t) for t in T),
                                rho_mean=rms(X[k] - Tm), sigma=sig, d_TT=d_tt, g_rms=rms(G[k]), seed_dep=rms(X[0] - X[1])))
            print(utt, flush=True)
        return out

    def deviation(self, n_utt=50):
        """First-step deviation over the w1 grid, t1 = 1 - cos(pi/4), teacher 10 steps (8 seeds), branches official
        (speaker-null) and all-null, seeds k = 0, 1; trimmed to the common length of the 1-step state and the teacher."""
        cut = lambda a, P: a[0, P:, :].double().numpy().T  # noqa: E731  [T, D] -> [D, T_gen]
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            Tm = np.stack([self.run(seg, utt, k, T10, 0.7)[0] for k in range(8)])
            for k in range(2):
                for br, lab in (("all", "official (speaker-null)"), ("allnull", "all-null")):
                    _, st, P = self.run(seg, utt, k, [0.0, 1.0], 0.7, br, dump=True)
                    eps, vc, vu = cut(st[0]["x"], P), cut(st[0]["vc"], P), cut(st[0]["vu"], P)
                    L = min(eps.shape[1], Tm.shape[2])
                    deviation_rows(rows, DECODER, utt, k, eps[:, :L], vc[:, :L], vu[:, :L], Tm[:, :, :L], T1_COS2, lab)
            print(utt, flush=True)
        return rows


def main(argv=None):
    main_for("glmtts", GLMTTSProbe, argv)


if __name__ == "__main__":
    sys.exit(main())
