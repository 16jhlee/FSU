"""IndexTTS-2 probes (environment env_index, GPU). s2mel CFM in mel space (80 x T), generated region of the first text
segment; per-utterance noise as the runner, seed_all(seed_of(f"{utt}|noise|{k}|0")) before the CFM call.

    python -m fsu.probes.indextts theory [--n_utt 50]

The CFM is driven through patches/indextts.diff: cfg_branch, cfg_interval / cfg_ramp / cfg_steps and dump_states
(per guided evaluation: t, v_c, v_u, prompt_len).
"""
import itertools
import sys

import numpy as np
import torch

from fsu.common import seed_of
from fsu.probes import (branch_summary, cloud_stats, cosv, deviation_rows, main_for, rms, token_cache,
                        utterance_rows)

DECODER = "indextts2"


class IndexTTSProbe:
    """Direct access to the IndexTTS-2 s2mel CFM; one method per measurement."""

    def __init__(self, decoder=DECODER):
        self.cache = token_cache(DECODER)
        from fsu.decoders.indextts import IndexTTS2, seed_all
        self.runner_cls, self.seed_all = IndexTTS2, seed_all
        self.tts = IndexTTS2(None).tts  # changes the working directory to the upstream repository
        self.cfm = self.tts.s2mel.models["cfm"]
        self.dev = self.tts.device

    def segment(self, utt):
        return torch.load(self.cache / f"{utt}.pt")[0]

    def seed(self, utt, k):
        self.seed_all(seed_of(f"{utt}|noise|{k}|0"))

    def infer(self, s, nfe, w):
        dev = self.dev
        with torch.inference_mode():
            return self.cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev),
                                      None, nfe, inference_cfg_rate=w)

    def dumped(self, s, utt, k, nfe, w):
        """Run the CFM with the state dump on; returns the dumped evaluations."""
        cfm = self.cfm
        cfm.dump_states = []
        self.seed(utt, k)
        self.infer(s, nfe, w)
        st, cfm.dump_states = cfm.dump_states, None
        return st

    def output(self, s, utt, k, n, w):
        """CFM output (generated region, float64) with n steps and weight w from noise draw k."""
        self.seed(utt, k)
        vc = self.infer(s, n, w)
        return vc[0, :, s["prompt"].size(-1):].double().cpu().numpy()

    # ------------------------------------------------------------ measurements
    def theory(self, n_utt=50):
        """lemma1: 8 seeds of d = v_c - v_u at t = 0 (pairwise cosine, CV of RMS); prop4: g_all = g_spk + g_tok with
        g_tok = v_u(speaker-null) - v_u(all-null)."""
        cfm = self.cfm

        def step(s, utt, k, branch):
            cfm.cfg_branch = branch
            st = self.dumped(s, utt, k, 1, 0.7)[0]
            P = st["prompt_len"]
            return st["vc"][0, :, P:].double().numpy(), st["vu"][0, :, P:].double().numpy()
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            s = self.segment(utt)
            D = []
            for k in range(8):
                vc, vu = step(s, utt, k, "all")
                D.append(vc - vu)
            pc = [cosv(a, b) for a, b in itertools.combinations(D, 2)]
            rm = [float(np.sqrt(np.mean(d ** 2))) for d in D]
            rows.append(dict(utt=utt, check="lemma1", cos_mean=np.mean(pc), cos_min=np.min(pc),
                             rms_cv=np.std(rm) / np.mean(rm)))
            for k in range(2):
                vc_a, vu_a = step(s, utt, k, "all")
                vc_s, vu_s = step(s, utt, k, "speaker")
                g_all, g_spk, g_tok = vc_a - vu_a, vc_s - vu_s, vu_s - vu_a
                rows.append(dict(utt=utt, check="prop4", k=k, n_all=np.sum(g_all ** 2), n_spk=np.sum(g_spk ** 2),
                                 n_tok=np.sum(g_tok ** 2), cos_spk_tok=cosv(g_spk, g_tok)))
            print(utt, flush=True)
        return rows

    def branches(self, n_utt=None):
        """g1 = 0.7 (v_c - v_u) at t = 0, branches all / speaker (prompt mel + style null) / token (content mu null)."""
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            s = self.segment(utt)
            G = {}
            for br in ["all", "speaker", "token"]:
                self.cfm.cfg_branch = br
                st = self.dumped(s, utt, 0, 1, 0.7)[0]
                G[br] = (0.7 * (st["vc"] - st["vu"]))[0, :, st["prompt_len"]:].numpy().astype("float64")
            rows += branch_summary(utt, G)
            print(utt, flush=True)
        return rows

    CONDITIONS = [(f"all_n{n}", "all", list(np.linspace(0, 1, n + 1))) for n in [1, 2, 3, 4, 10, 25]]
    CONDITIONS += [(f"speaker_n{n}", "speaker", list(np.linspace(0, 1, n + 1))) for n in [2, 3, 10]]
    CONDITIONS += [("t1_0.3", "all", [0.0, 0.3, 1.0])]

    def steps(self, n_utt=None, grid=False):
        """RMS(v_c - v_u)(t_k) at every evaluation (w 0.7, noise 0): official uniform grids, speaker branch, grid
        [0, 0.3, 1] (grid=False) or the 20-step uniform grid (grid=True)."""
        cond_list = [("grid20", "all", list(np.linspace(0, 1, 21)))] if grid else self.CONDITIONS
        cfm = self.cfm
        orig_solve = cfm.solve_euler
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            s = self.segment(utt)
            for name, br, ts in cond_list:
                cfm.solve_euler = orig_solve
                self.runner_cls.install_schedule_solver(cfm, ",".join(str(float(x)) for x in ts), "euler")
                cfm.cfg_branch = br
                st = self.dumped(s, utt, 0, len(ts) - 1, 0.7)
                for k, e in enumerate(st):
                    d = (e["vc"] - e["vu"])[0, :, e["prompt_len"]:].double()
                    rows.append(dict(cond=name, utt=utt, k=k, t=ts[k], dt=ts[k + 1] - ts[k],
                                     rms=float(d.pow(2).mean().sqrt())))
            print(utt, flush=True)
        return rows

    def gamma(self, n_utt=None):
        return self.steps(n_utt, grid=True)

    def rho(self, n_utt=100, teachers=8):
        """Teacher (25 steps, w 0.7) seeds 0..teachers-1; X_k = 1-step w = 0 output, G_k = [1-step w = 1] - X_k."""
        cfm = self.cfm
        cfm.cfg_interval, cfm.cfg_ramp, cfm.cfg_branch = None, None, "all"
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            s = self.segment(utt)
            T = [self.output(s, utt, k, 25, 0.7) for k in range(teachers)]
            X = [self.output(s, utt, k, 1, 0.0) for k in range(2)]
            G = [self.output(s, utt, k, 1, 1.0) - X[k] for k in range(2)]
            rows += cloud_stats(utt, T, X, G)
            print(utt, flush=True)
        return rows

    def deviation(self, n_utt=100):
        """First-step deviation over the w1 grid: uniform 2-step grid (t1 = 0.5), teacher 25 steps (8 seeds); eps = the
        state entering the first estimator call (prompt region zeroed by the CFM), seeds k = 0, 1."""
        cfm = self.cfm
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
            self.seed(utt, k)
            out = self.infer(s, nfe, w)
            st, cfm.dump_states = cfm.dump_states, None
            P = s["prompt"].size(-1)
            return out[0, :, P:].double().cpu().numpy(), st, rec[0][0, :, P:].double().numpy() if rec else None, P
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            s = self.segment(utt)
            Tm = np.stack([go(s, utt, k, 25, 0.7)[0] for k in range(8)])
            for k in range(2):
                _, st, eps, P = go(s, utt, k, 1, 0.7, dump=True)
                vc, vu = st[0]["vc"][0, :, P:].double().numpy(), st[0]["vu"][0, :, P:].double().numpy()
                deviation_rows(rows, DECODER, utt, k, eps, vc, vu, Tm, 0.5)
            print(utt, flush=True)
        return rows

    def one_step_slope(self, n_utt=100):
        """x1(w) = 1-step output, w in {0, 0.7, 1.5}; slope = (x1(1.5) - x1(0)) / 1.5; g_est = mean(teacher 25 steps
        w 0.7, seeds 0-7) - mean(unconditional 25 steps w = -1, seeds 0-7). k in {0, 1}: seeds shared with g_est;
        k in {8, 9}: disjoint seeds."""
        cfm = self.cfm
        cfm.cfg_interval, cfm.cfg_ramp, cfm.cfg_branch = None, None, "all"
        if hasattr(cfm, "cfg_steps"):
            cfm.cfg_steps = None
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            s = self.segment(utt)
            T = [self.output(s, utt, k, 25, 0.7) for k in range(8)]
            U = [self.output(s, utt, k, 25, -1.0) for k in range(8)]
            g_est = np.mean(T, 0) - np.mean(U, 0)
            for k in (0, 1, 8, 9):
                x = {w: self.output(s, utt, k, 1, w) for w in (0.0, 0.7, 1.5)}
                slope = (x[1.5] - x[0.0]) / 1.5
                rows.append(dict(utt=utt, k=k, seeds="shared" if k < 8 else "disjoint",
                                 lin_resid=rms(x[0.7] - (x[0.0] + 0.7 * slope)) / rms(1.5 * slope),
                                 cos_slope_gest=cosv(slope, g_est), rms_slope=rms(slope), rms_gest=rms(g_est)))
            print(utt, flush=True)
        return rows


def main(argv=None):
    main_for("indextts", IndexTTSProbe, argv)


if __name__ == "__main__":
    sys.exit(main())
