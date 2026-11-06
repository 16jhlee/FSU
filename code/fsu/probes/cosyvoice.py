"""CosyVoice 2 / Fun-CosyVoice 3 probes (environment env_cosy, GPU). Mel space (80 x T), generated region.

    python -m fsu.probes.cosyvoice theory --decoder cosyvoice2 [--n_utt 50]

The flow decoder is driven through the patched sampler of patches/cosyvoice.diff: t_span_override (step grid),
inference_cfg_rate (w), cfg_branch, cfg_interval, rand_noise (the noise buffer the flow reads its initial state from) and
dump_states (per evaluation: t, x, guided v, v_c, v_u).
"""
import itertools
import sys
from types import SimpleNamespace

import numpy as np
import torch

from fsu.common import CKPT, seed_of
from fsu.probes import (T1_COS2, W1_GRID, branch_summary, cloud_stats, cos_grid, cosv, deviation_rows, main_for, rms,
                        token_cache, utterance_rows)

MODEL_DIRS = {"cosyvoice2": "CosyVoice2-0.5B", "cosyvoice3": "Fun-CosyVoice3-0.5B"}
T10 = cos_grid(10)  # teacher: official 10-step cosine grid
CROSSLING = ["crossling_zh2en", "crossling_en2zh", "crossling_en2ko", "crossling_ko2en", "crossling_en2en",
             "crossling_zh2zh", "crossling_ko2ko"]


class CosyVoiceProbe:
    """Direct access to the CosyVoice flow decoder of one model; one method per measurement."""

    def __init__(self, decoder):
        from fsu.decoders.cosyvoice import CosyVoice
        self.decoder = decoder
        self.cache = token_cache(decoder)
        self.cv = CosyVoice(SimpleNamespace(model_dir=str(CKPT / MODEL_DIRS[decoder]))).cv
        self.m, self.dec = self.cv.model, self.cv.model.flow.decoder

    # ------------------------------------------------------------ flow access
    def segment(self, utt, cache=None):
        """First text segment of the cached conditions of `utt`."""
        return torch.load((cache or self.cache) / f"{utt}.pt")[0]

    @staticmethod
    def noise(utt, k):
        g = torch.Generator().manual_seed(seed_of(f"{utt}|noise|{k}"))
        return torch.randn([1, 80, 50 * 300], generator=g)

    def states(self, seg, noise, **settings):
        """Set the given decoder attributes, run flow.inference from `noise` and return the dumped states."""
        dec, m = self.dec, self.m
        for k, v in settings.items():
            setattr(dec, k, v)
        dec.rand_noise = noise
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

    def run(self, seg, utt, k, ts, w, branch="all", iv=None, noise=None):
        """One sampler run on grid ts with weight w (noise draw k of utt unless `noise` is given). Returns the final
        output and the first-evaluation state eps, v_c, v_u, v, each as float64 arrays of the generated region."""
        st = self.states(seg, self.noise(utt, k) if noise is None else noise, inference_cfg_rate=w, t_span_override=ts,
                         cfg_branch=branch, cfg_interval=iv)
        P = seg["prompt_feat"].shape[1]
        cut = lambda a: a[0, :, P:].double().numpy()  # noqa: E731
        final = cut(st[-1]["x"] + (ts[-1] - ts[-2]) * st[-1]["v"])
        s0 = st[0]
        return dict(final=final, eps=cut(s0["x"]), vc=cut(s0["vc"]), vu=cut(s0["vu"]), v=cut(s0["v"]))

    # ------------------------------------------------------------ measurements
    def theory(self, n_utt=50):
        """Per utterance (8 seeds; RMS over 80 x T_gen):
        lemma1  d = v_c - v_u at t = 0: mean / min pairwise cosine, coefficient of variation and mean of RMS(d);
        prop1   x1(w) = one Euler step from noise, w in {0, 0.35, 0.7, 1.05, 1.5}: quadratic / linear energy of a
                per-element polynomial fit; slope vs g_est = mean(teacher, 8 seeds) - mean(unconditional [w = -1], 8 seeds);
        prop2   z = eps + t1 v(w), t1 in {0.1, 0.2, 0.3, 0.5, 0.7}: distance to the teacher paths (1 - t1) eps + t1 x_k;
        prop3   grid [0, 0.29, NFE-1 uniform steps to 1]: official vs FSU first step, identical later steps;
        prop4   g_all = v_c - v_u(all-null), g_spk = v_c - v_u(speaker-null), g_tok = v_u(speaker-null) - v_u(all-null)."""
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            K = range(8)
            T = [self.run(seg, utt, k, T10, 0.7)["final"] for k in K]
            U = [self.run(seg, utt, k, T10, -1.0)["final"] for k in K]
            g_est = np.mean(T, 0) - np.mean(U, 0)
            one = {w: [self.run(seg, utt, k, [0.0, 1.0], w) for k in K] for w in (0.0, 0.35, 0.7, 1.05, 1.5)}
            D0 = [o["vc"] - o["vu"] for o in one[0.0]]
            pc = [cosv(a, b) for a, b in itertools.combinations(D0, 2)]
            rm = [rms(d) for d in D0]
            rows.append(dict(utt=utt, check="lemma1", cos_mean=np.mean(pc), cos_min=np.min(pc),
                             rms_cv=np.std(rm) / np.mean(rm), rms_d0=np.mean(rm)))
            ws = np.array([0.0, 0.35, 0.7, 1.05, 1.5])
            for k in range(2):
                X = np.stack([one[w][k]["final"].ravel() for w in ws])  # 5 x D
                V = np.vander(ws, 3, increasing=True)  # [1, w, w^2]
                coef = np.linalg.lstsq(V, X, rcond=None)[0]  # 3 x D
                lin, quad = coef[1] * (ws.max() - ws.min()), coef[2] * (ws.max() - ws.min()) ** 2
                slope = coef[1]
                rows.append(dict(utt=utt, check="prop1", k=k, quad_over_lin=float(np.sum(quad ** 2) / np.sum(lin ** 2)),
                                 cos_slope_gest=cosv(slope, g_est), cos_slope_d0=cosv(slope, D0[k]),
                                 rms_slope=rms(slope), rms_gest=rms(g_est)))
            Tm = np.stack(T)
            for t1 in (0.1, 0.2, 0.3, 0.5, 0.7):
                for w in (0.0, 0.35, 0.7, 1.05):
                    for k in range(2):
                        o = one[0.0][k]
                        v = (1 + w) * o["vc"] - w * o["vu"]
                        z = o["eps"] + t1 * v
                        pi = (1 - t1) * o["eps"][None] + t1 * Tm
                        d = min(rms(z - p) for p in pi)
                        rows.append(dict(utt=utt, check="prop2", k=k, t1=t1, w=w, dist=d, dist_over_t1=d / t1))
            d0 = np.mean(rm)
            for n in (2, 4, 8, 16):
                ts = [0.0] + list(np.linspace(0.29, 1.0, n))
                for k in range(2):
                    a = self.run(seg, utt, k, ts, 0.7)["final"]
                    b = self.run(seg, utt, k, ts, 0.7, iv=[0.1, 1.0])["final"]
                    rows.append(dict(utt=utt, check="prop3", k=k, nfe=n, diff=rms(a - b), E1=0.29 * 0.7 * d0))
            for k in range(2):
                a = self.run(seg, utt, k, [0.0, 1.0], 0.7, "all")
                s = self.run(seg, utt, k, [0.0, 1.0], 0.7, "speaker")
                g_all, g_spk, g_tok = a["vc"] - a["vu"], s["vc"] - s["vu"], s["vu"] - a["vu"]
                rows.append(dict(utt=utt, check="prop4", k=k, n_all=np.sum(g_all ** 2), n_spk=np.sum(g_spk ** 2),
                                 n_tok=np.sum(g_tok ** 2), cos_spk_tok=cosv(g_spk, g_tok)))
            print(utt, flush=True)
        return rows

    def branches(self, n_utt=None):
        """g1 = 0.7 (v_c - v_u) at t = 0 (one step [0, 1], noise 0) for the branches all / speaker / token."""
        dec = self.dec
        dec.inference_cfg_rate = 0.7
        dec.t_span_override = [0.0, 1.0]
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            P = seg["prompt_feat"].shape[1]
            G = {}
            for br in ["all", "speaker", "token"]:
                st = self.states(seg, self.noise(utt, 0), cfg_branch=br)[0]
                G[br] = (0.7 * (st["vc"] - st["vu"]))[0, :, P:].numpy().astype("float64")
            rows += branch_summary(utt, G)
            print(utt, flush=True)
        return rows

    def step_conditions(self):
        c = [(f"all_n{n}", "all", cos_grid(n)) for n in [1, 2, 3, 4, 10]]
        c += [(f"speaker_n{n}", "speaker", cos_grid(n)) for n in [2, 3, 10]]
        if self.decoder == "cosyvoice2":
            c += [(f"token_n{n}", "token", cos_grid(n)) for n in [2, 3, 10]]
        c += [(f"t1_{t1}", "all", [0.0, t1, 1.0]) for t1 in [0.1, 0.2, 0.3, 0.5, 0.7]]
        return c

    def steps(self, n_utt=None, grid=False):
        """RMS(v_c - v_u)(t_k) at every evaluation (w = 0.7, noise 0): official grids, speaker / token branches and
        first-step lengths t1 (grid=False), or the 20-step uniform grid (grid=True)."""
        cond_list = [("grid20", "all", list(np.linspace(0, 1, 21)))] if grid else self.step_conditions()
        self.dec.inference_cfg_rate = 0.7
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            P = seg["prompt_feat"].shape[1]
            for name, br, ts in cond_list:
                st = self.states(seg, self.noise(utt, 0), t_span_override=ts, cfg_branch=br)
                for k, s in enumerate(st):
                    d = (s["vc"] - s["vu"])[0, :, P:].double()
                    rows.append(dict(cond=name, utt=utt, k=k, t=ts[k], dt=ts[k + 1] - ts[k],
                                     rms=float(d.pow(2).mean().sqrt())))
            print(utt, flush=True)
        return rows

    def gamma(self, n_utt=None):
        return self.steps(n_utt, grid=True)

    def one_step(self, n_utt=None):
        """x1(w) = 1-step output (w in {0, 0.7, 1.5}), x_c = z + v_c, x_u = z + v_u, teacher T_k (8 seeds):
        lin_resid (collinearity of the three runs), slope_cos = cos(x1(1.5) - x1(0), x1(0) - x_u),
        d_T(w) = RMS(x1(w) - mean T), d_T_xu, d_TT = teacher leave-one-out spread; per w the seed dependence d_seed."""
        def run(seg, utt, k, ts, w):
            st = self.states(seg, self.noise(utt, k), inference_cfg_rate=w, t_span_override=ts)
            P = seg["prompt_feat"].shape[1]
            last = st[-1]
            x_end = (last["x"] + (ts[-1] - ts[-2]) * last["v"])[0, :, P:].double().numpy()
            first = st[0]
            xc = (first["x"] + first["vc"])[0, :, P:].double().numpy()
            xu = (first["x"] + first["vu"])[0, :, P:].double().numpy()
            return x_end, xc, xu
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            X = {w: [] for w in (0.0, 0.7, 1.5)}
            XU, T = [], []
            for k in range(8):
                for w in X:
                    x1, xc, xu = run(seg, utt, k, [0.0, 1.0], w)
                    X[w].append(x1)
                    if w == 0.0:
                        XU.append(xu)
                T.append(run(seg, utt, k, T10, 0.7)[0])
            Tm = np.mean(T, 0)
            d_TT = np.mean([rms(T[k] - np.mean([T[j] for j in range(8) if j != k], 0)) for k in range(8)])
            for k in range(8):
                a, b, c = X[0.0][k], X[0.7][k], X[1.5][k]
                slope = c - a
                rows.append(dict(utt=utt, k=k, lin_resid=rms(b - (a + 0.7 / 1.5 * slope)) / rms(slope),
                                 slope_cos=float((slope * (a - XU[k])).sum() / np.linalg.norm(slope)
                                                 / np.linalg.norm(a - XU[k])),
                                 d_T_w0=rms(a - Tm), d_T_w07=rms(b - Tm), d_T_w15=rms(c - Tm), d_TT=d_TT,
                                 d_T_xu=rms(XU[k] - Tm)))
            for w in X:
                rows.append(dict(utt=utt, k=-1, w=w, d_seed=np.mean([rms(X[w][i] - X[w][j])
                                                                     for i, j in itertools.combinations(range(8), 2)]),
                                 d_seed_T=np.mean([rms(T[i] - T[j]) for i, j in itertools.combinations(range(8), 2)])))
            print(utt, flush=True)
        return rows

    def rho(self, n_utt=100, teachers=8):
        """Teacher (10 steps, w 0.7) seeds 0..teachers-1; X_k = 1-step w = 0 output, G_k = [1-step w = 1] - X_k, k = 0, 1."""
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            T = [self.run(seg, utt, k, T10, 0.7)["final"] for k in range(teachers)]
            X = [self.run(seg, utt, k, [0.0, 1.0], 0.0)["final"] for k in range(2)]
            G = [self.run(seg, utt, k, [0.0, 1.0], 1.0)["final"] - X[k] for k in range(2)]
            rows += cloud_stats(utt, T, X, G)
            print(utt, flush=True)
        return rows

    def deviation_nn(self, n_utt=50):
        """First-step state z = eps + t1 v(w1), t1 = 1 - cos(pi/4): nearest distance to the teacher paths (8 seeds)
        over the w1 grid, seeds k = 0, 1; also g_rms = RMS(v_c - v_u) at t = 0."""
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            Tm = np.stack([self.run(seg, utt, k, T10, 0.7)["final"] for k in range(8)])
            for k in range(2):
                o = self.run(seg, utt, k, [0.0, 1.0], 0.0)
                g = rms(o["vc"] - o["vu"])
                pi = (1 - T1_COS2) * o["eps"][None] + T1_COS2 * Tm
                for w in W1_GRID:
                    z = o["eps"] + T1_COS2 * ((1 + w) * o["vc"] - w * o["vu"])
                    d = min(rms(z - p) for p in pi)
                    rows.append(dict(utt=utt, k=k, t1=T1_COS2, w1=w, dist=d, dist_over_t1=d / T1_COS2, g_rms=g))
            print(utt, flush=True)
        return rows

    def deviation(self, n_utt=50):
        """First-step deviation (nearest and RMS over the 8 teacher paths), t1 = 1 - cos(pi/4), seeds k = 0, 1."""
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            Tm = np.stack([self.run(seg, utt, k, T10, 0.7)["final"] for k in range(8)])
            for k in range(2):
                o = self.run(seg, utt, k, [0.0, 1.0], 0.0)
                deviation_rows(rows, self.decoder, utt, k, o["eps"], o["vc"], o["vu"], Tm, T1_COS2)
            print(utt, flush=True)
        return rows

    def direction(self, n_utt=100):
        """Where the displacement goes. Per utterance u, bases from teacher outputs (10 steps, w 0.7), each the mean over
        seeds k = 0, 1, with the generated-region noise aligned across variants (prompt-region noise drawn separately
        with seed_of(f"{u}|pnoise|{k}"), generated-region noise = noise draw k of u):
          speaker basis: tokens of u with the prompts of u and the next 7 list utterances, top-3 principal directions;
          content basis: prompt of u with the tokens of u and of utterances +10, +20, +30, resized to len(tokens of u)
                         (truncated or tiled), top-3 directions, orthogonalised against the speaker basis.
        Energy shares (speaker, content, residual) of: first-step displacements t1 w g_b (b = all / token / speaker
        null, default noise), final official minus FSU outputs (2 steps), and controls (random Gaussian, teacher seed
        difference, held-out speaker = utterance +8, held-out content = utterance +40); hf share (>= 4 kHz) and frame
        flux of the final mels."""
        T2 = cos_grid(2)
        t1, W = float(T2[1]), 0.7
        meta = utterance_rows()
        utts = [r[0] for r in meta]
        segs = {u: self.segment(u) for u in utts}
        rng = np.random.default_rng(0)
        rows = []
        for i, u in enumerate(utts[:n_utt]):
            su = segs[u]
            spk = [u] + [utts[(i + j) % len(utts)] for j in range(1, 8)]
            spk_ho = utts[(i + 8) % len(utts)]
            con = [u] + [utts[(i + j) % len(utts)] for j in (10, 20, 30)]
            con_ho = utts[(i + 40) % len(utts)]
            n_tok = su["token"].shape[1]

            def teacher(prompt_of, token_of):
                sp = segs[prompt_of]
                seg = dict(token=su["token"] if token_of == u else resize_tokens(segs[token_of]["token"], n_tok),
                           prompt_token=sp["prompt_token"], prompt_feat=sp["prompt_feat"], embedding=sp["embedding"])
                P = sp["prompt_feat"].shape[1]
                return [self.run(seg, None, None, T10, W, noise=aligned_noise(u, k, P))["final"] for k in (0, 1)]
            Ts = {p: teacher(p, u) for p in spk + [spk_ho]}
            Tc = {c: teacher(u, c) for c in con[1:] + [con_ho]}
            Tc[u] = Ts[u]
            L = min(min(v[0].shape[1] for v in Ts.values()), min(v[0].shape[1] for v in Tc.values()))
            mean = lambda v: (v[0][:, :L] + v[1][:, :L]) / 2  # noqa: E731
            Qs = basis([mean(Ts[p]) for p in spk])
            Ct = basis([mean(Tc[c]) for c in con])
            Ct = Ct - Qs @ (Qs.T @ Ct)
            Qt, _ = np.linalg.qr(Ct)
            D = 80 * L
            ctrl = {"random": rng.standard_normal((80, L)), "seed_diff": Ts[u][0][:, :L] - Ts[u][1][:, :L],
                    "heldout_speaker": mean(Ts[spk_ho]) - mean(Ts[u]), "heldout_content": mean(Tc[con_ho]) - mean(Ts[u])}
            for name, d in ctrl.items():
                fs, ft, fr, en = shares(d, Qs, Qt)
                rows.append(dict(utt=u, k=-1, kind="control", what=name, f_spk=fs, f_con=ft, f_res=fr, energy=en, L=L, D=D))
            for k in (0, 1):
                nz = self.noise(u, k)
                for b in ("all", "token", "speaker"):
                    o = self.run(su, None, None, [0.0, 1.0], W, b, noise=nz)
                    d = t1 * W * (o["vc"] - o["vu"])[:, :L]
                    fs, ft, fr, en = shares(d, Qs, Qt)
                    rows.append(dict(utt=u, k=k, kind="first_step", what=f"{b}-null", f_spk=fs, f_con=ft, f_res=fr,
                                     energy=en, L=L, D=D))
                xb = self.run(su, None, None, T2, W, noise=nz)["final"]
                xr = self.run(su, None, None, T2, W, iv=[0.1, 1.0], noise=nz)["final"]
                fs, ft, fr, en = shares((xb - xr)[:, :L], Qs, Qt)
                hb, fb = spectral_features(xb)
                hr, fr_ = spectral_features(xr)
                pr = (xb - xr)[:, :L].ravel()
                res = pr - Qs @ (Qs.T @ pr) - Qt @ (Qt.T @ pr)
                rows.append(dict(utt=u, k=k, kind="final", what="baseline-R", f_spk=fs, f_con=ft, f_res=fr, energy=en,
                                 L=L, D=D, res_energy=float(res @ res) / D, hf_base=hb, hf_R=hr, flux_base=fb,
                                 flux_R=fr_))
            print(i, u, flush=True)
        return rows

    def lipschitz(self, n_utt=50):
        """Finite-difference Lipschitz ratios on the teacher path z_t = (1 - t) eps + t x_T (x_T = teacher output from
        noise draw 0, full sequence), t in {0.1, 0.3, 0.5, 0.7}; Gaussian perturbations on the generated region with
        RMS = a RMS(z_t), a in {0.01, 0.1, 1}, 4 directions; velocities from one Euler step [t, t + 1].
        ratio_f = ||f(z + delta) - f(z)|| / ||delta|| for f = v_c, v_u, v = (1 + w) v_c - w v_u (w 0.7)."""
        W = 0.7

        def flow(seg, x0, ts, w):
            full = torch.zeros([1, 80, 50 * 300])
            full[:, :, :x0.shape[2]] = x0
            return self.states(seg, full, inference_cfg_rate=w, t_span_override=ts, cfg_branch="all", cfg_interval=None)
        rng = np.random.default_rng(0)
        rows = []
        for utt, *_ in utterance_rows(n_utt):
            seg = self.segment(utt)
            P = seg["prompt_feat"].shape[1]
            eps = self.noise(utt, 0)
            st = flow(seg, eps, T10, W)
            xT = st[-1]["x"] + (T10[-1] - T10[-2]) * st[-1]["v"]  # full sequence
            Ltot = xT.shape[2]
            eps = eps[:, :, :Ltot]
            for t in (0.1, 0.3, 0.5, 0.7):
                z = (1 - t) * eps + t * xT

                def vel(x):
                    s0 = flow(seg, x, [t, t + 1.0], W)[0]
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
        return rows

    def crossling(self, n_utt=None):
        """RMS(v_c - v_u) at t = 0 (all-null, w 0.7, noise 0) on every cross-lingual / same-language list."""
        dec = self.dec
        dec.inference_cfg_rate, dec.t_span_override = 0.7, [0.0, 1.0]
        rows = []
        for s in CROSSLING:
            cache = token_cache(self.decoder, s)
            for utt, *_ in utterance_rows(n_utt, s):
                seg = self.segment(utt, cache)
                P = seg["prompt_feat"].shape[1]
                st = self.states(seg, self.noise(utt, 0))[0]
                d = (st["vc"] - st["vu"])[0, :, P:].double()
                rows.append(dict(list=s, utt=utt, rms=float(d.pow(2).mean().sqrt())))
            print(s, flush=True)
        return rows


# ---------------------------------------------------------------- helpers of the direction measurement
def aligned_noise(u, k, P):
    """Noise with an own prompt-region draw (P frames) followed by noise draw k of u for the generated region."""
    gp = torch.Generator().manual_seed(seed_of(f"{u}|pnoise|{k}"))
    pn = torch.randn([1, 80, 2000], generator=gp)[:, :, :P]
    return torch.cat([pn, CosyVoiceProbe.noise(u, k)], 2)[:, :, :50 * 300]


def resize_tokens(tok, n):
    reps = int(np.ceil(n / tok.shape[1]))
    return tok.repeat(1, reps)[:, :n]


def basis(V, k=3):
    X = np.stack([v.ravel() for v in V])
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt[:k].T  # D x k orthonormal


def shares(d, Qs, Qt):
    d = d.ravel()
    e = d @ d
    ps, pt = Qs.T @ d, Qt.T @ d
    es, et = float(ps @ ps), float(pt @ pt)
    return es / e, et / e, 1 - (es + et) / e, e / d.size


_CENTRES = []


def spectral_features(m):
    """hf share (mel bands centred >= 4 kHz, energy exp(2 log-mel)) and frame flux variance of a log-mel (80 x T)."""
    import librosa
    if not _CENTRES:
        _CENTRES.append(librosa.mel_frequencies(n_mels=82, fmin=0, fmax=8000)[1:-1])
    e = np.exp(2 * m.astype(np.float64))
    hf = e[_CENTRES[0] >= 4000].sum() / e.sum()
    flux = np.var(np.linalg.norm(np.diff(m, axis=1), axis=0))
    return hf, flux


def main(argv=None):
    main_for("cosyvoice", CosyVoiceProbe, argv)


if __name__ == "__main__":
    sys.exit(main())
