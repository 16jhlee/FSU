"""v8 §4: measured checks of THEORY_first_step_guidance.md (CosyVoice2 / CosyVoice3), mel space, generated region only.
python scripts/v8_theory_cosy.py cv2|cv3 [n_utt=50]   (env_cosy, GPU)
RMS(a) = sqrt(mean a^2) over (80 x T_gen).  Seeds = noises hash(utt|noise|k).
  Lemma 1  : per utterance, 8 seeds of d = v_c - v_u at t=0: mean pairwise cosine, coefficient of variation of RMS(d).
  Prop 1   : x1(w) = one Euler step from noise, w in {0,0.35,0.7,1.05,1.5}: quadratic-term energy / linear-term energy of a
             per-element polynomial fit (implementation check); slope vs an INDEPENDENT estimate of g_c =
             mean(teacher NFE 10, 8 seeds) - mean(unconditional NFE 10 [w=-1 -> v=v_u], 8 seeds): cosine.
  Prop 2   : first Euler state z_E = eps + t1 v(w) for t1 in {0.1,0.2,0.3,0.5,0.7}, w in {0.35,0.7,1.05}; distance to the
             path set Pi_t1(eps) = {(1-t1) eps + t1 x_k}, x_k = teacher samples (8 seeds); report dist and dist/t1.
             R first state (w=0) gives the averaging term t1*rho.
  Prop 3   : schedule [0, 0.29, then NFE-1 linspace steps to 1], NFE in {2,4,8,16}; same eps, CFG first step vs R first
             step, identical later steps (CFG): RMS(final_CFG - final_R) vs E1 = 0.29*0.7*RMS(d0).
  Prop 4   : at t=0, g_all = v_c - v_u(all-null), g_spk = v_c - v_u(speaker-null), g_tok = v_u(speaker-null) - v_u(all-null)
             (exact nested complement): ||g_all||^2 vs ||g_spk||^2 + ||g_tok||^2 (relative error), cos(g_spk, g_tok).
Output results/E8v8_theory_<model>.csv (one row per utterance x check) + printed summary.
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
MODELS = {"cv2": ("CosyVoice2-0.5B", "cache/cosy2_tokens/en"), "cv3": ("Fun-CosyVoice3-0.5B", "cache/cosy3_tokens/en_full")}


def rms(a):
    return float(np.sqrt(np.mean(a ** 2)))


def cosv(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


class Runner:
    def __init__(self, key):
        ckpt, self.cache = MODELS[key]
        cv = AutoModel(model_dir=str(ROOT / "ckpt" / ckpt))
        self.m, self.dec = cv.model, cv.model.flow.decoder

    def run(self, seg, utt, k, ts, w, branch="all", iv=None):
        dec, m = self.dec, self.m
        dec.inference_cfg_rate, dec.t_span_override, dec.cfg_branch, dec.cfg_interval = w, ts, branch, iv
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
        st, dec.dump_states = dec.dump_states, None
        P = seg["prompt_feat"].shape[1]
        cut = lambda a: a[0, :, P:].double().numpy()  # noqa: E731
        final = cut(st[-1]["x"] + (ts[-1] - ts[-2]) * st[-1]["v"])
        s0 = st[0]
        return dict(final=final, eps=cut(s0["x"]), vc=cut(s0["vc"]), vu=cut(s0["vu"]), v=cut(s0["v"]))


def main():
    key = sys.argv[1]
    n_utt = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    R = Runner(key)
    t10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
    rows = []
    for utt, *_ in read_meta(ROOT / "data/lists/en_100.meta.lst")[:n_utt]:
        seg = torch.load(ROOT / R.cache / f"{utt}.pt")[0]
        K = range(8)
        # teacher and unconditional samples (8 seeds)
        T = [R.run(seg, utt, k, t10, 0.7)["final"] for k in K]
        U = [R.run(seg, utt, k, t10, -1.0)["final"] for k in K]
        g_est = np.mean(T, 0) - np.mean(U, 0)
        # Lemma 1 + Prop 1
        one = {w: [R.run(seg, utt, k, [0.0, 1.0], w) for k in K] for w in (0.0, 0.35, 0.7, 1.05, 1.5)}
        D0 = [o["vc"] - o["vu"] for o in one[0.0]]
        pc = [cosv(a, b) for a, b in itertools.combinations(D0, 2)]
        rm = [rms(d) for d in D0]
        rows.append(dict(utt=utt, check="lemma1", cos_mean=np.mean(pc), cos_min=np.min(pc), rms_cv=np.std(rm) / np.mean(rm),
                         rms_d0=np.mean(rm)))
        ws = np.array([0.0, 0.35, 0.7, 1.05, 1.5])
        for k in range(2):
            X = np.stack([one[w][k]["final"].ravel() for w in ws])           # 5 x D
            V = np.vander(ws, 3, increasing=True)                            # [1, w, w^2]
            coef = np.linalg.lstsq(V, X, rcond=None)[0]                      # 3 x D
            lin, quad = coef[1] * (ws.max() - ws.min()), coef[2] * (ws.max() - ws.min()) ** 2
            slope = coef[1]
            rows.append(dict(utt=utt, check="prop1", k=k, quad_over_lin=float(np.sum(quad ** 2) / np.sum(lin ** 2)),
                             cos_slope_gest=cosv(slope, g_est), cos_slope_d0=cosv(slope, D0[k]),
                             rms_slope=rms(slope), rms_gest=rms(g_est)))
        # Prop 2
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
        # Prop 3
        d0 = np.mean(rm)
        for n in (2, 4, 8, 16):
            ts = [0.0] + list(np.linspace(0.29, 1.0, n))
            for k in range(2):
                a = R.run(seg, utt, k, ts, 0.7)["final"]
                b = R.run(seg, utt, k, ts, 0.7, iv=[0.1, 1.0])["final"]
                rows.append(dict(utt=utt, check="prop3", k=k, nfe=n, diff=rms(a - b), E1=0.29 * 0.7 * d0))
        # Prop 4
        for k in range(2):
            a = R.run(seg, utt, k, [0.0, 1.0], 0.7, "all")
            s = R.run(seg, utt, k, [0.0, 1.0], 0.7, "speaker")
            g_all, g_spk, g_tok = a["vc"] - a["vu"], s["vc"] - s["vu"], s["vu"] - a["vu"]
            rows.append(dict(utt=utt, check="prop4", k=k, n_all=np.sum(g_all ** 2), n_spk=np.sum(g_spk ** 2),
                             n_tok=np.sum(g_tok ** 2), cos_spk_tok=cosv(g_spk, g_tok)))
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / f"results/E8v8_theory_{key}.csv", index=False)
    pd.set_option("display.width", 200)
    for c, g in D.groupby("check"):
        print(c, g.drop(columns=["utt", "check"]).mean(numeric_only=True).round(4).to_dict())


if __name__ == "__main__":
    main()
