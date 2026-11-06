"""E12 (v10.4) analysis.   python scripts/e12_analyze.py [direction norm rk lip stream e1 xling]   (env_eval)
direction : results/E12_direction.csv + E12 c1_* metrics -> results/E12_direction_summary.csv, E12_direction_corr.csv (X1-X3)
norm      : results/E12_norm_{cv2,cv3,idx}.csv -> results/E12_norm_summary.csv (N1)
rk        : results/E12_rk_<model>.csv -> results/E12_rk_summary.csv (r_hat_k / ||g|| for k = 2, 4, 8 (16, 32 CosyVoice2))
lip       : results/E12_lip.csv -> results/E12_lip_summary.csv, E12_lip_hL.csv
stream    : out/E12/st_h* metrics -> results/E12_stream.csv (S1)
e1        : results/E8p8_law.csv + E11_law.csv -> results/E12_e1_fit.csv (per-model linear fit loss ~ E1)
xling     : out/E8v8/x_* metrics -> results/E12_xling_effect.csv (paired gain R - baseline with 95% CI, d_z)
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e12_jobs as J  # noqa: E402

ROOT = J.ROOT
RES = ROOT / "results"
rng = np.random.default_rng(0)
W = 0.7


def boot_ci(x, f=np.mean, n=5000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = [f(rng.choice(x, len(x))) for _ in range(n)]
    return f(x), *np.percentile(b, [2.5, 97.5])


def spear_ci(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    r = spearmanr(a, b)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        bs.append(spearmanr(a[i], b[i])[0])
    return r[0], *np.nanpercentile(bs, [2.5, 97.5]), r[1], len(a)


# ---------------- E12-1 ----------------
def direction():
    D = pd.read_csv(RES / "E12_direction.csv")
    rows = []
    for (kind, what), g in D.groupby(["kind", "what"], sort=False):
        r = dict(kind=kind, what=what, n=len(g))
        for c in ("f_spk", "f_con", "f_res"):
            m, lo, hi = boot_ci(g[c])
            r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
        r["chance_6_over_D"] = float(np.mean(6 / g.D))
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RES / "E12_direction_summary.csv", index=False)
    F = D[D.kind == "final"].copy()
    for k in (0, 1):
        B = EA.load_metrics(J.wav_dir("c1_base_n2", k, "en"))
        R = EA.load_metrics(J.wav_dir("c1_R_n2", k, "en"))
        if B is None or R is None:
            print("c1 metrics missing for seed", k)
            continue
        m = F.k == k
        F.loc[m, "d_sim"] = (B.sim_eres2net - R.sim_eres2net).reindex(F.loc[m, "utt"]).values
        if "wer" in B and "wer" in R:  # seeds > 0 are SIM-only in queue_eval unless re-evaluated with wer
            F.loc[m, "d_wer"] = (B.wer - R.wer).reindex(F.loc[m, "utt"]).values
        F.loc[m, "d_sim_wavlm"] = (B.sim_wavlm - R.sim_wavlm).reindex(F.loc[m, "utt"]).values
    F["d_hf"], F["d_flux"] = F.hf_base - F.hf_R, F.flux_base - F.flux_R
    C = []
    for x in ("res_energy", "energy"):
        for y in ("d_sim", "d_sim_wavlm", "d_wer", "d_hf", "d_flux"):
            if y in F:
                rho, lo, hi, p, n = spear_ci(F[x], F[y])
                C.append(dict(x=x, y=y, spearman=rho, lo=lo, hi=hi, p=p, n=n))
    C = pd.DataFrame(C)
    C.to_csv(RES / "E12_direction_corr.csv", index=False)
    F.to_csv(RES / "E12_direction_final.csv", index=False)
    g = S.set_index("what")
    v = {}
    v["X1"] = bool(g.loc["all-null", "f_spk"] < 0.30 and g.loc["all-null", "f_res"] >= 0.50)
    c = C.set_index(["x", "y"])
    x2 = g.loc["baseline-R", "f_res"] >= 0.50
    if ("res_energy", "d_sim") in c.index:
        x2 = x2 and c.loc[("res_energy", "d_sim"), "spearman"] <= -0.6 and abs(c.loc[("res_energy", "d_wer"), "spearman"]) < 0.3
    v["X2"] = bool(x2)
    v["X3"] = bool(abs(g.loc["token-null", "f_res"] - g.loc["all-null", "f_res"]) <= 0.10
                   and g.loc["speaker-null", "f_spk"] > g.loc["all-null", "f_spk"])
    pd.set_option("display.width", 250)
    print(S.round(4).to_string())
    print(C.round(3).to_string())
    print("verdicts", v)
    return v


# ---------------- E12-2 ----------------
def norm():
    rows = []
    for key in ("cv2", "cv3", "idx"):
        f = RES / f"E12_norm_{key}.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f)
        P = D.pivot_table(index="utt", columns="what", values="rms")
        S = D[D.what == "z_t1_base"].set_index("utt").step_rms
        Q = {q: D.pivot_table(index="utt", columns="what", values=f"frame_p{q}") for q in (5, 50, 95)}
        r = dict(model=key, n=len(P))
        for a, b in (("x1_base", "x1_teacher"), ("x1_R", "x1_teacher"), ("z_t1_base", "z_t1_R"), ("z_t1_base", "z_t1_teacher"),
                     ("z_t1_R", "z_t1_teacher")):
            m, lo, hi = boot_ci(P[a] / P[b])
            r.update({f"{a}/{b}": m, f"{a}/{b}_lo": lo, f"{a}/{b}_hi": hi})
            for q in (5, 50, 95):
                r[f"{a}/{b}_frame_p{q}"] = float((Q[q][a] / Q[q][b]).mean())
        r["step_rms_over_zR"] = float((S / P["z_t1_R"]).mean())
        for c in ("x1_base", "x1_R", "x1_teacher", "z_t1_base", "z_t1_R", "z_t1_teacher", "eps"):
            r[f"rms_{c}"] = float(P[c].mean())
        r["N1_final"] = bool(0.97 <= r["x1_base/x1_teacher"] <= 1.03 and 0.98 <= r["x1_R/x1_teacher"] <= 1.02)
        r["N1_zt1"] = bool(abs(r["z_t1_base/z_t1_R"] - 1) <= 0.03)
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(RES / "E12_norm_summary.csv", index=False)
    print(S.T.to_string())


# ---------------- E12-3 ----------------
RK_MODELS = [("cv2", "CosyVoice2", "token"), ("cv3", "CosyVoice3", "token"), ("idx", "IndexTTS-2", "token"),
             ("f5", "F5-TTS v0", "text"), ("zip", "ZipVoice", "text")]


def rk():
    rows = []
    for key, name, fam in RK_MODELS:
        f = RES / f"E12_rk_{key}.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f)
        Kmax = len(str(D.d_all.iloc[0]).split(";"))
        for k in [k for k in (1, 2, 4, 8, 16, 32) if k <= Kmax]:
            vals, vals_s = [], []
            for _, r in D.iterrows():
                d = np.array([float(x) for x in r.d_all.split(";")])
                subs = list(itertools.combinations(range(Kmax), k)) if k <= 8 and Kmax <= 8 else \
                    [rng.choice(Kmax, k, replace=False) for _ in range(200)]
                rk_ = np.mean([d[list(s)].min() for s in subs])
                vals.append(rk_ / r.g_rms)
                vals_s.append(rk_ / r.sigma)
            m, lo, hi = boot_ci(vals)
            rows.append(dict(model=name, family=fam, k=k, n_rows=len(D), r_over_g=m, lo=lo, hi=hi,
                             r_over_sigma=float(np.mean(vals_s)), sigma_from_k=Kmax))
    S = pd.DataFrame(rows)
    sep = []
    for k, g in S.groupby("k"):
        tok, txt = g[g.family == "token"], g[g.family == "text"]
        if len(tok) and len(txt):
            sep.append(dict(k=k, max_token=tok.r_over_g.max(), min_text=txt.r_over_g.min(),
                            separated=bool(tok.r_over_g.max() < txt.r_over_g.min()),
                            sep_ci=bool(tok.hi.max() < txt.lo.min())))
    S.to_csv(RES / "E12_rk_summary.csv", index=False)
    print(S.round(4).to_string())
    print(pd.DataFrame(sep).to_string())


# ---------------- E12-4 ----------------
def lip():
    D = pd.read_csv(RES / "E12_lip.csv")
    q95 = lambda x: x.quantile(0.95)  # noqa: E731
    S = D.groupby(["t", "a"])[["ratio_vc", "ratio_vu", "ratio_v"]].agg(["median", q95])
    S.columns = [f"{a}_{'med' if b == 'median' else 'p95'}" for a, b in S.columns]
    S = S.reset_index()
    S.to_csv(RES / "E12_lip_summary.csv", index=False)
    rows = []
    sched = {2: [0.0, 1 - np.cos(np.pi / 4), 1.0], 3: list(1 - np.cos(np.pi / 2 * np.arange(4) / 3))}
    ts = sorted(D.t.unique())
    for nfe, grid in sched.items():
        for i in range(nfe):
            t0, h = grid[i], grid[i + 1] - grid[i]
            tn = min(ts, key=lambda x: abs(x - max(t0, 0.1)))
            for a in sorted(D.a.unique()):
                g = S[(S.t == tn) & (S.a == a)].iloc[0]
                L = g.ratio_vc_med
                rows.append(dict(nfe=nfe, step=i + 1, t_start=t0, h=h, t_measured=tn, a=a, L_med=L, L_p95=g.ratio_vc_p95,
                                 Ltilde=(1 + 2 * W) * L, hLtilde=h * (1 + 2 * W) * L, hLtilde_p95=h * (1 + 2 * W) * g.ratio_vc_p95,
                                 L_guided_med=g.ratio_v_med, hL_guided=h * g.ratio_v_med))
    H = pd.DataFrame(rows)
    H.to_csv(RES / "E12_lip_hL.csv", index=False)
    print(S.round(3).to_string())
    print(H.round(3).to_string())


# ---------------- E12-5 ----------------
def stream():
    rows = []
    off = pd.read_csv(RES / "E8v8_stream.csv")
    off = off[(off.set == "long_en") & (off.nfe == 2)].set_index("mode")
    for H in (25, 50, 100):
        Bk, Rk, Tk = (EA.load_metrics(J.wav_dir(f"st_h{H}_{c}", 0, "long_en")) for c in ("def_n2", "R_n2", "n10"))
        if Bk is None or Rk is None or Tk is None:
            continue
        ch = pd.read_csv(J.wav_dir(f"st_h{H}_def_n2", 0, "long_en") / "_chunks.tsv", sep="\t", header=None)
        r = dict(hop=H, n=len(Bk), chunks=float(ch[2].astype(str).str.count(",").add(2).mean()))
        c = EA.compare(Rk, Bk, Tk)
        gap = Tk.sim_eres2net.mean() - Bk.sim_eres2net.mean()
        r.update(teacher=Tk.sim_eres2net.mean(), base=Bk.sim_eres2net.mean(), R=Rk.sim_eres2net.mean(), loss=gap,
                 gain=c["vs_base_d_sim_eres2net"], gain_lo=c["vs_base_d_sim_eres2net_lo"], gain_hi=c["vs_base_d_sim_eres2net_hi"],
                 gain_p=c["vs_base_d_sim_eres2net_p"], rec=c["vs_base_d_sim_eres2net"] / gap if gap >= 0.42 else np.nan,
                 wer_B=Bk.wer.mean(), wer_R=Rk.wer.mean(), dwer_p=c.get("vs_base_d_wer_p"), utmos_B=Bk.utmos.mean(),
                 utmos_R=Rk.utmos.mean())
        rr = []
        for _ in range(2000):  # bootstrap CI of the recovery ratio (utterances resampled jointly)
            i = rng.integers(0, len(Bk), len(Bk))
            idx = Bk.index[i]
            g_ = Tk.loc[idx].sim_eres2net.mean() - Bk.loc[idx].sim_eres2net.mean()
            rr.append((Rk.loc[idx].sim_eres2net.mean() - Bk.loc[idx].sim_eres2net.mean()) / g_)
        r["rec_lo"], r["rec_hi"] = np.percentile(rr, [2.5, 97.5])
        rows.append(r)
    S = pd.DataFrame(rows)
    S["rec_official_stream"], S["rec_offline"] = off.loc["str", "rec"], off.loc["off", "rec"]
    S.to_csv(RES / "E12_stream.csv", index=False)
    print(S.round(3).to_string())
    if len(S) == 3:
        mono = bool(S.rec.is_monotonic_increasing)
        near = bool(abs(S.set_index("hop").loc[100, "rec"] - off.loc["off", "rec"]) <= 0.05)
        print("S1 monotone", mono, "| hop 100 within 5 %p of offline", near)


# ---------------- E12-6.1 ----------------
def e1():
    L = pd.read_csv(RES / "E8p8_law.csv")
    G = pd.read_csv(RES / "E11_law.csv").assign(model="GLM-TTS")
    D = pd.concat([L[["model", "E1", "loss", "config"]], G[["model", "E1", "loss", "config"]]])
    rows = []
    for m, g in D.groupby("model", sort=False):
        x, y = g.E1.values, g.loss.values
        A = np.vstack([x, np.ones_like(x)]).T
        (b, a), res, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ [b, a]
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        rho = spearmanr(x, y)
        rows.append(dict(model=m, n=len(g), slope=b, intercept=a, R2=r2, spearman=rho[0], spearman_p=rho[1],
                         E1_range=f"{x.min():.3f}-{x.max():.3f}", loss_range=f"{y.min():.2f}-{y.max():.2f}"))
    S = pd.DataFrame(rows)
    S.to_csv(RES / "E12_e1_fit.csv", index=False)
    print(S.round(3).to_string())


# ---------------- E12-6.2 ----------------
def xling():
    import e8v8x_jobs as X
    rows = []
    for model, name in (("cv2", "CosyVoice2"), ("cv3", "CosyVoice3")):
        for s, _ in X.LISTS:
            T = EA.load_metrics(X.wav_dir(f"x_{model}_n10_def", 0, s))
            for n in (2, 3):
                B, R = EA.load_metrics(X.wav_dir(f"x_{model}_n{n}_def", 0, s)), EA.load_metrics(X.wav_dir(f"x_{model}_n{n}_R", 0, s))
                d = (R.sim_eres2net - B.sim_eres2net).dropna()
                m, lo, hi = boot_ci(d)
                loss = T.sim_eres2net.mean() - B.sim_eres2net.mean()
                rows.append(dict(model=name, list=s, cross=s[3:5] != s[6:8], nfe=n, n=len(d), gain=m, lo=lo, hi=hi,
                                 d_z=float(d.mean() / d.std(ddof=1)), p=wilcoxon(d).pvalue, loss=loss,
                                 rec=m / loss if loss >= 0.42 else np.nan))
    S = pd.DataFrame(rows)
    S["p_holm"] = EA.holm(S.p.values)
    S.to_csv(RES / "E12_xling_effect.csv", index=False)
    print(S.round(3).to_string())


if __name__ == "__main__":
    todo = sys.argv[1:] or ["direction", "norm", "rk", "lip", "stream", "e1", "xling"]
    for t in todo:
        print(f"===== {t}")
        try:
            globals()[t]()
        except FileNotFoundError as e:
            print("missing:", e)
