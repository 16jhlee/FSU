"""E13 (v10.5) analysis.  python scripts/e13_analyze.py [grid ctrl mask10 idxsweep rms glmT f5 tost tg ddc]  (env_eval)
grid     E13-6(a) -> results/E13_grid.csv                 (J1, J2)
ctrl     E13-6(b) -> results/E13_ctrl.csv, E13_ctrl_recovery.csv, E13_ctrl_diff.csv   (J3, J4, J5)
mask10   E13-3    -> results/E13_mask10.csv               (M1–M3; geometry 8 seeds en 100)
idxsweep E13-2    -> results/E13_idx_wsweep.csv           (I1–I4; I2 from results/E13_rms_idx.csv)
rms      E13-4    -> results/E13_rms_fit.csv              (R1–R3)
glmT     E13-5    -> results/E13_glm_teacher.csv          (G1–G3)
f5       E13-8    -> results/E13_f5_sway.csv              (S1, S2)
tost     E13-9.1  -> results/E13_tost.csv                 (T1, T2)
tg       E13-9.2  -> results/E13_total_guidance_glm.csv
ddc      E13-9.3  -> results/E13_ddcond_3emb.csv
SIM = ERes2Net x100 (WavLM / ECAPA alongside), WER/CER %, eps = 0.42 (SIM resolution).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e13_jobs as J  # noqa: E402

ROOT = J.ROOT
RES = ROOT / "results"
EPS = 0.42
rng = np.random.default_rng(0)
SIMS = ["sim_eres2net", "sim_wavlm", "sim_ecapa"]


def L(name, k=0, s="en"):
    return EA.load_metrics(J.wav_dir(name, k, s))


def boot(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = rng.choice(x, (5000, len(x))).mean(1)
    return x.mean(), *np.percentile(b, q)


def paired(X, Y, col="sim_eres2net"):
    c = X.index.intersection(Y.index)
    d = (X.loc[c, col] - Y.loc[c, col]).dropna()
    m, lo, hi = boot(d)
    return m, lo, hi, (wilcoxon(d).pvalue if np.any(d != 0) else 1.0), len(d)


# ---------------- E13-6(a) ----------------
def grid():
    rows = []
    for s in ("en", "zh"):
        for a in J.W1G:
            for b in J.W2G:
                X = L(f"c2g_{a}_{b}", 0, s)
                if X is None:
                    continue
                rows.append(dict(set=s, w1=float(a), w2=float(b), n=len(X), **{c: X[c].mean() for c in SIMS + ["wer", "utmos"] if c in X}))
    G = pd.DataFrame(rows)
    G.to_csv(RES / "E13_grid.csv", index=False)
    out = []
    for (s, b), g in G.groupby(["set", "w2"]):
        best = g.loc[g.sim_eres2net.idxmax()]
        at0 = g[g.w1 == 0].sim_eres2net.iloc[0]
        out.append(dict(set=s, w2=b, argmax_w1=best.w1, sim_best=best.sim_eres2net, sim_w1_0=at0, J1=bool(best.w1 == 0 or best.sim_eres2net - at0 <= EPS)))
    O = pd.DataFrame(out)
    R0 = G[(G.w1 == 0) & (G.w2 == 0.7)].set_index("set").sim_eres2net
    J2 = {s: float(G[(G.set == s) & (G.w1 == 0)].sim_eres2net.max() - R0[s]) for s in R0.index}
    print(G.round(2).to_string(index=False))
    print(O.to_string(index=False))
    print("J2: best (0, w2*) - R(0, 0.7):", {k: round(v, 3) for k, v in J2.items()}, "-> J2", all(v < EPS for v in J2.values()))
    return G


# ---------------- E13-6(b) ----------------
DECS = {"c2": "CosyVoice2", "c3": "CosyVoice3", "ix": "IndexTTS-2", "gs": "GLM-TTS speaker-null", "ga": "GLM-TTS all-null"}
CONDS = ["11", "01", "10", "00", "h", "w03"]


def cname(d, c):
    return "gs_00" if (d == "ga" and c == "00") else f"{d}_{c}"


def ctrl():
    rows, diffs, recs = [], [], []
    for d, dn in DECS.items():
        for s in ("en", "zh"):
            for k in (0, 1, 2):
                M = {c: L(cname(d, c), k, s) for c in CONDS}
                if any(v is None for v in M.values()):
                    continue
                for c, X in M.items():
                    rows.append(dict(decoder=dn, set=s, seed=k, cond=c, n=len(X), **{m: X[m].mean() for m in SIMS + ["wer", "utmos"] if m in X}))
                for other in ("10", "00", "h"):
                    m, lo, hi, p, n = paired(M["01"], M[other])
                    diffs.append(dict(decoder=dn, set=s, seed=k, comparison=f"01 - {other}", d_sim=m, lo=lo, hi=hi, p=p, n=n,
                                      ge_eps=bool(m >= EPS)))
                for tn, tname in ([("speaker-null teacher", "gs_T"), ("all-null teacher", "ga_T")] if d == "ga" else [("teacher", f"{'gs' if d == 'gs' else d}_T")]):
                    T = L(tname, k, s)
                    if T is None:
                        continue
                    gap = T.sim_eres2net.mean() - M["11"].sim_eres2net.mean()
                    recs.append(dict(decoder=dn, set=s, seed=k, teacher=tn, gap=gap,
                                     recovery=(M["01"].sim_eres2net.mean() - M["11"].sim_eres2net.mean()) / gap if gap >= EPS else np.nan))
    C, D, R = pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(recs)
    if len(D):
        D["p_holm"] = EA.holm(D.p.values)
    C.to_csv(RES / "E13_ctrl.csv", index=False)
    D.to_csv(RES / "E13_ctrl_diff.csv", index=False)
    R.to_csv(RES / "E13_ctrl_recovery.csv", index=False)
    if len(D):
        print(D.round(3).to_string(index=False))
        main = D[D.decoder != "GLM-TTS speaker-null"]
        print("J3 (CV2, CV3, IndexTTS-2, GLM all-null; every seed / set / comparison >= eps):", bool(main.ge_eps.all()),
              f"({int(main.ge_eps.sum())}/{len(main)})")
        print("J5 GLM speaker-null cells >= eps:", int(D[D.decoder == 'GLM-TTS speaker-null'].ge_eps.sum()), "/",
              int((D.decoder == 'GLM-TTS speaker-null').sum()))
    if len(R):
        S = R.groupby(["decoder", "set", "teacher"]).recovery.agg(["mean", "std", "count"]).reset_index()
        S["J4_sd_le_3pp"] = S["std"] <= 0.03
        print(S.round(3).to_string(index=False))
    return C, D, R


# ---------------- E13-3 ----------------
def mask10():
    rows = []
    for s in ("en", "zh"):
        B, R0, M10, Z = L("c2_11", 0, s), L("c2_01", 0, s), L("c2_10", 0, s), L("c2_00", 0, s)
        if M10 is None:
            continue
        r = dict(set=s, sim_11=B.sim_eres2net.mean(), sim_01=R0.sim_eres2net.mean(), sim_10=M10.sim_eres2net.mean(),
                 sim_00=Z.sim_eres2net.mean())
        r["M1"] = bool(r["sim_10"] <= r["sim_11"] + EPS and r["sim_10"] < r["sim_00"])
        rows.append(r)
    G = EA.geometry_on([("CosyVoice2", "baseline (11)", "c2_11"), ("CosyVoice2", "first step only (10)", "c2_10")],
                       {"en": "data/lists/en_100"}, J.wav_dir)
    G = G[(G.emb == "eres2net") & (G.measure == "ddcond")]
    ddc = G.set_index("label").value
    out = pd.DataFrame(rows)
    out["ddcond_11"], out["ddcond_10"] = ddc.get("baseline (11)", np.nan), ddc.get("first step only (10)", np.nan)
    out["M2"] = out.ddcond_10 >= 0.8 * out.ddcond_11
    out.to_csv(RES / "E13_mask10.csv", index=False)
    print(out.round(3).to_string(index=False))


# ---------------- E13-2 ----------------
def idxsweep():
    rows = []
    for s in ("en", "zh"):
        for w in J.W1S:
            X = L(f"ix_w{w}", 0, s)
            if X is None:
                continue
            rows.append(dict(set=s, w1=float(w), n=len(X), **{c: X[c].mean() for c in SIMS + ["wer", "utmos"] if c in X}))
    D = pd.DataFrame(rows)
    D.to_csv(RES / "E13_idx_wsweep.csv", index=False)
    print(D.round(2).to_string(index=False))
    for s, g in D.groupby("set"):
        g = g.set_index("w1")
        best = g.sim_eres2net.idxmax()
        i1 = best in (0.0, 0.2) or g.sim_eres2net.max() - g.loc[0.0, "sim_eres2net"] <= EPS
        i3 = (g.loc[0.0, "sim_eres2net"] - g.loc[-0.7, "sim_eres2net"]) > (g.loc[0.0, "sim_eres2net"] - g.loc[0.7, "sim_eres2net"])
        print(s, "I1 argmax", best, i1, "| I3 asym", bool(i3), "| I4 WER(-1) =", round(g.loc[-1.0, "wer"], 1), bool(g.loc[-1.0, "wer"] > 30))
    f = RES / "E13_rms_idx.csv"
    if f.exists():
        R = pd.read_csv(f).groupby("w1")[["dist_nn", "dist_rms", "g_rms"]].mean()
        sl_p = (R.loc[1.05] - R.loc[0.7]) / 0.35
        sl_n = (R.loc[-1.0] - R.loc[-0.7]) / 0.3
        print("I2 large-|w1| slopes dist_rms +/-:", round(sl_p.dist_rms, 3), round(sl_n.dist_rms, 3), "vs ||g|| 1.93 (+-10%)")


# ---------------- E13-4 ----------------
SIG = {"cv2": ("E10_rho_cv2", 2.11), "cv3": ("E10_rho_cv3", 2.10), "idx": ("E10_rho_idx", 1.93)}


def rms_fit():
    out = []
    for key in ("cv2", "cv3", "idx", "glm"):
        f = RES / f"E13_rms_{key}.csv"
        if not f.exists():
            continue
        D = pd.read_csv(f)
        for br, g in D.groupby("branch"):
            M = g.groupby("w1")[["dist_nn", "dist_rms", "g_rms"]].mean()
            w = M.index.values
            for col in ("dist_rms", "dist_nn"):
                A = np.vstack([np.ones_like(w), w ** 2]).T
                (a2, b2), *_ = np.linalg.lstsq(A, M[col].values ** 2, rcond=None)
                a, b = np.sqrt(max(a2, 0)), np.sqrt(max(b2, 0))
                pred = np.sqrt(a ** 2 + (b * w) ** 2)
                g_ref = float(g.g_rms.mean())
                sig = np.nan
                if key in SIG:
                    sig = pd.read_csv(RES / f"{SIG[key][0]}.csv").sigma.mean()
                elif key == "glm":
                    E = pd.read_csv(RES / "E11_dumps.csv")
                    sig = E[E.check == "onestep"].sigma.mean()
                out.append(dict(model=key, branch=br, measure=col, a=a, b=b, g_rms=g_ref, b_over_g=b / g_ref,
                                sigma_c=sig, a_over_sigma_sqrt2=a / (sig / np.sqrt(2)) if np.isfinite(sig) else np.nan,
                                max_abs_resid=float(np.max(np.abs(pred - M[col].values))),
                                R3_rms_gt_nn=bool((M.dist_rms > M.dist_nn).all()) if col == "dist_rms" else np.nan))
    O = pd.DataFrame(out)
    O["R1"] = (O.b_over_g - 1).abs() <= 0.05
    O["R2"] = (O.a_over_sigma_sqrt2 - 1).abs() <= 0.15
    O.to_csv(RES / "E13_rms_fit.csv", index=False)
    print(O.round(3).to_string(index=False))


# ---------------- E13-5 ----------------
def glmT():
    rows = []
    for s in ("en", "zh"):
        import e11_jobs as J11
        Ts = EA.load_metrics(J11.wav_dir("g_base_n10", 0, s))  # speaker-null teacher (E11, reused)
        Ta, B, Rr = L("ga_T", 0, s), L("ga_11", 0, s), L("ga_01", 0, s)
        if Ta is None:
            continue
        m, lo, hi, p, n = paired(Ta, Ts)
        r = dict(set=s, sim_T_speakernull=Ts.sim_eres2net.mean(), sim_T_allnull=Ta.sim_eres2net.mean(), d=m, lo=lo, hi=hi, p=p,
                 G1=bool(abs(m) <= 0.5))
        for tag, T in (("a_speakernull_T", Ts), ("b_allnull_T", Ta)):
            gap = T.sim_eres2net.mean() - B.sim_eres2net.mean()
            r[f"tax_{tag}"] = gap
            r[f"rec_{tag}"] = (Rr.sim_eres2net.mean() - B.sim_eres2net.mean()) / gap
        r["wer_T_allnull"], r["wer_T_speakernull"] = Ta.wer.mean(), Ts.wer.mean()
        r["utmos_T_allnull"], r["utmos_T_speakernull"] = Ta.utmos.mean(), Ts.utmos.mean()
        rows.append(r)
    O = pd.DataFrame(rows)
    if len(O):
        O["G2"] = [abs(r.rec_b_allnull_T - ref) <= 0.10 for r, ref in zip(O.itertuples(), (0.95, 0.79))]
    O.to_csv(RES / "E13_glm_teacher.csv", index=False)
    print(O.round(3).T.to_string())


# ---------------- E13-8 ----------------
def f5():
    import e8p8_jobs as J8
    rows = []
    for n in (8, 16):
        for sway, dn, rn, wd in ((-1, f"e81_f5_n{n}_def", f"e81_f5_n{n}_R", J8.wav_dir), (0, f"f5s0_n{n}_def", f"f5s0_n{n}_R", J.wav_dir)):
            B, R0 = EA.load_metrics(wd(dn, 0, "en")), EA.load_metrics(wd(rn, 0, "en"))
            if B is None or R0 is None:
                continue
            dw = paired(R0, B, "wer")
            ds = paired(R0, B, "sim_eres2net")
            rows.append(dict(nfe=n, sway=sway, wer_base=B.wer.mean(), wer_R=R0.wer.mean(), d_wer=dw[0], d_wer_lo=dw[1], d_wer_hi=dw[2],
                             d_wer_p=dw[3], sim_base=B.sim_eres2net.mean(), sim_R=R0.sim_eres2net.mean(), d_sim=ds[0], d_sim_p=ds[3]))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "E13_f5_sway.csv", index=False)
    print(O.round(3).to_string(index=False))
    for n, g in O.groupby("nfe"):
        if len(g) == 2:
            g = g.set_index("sway")
            print(n, "S1 R WER cost sway0 >= sway-1:", bool(g.loc[0, "d_wer"] >= g.loc[-1, "d_wer"]),
                  "| S2 baseline better at sway -1 (WER, SIM):", bool(g.loc[-1, "wer_base"] <= g.loc[0, "wer_base"]),
                  bool(g.loc[-1, "sim_base"] >= g.loc[0, "sim_base"]))


# ---------------- E13-9.1 ----------------
def tost(delta=0.3):
    import e10_jobs as J10
    import e11d_jobs as J11D
    rows = []
    srcs = [(m, lambda n, s, m=m: J10.wav_dir(f"f_{m}_{n}", 0, s)) for m in ("cv2", "cv3", "idx")] + \
           [("glm", lambda n, s: J11D.wav_dir(f"d_{n}", 0, s))]
    for m, wd in srcs:
        for s in ("en", "zh", "hard", "lspc"):
            for x, ref in (("R_n2", "base_n2"), ("midR", "base_n2"), ("R_n3", "base_n3")):
                X, B = EA.load_metrics(wd(x, s)), EA.load_metrics(wd(ref, s))
                if X is None or B is None or "wer" not in X:
                    continue
                c = X.index.intersection(B.index)
                d = (X.loc[c, "wer"] - B.loc[c, "wer"]).dropna()
                est, lo, hi = boot(d, (5, 95))
                rows.append(dict(model=m, set=s, cond=x, ref=ref, n=len(d), d_wer=est, ci90_lo=lo, ci90_hi=hi,
                                 equivalent=bool(lo > -delta and hi < delta), noninferior=bool(hi < delta)))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "E13_tost.csv", index=False)
    print(O.round(3).to_string(index=False))
    cv = O[O.model.isin(["cv2", "cv3"])]
    print("T1 CosyVoice2/3 all cells equivalent:", bool(cv.equivalent.all()), f"({int(cv.equivalent.sum())}/{len(cv)}); non-inferior (upper 90% < delta): {int(cv.noninferior.sum())}/{len(cv)}")


# ---------------- E13-9.2 ----------------
def tg():
    """Total guidance sum_k h_k |w_k| gamma(t_{k-1}) for the GLM-TTS E11 configurations (E8'-9a definition), gamma from the
    E11 dumps (official branch); branch scaling by the t=0 guidance ratio g1(branch)/g1(official)."""
    E = pd.read_csv(RES / "E11_dumps.csv")
    gam = E[E.check == "gamma"].groupby("t").value.mean()
    g1 = E[E.check == "g1"].groupby("branch").value.mean()
    Lw = pd.read_csv(RES / "E11_law.csv")
    import e11_jobs as J11
    rows = []
    for _, r in Lw.iterrows():
        c = J11.CONFIGS.get(r.config, {})
        ex = c.get("extra", [])
        get = lambda f, d=None: ex[ex.index(f) + 1] if f in ex else d  # noqa: E731
        if any(x.startswith("--cfg_steps=") for x in ex):
            ws = [float(v) for v in [x for x in ex if x.startswith("--cfg_steps=")][0].split("=")[1].split(",")]
            n = len(ws)
        else:
            n = int(get("--nfe", 2))
            ws = [float(get("--cfg", 0.7))] * n
        ts = [float(v) for v in get("--t_span").split(",")] if get("--t_span") else list(1 - np.cos(np.pi / 2 * np.arange(n + 1) / n))
        iv = [float(v) for v in get("--cfg_interval").split(",")] if get("--cfg_interval") else None
        br = get("--cfg_branch", "all")
        scale = g1.get(br, g1["all"]) / g1["all"]
        tot = 0.0
        for i in range(len(ts) - 1):
            w = ws[i] if i < len(ws) else ws[-1]
            if iv and not (iv[0] <= ts[i] <= iv[1]):
                w = 0.0
            tot += (ts[i + 1] - ts[i]) * abs(w) * scale * float(np.interp(ts[i], gam.index.values, gam.values))
        rows.append(dict(config=r.config, kind=r.kind, branch=r.branch, E1=r.E1, E_tot=tot, loss=r.loss))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "E13_total_guidance_glm.csv", index=False)
    print(O.round(3).to_string(index=False))
    print("Spearman loss ~ E1:", round(spearmanr(O.E1, O.loss)[0], 3), "| loss ~ E_tot:", round(spearmanr(O.E_tot, O.loss)[0], 3), "n", len(O))


# ---------------- E13-9.3 ----------------
def ddc():
    G = pd.concat([pd.read_csv(RES / f"{f}.csv").assign(src=f) for f in ("E8p8_geometry", "E8p7_geometry", "E8p10_geometry")])
    G = G[G.measure == "ddcond"]
    O = G.pivot_table(index=["src", "model", "branch", "set"], columns="emb", values="R").reset_index()
    O.to_csv(RES / "E13_ddcond_3emb.csv", index=False)
    print(O.round(4).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    todo = sys.argv[1:] or ["grid", "ctrl", "mask10", "idxsweep", "rms", "glmT", "f5", "tost", "tg", "ddc"]
    fn = {"grid": grid, "ctrl": ctrl, "mask10": mask10, "idxsweep": idxsweep, "rms": rms_fit, "glmT": glmT, "f5": f5,
          "tost": tost, "tg": tg, "ddc": ddc}
    for t in todo:
        print(f"===== {t}")
        fn[t]()
