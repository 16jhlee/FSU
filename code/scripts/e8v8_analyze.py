"""v8 analysis. env_eval.  python scripts/e8v8_analyze.py [tstar|seeds|latency|all]
tstar  (§1.3): SIM (ERes2Net, en/zh) vs t* for each model x NFE (R(t*): w=0 for steps starting at t < t*), and the
         guidance size gamma(t) = RMS(v_c - v_u)(t) on a 20-step grid (en 100) -> Fig 13 (two stacked panels per model,
         no dual axis), results/E8v8_tstar.csv, E8v8_gamma_summary.csv. Pre-registered: SIM flat on t* in [0.05, 0.2]
         (spread <= eps), decreasing from t* >= 0.3 (NFE 10 first); gamma maximal at t=0 and monotone decreasing.
seeds  (§1.2): Table-1 main rows over seeds {0,1,2}: mean ± SD of SIM x3, WER, UTMOS; flags cells whose conclusion
         (sign of R - baseline, mid+R - teacher within ±1, R recovery >= 70%) differs between seeds.
latency(§1.4): Table 10 from results/E8v8_latency.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e8v8_jobs as J  # noqa: E402
import e8p_analyze as A  # noqa: E402

ROOT = A.ROOT
A.wdir = J.wav_dir
MODEL = {"cv2": "CosyVoice2", "cv3": "CosyVoice3", "idx": "IndexTTS-2"}
TEACH = {"cv2": "e82_cv2_all_n10_def", "cv3": "e81_cv3_n10_def", "idx": "e81_idx_n25_def"}


def mean_k(cfg, k, s, m):
    f = J.wav_dir(cfg, k, s) / "_metrics" / "metrics.csv"
    if not f.exists():
        return np.nan
    D = pd.read_csv(f)
    if m not in D:
        return np.nan
    return D[m].mean() * (100 if m.startswith("sim") or m == "wer" else 1)  # SIM x100, WER in %


def tstar():
    rows = []
    for (model, n, ts), cfg in J.TSTAR_MAP.items():
        for s in ("en", "zh"):
            rows.append(dict(model=MODEL[model], nfe=n, tstar=ts, set=s, config=cfg, mask=J.mask(model, n, ts),
                             sim=mean_k(cfg, 0, s, "sim_eres2net"), wer=mean_k(cfg, 0, s, "wer"), utmos=mean_k(cfg, 0, s, "utmos"),
                             teacher=mean_k(TEACH[model], 0, s, "sim_eres2net")))
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "results/E8v8_tstar.csv", index=False)
    G = []
    for key, model in MODEL.items():
        f = ROOT / f"results/E8v8_gamma_{key}.csv"
        if f.exists():
            D = pd.read_csv(f)
            S = D.groupby("k").agg(t=("t", "first"), gamma=("rms", "mean"), sd=("rms", "std")).reset_index()
            S["model"], S["gamma_rel"] = model, S.gamma / S.gamma.iloc[0]
            G.append(S)
    G = pd.concat(G) if G else pd.DataFrame()
    G.to_csv(ROOT / "results/E8v8_gamma_summary.csv", index=False)
    checks = []
    for (model, n, s), g in T.groupby(["model", "nfe", "set"]):
        g = g.sort_values("tstar")
        plateau = g[(g.tstar >= 0.05) & (g.tstar <= 0.2)].sim
        hi = g[g.tstar >= 0.3].sim
        checks.append(dict(model=model, nfe=n, set=s, plateau_spread=plateau.max() - plateau.min(),
                           plateau_mean=plateau.mean(), sim_t03=g[g.tstar == 0.3].sim.iloc[0], sim_t05=g[g.tstar == 0.5].sim.iloc[0],
                           drop_from_plateau_t05=plateau.mean() - g[g.tstar == 0.5].sim.iloc[0]))
    C = pd.DataFrame(checks)
    fig13(T, G)
    return T, G, C


def fig13(T, G):
    models = list(MODEL.values())
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True, facecolor=A.SURF,
                             gridspec_kw=dict(height_ratios=[1.6, 1]), sharex="col")
    cols = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a", 10: "#6250d6"}
    for j, model in enumerate(models):
        ax = axes[0, j]
        g = T[(T.model == model) & (T.set == "en")]
        for n in J.NFES:
            q = g[g.nfe == n].sort_values("tstar")
            ax.plot(q.tstar, q.sim, "-o", color=cols[n], lw=2, ms=5, label=f"NFE {n}")
        ax.axhline(g.teacher.iloc[0], color="#52514e", lw=1, ls="--")
        ax.axvspan(0.05, 0.2, color=A.GRID, alpha=0.6, lw=0)
        ax.set_title(f"{model} — SIM vs t* (en)", loc="left", fontsize=10, color=A.INK)
        ax.set_ylabel("SIM-o ×100 (ERes2Net)" if j == 0 else "", fontsize=9, color=A.INK2)
        A.style(ax)
        ax2 = axes[1, j]
        if len(G) and model in set(G.model):
            q = G[G.model == model]
            ax2.plot(q.t, q.gamma_rel, "-o", color="#52514e", lw=2, ms=4)
        ax2.axvspan(0.05, 0.2, color=A.GRID, alpha=0.6, lw=0)
        ax2.set_xlabel("t* (start-time threshold) / t (repo time, 0 = noise)", fontsize=8.5, color=A.INK2)
        ax2.set_ylabel("γ(t) / γ(0)" if j == 0 else "", fontsize=9, color=A.INK2)
        ax2.set_title("guidance size γ(t) = RMS(v_c − v_u)", loc="left", fontsize=9.5, color=A.INK)
        A.style(ax2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 13 — R(t*) sensitivity (top; dashed = teacher, band = pre-registered plateau [0.05, 0.2]) and γ(t) decay "
                 "(bottom, Lemma 2)", x=0.01, ha="left", fontsize=11, color=A.INK)
    fig.savefig(ROOT / "reports/fig/E8v8_fig13_tstar.png", dpi=150, facecolor=A.SURF)
    plt.close(fig)


ROWS = {"CosyVoice2": [("NFE 2", "baseline", "e82_cv2_all_n2_def"), ("NFE 2", "R (ours)", "e82_cv2_all_n2_R"),
                       ("NFE 2", "mid+R (ours)", "A_cv2_midR"), ("NFE 3", "baseline", "e82_cv2_all_n3_def"),
                       ("NFE 3", "R (ours)", "e82_cv2_all_n3_R"), ("NFE 3", "mid+R +1 Euler (ours)", "s_cv2_midR1"),
                       ("teacher", "teacher", "e82_cv2_all_n10_def")],
        "CosyVoice3": [("NFE 2", "baseline", "e81_cv3_n2_def"), ("NFE 2", "R (ours)", "e81_cv3_n2_R"),
                       ("NFE 2", "mid+R (ours)", "A_cv3_midR"), ("NFE 3", "baseline", "s_cv3_n3_def"),
                       ("NFE 3", "R (ours)", "s_cv3_n3_R"), ("NFE 3", "mid+R +1 Euler (ours)", "s_cv3_midR1"),
                       ("teacher", "teacher", "e81_cv3_n10_def")],
        "IndexTTS-2": [("NFE 2", "baseline", "e81_idx_n2_def"), ("NFE 2", "R (ours)", "e81_idx_n2_R"),
                       ("NFE 2", "mid+R (ours)", "A_idx_midR"), ("NFE 3", "baseline", "s_idx_n3_def"),
                       ("NFE 3", "R (ours)", "s_idx_n3_R"), ("teacher", "teacher", "e81_idx_n25_def")]}
METRICS = ["sim_eres2net", "sim_wavlm", "sim_ecapa", "wer", "utmos"]


def seeds():
    rows = []
    for model, items in ROWS.items():
        for nfe, meth, cfg in items:
            for s in ("en", "zh"):
                for k in (0, 1, 2):
                    r = dict(model=model, nfe=nfe, method=meth, set=s, seed=k, config=cfg)
                    for m in METRICS:
                        r[m] = mean_k(cfg, k, s, m)
                    rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(ROOT / "results/E8v8_seeds_long.csv", index=False)
    agg = S.groupby(["model", "nfe", "method", "set"], sort=False)[METRICS].agg(["mean", "std"])
    # conclusion flags per seed
    flags = []
    for model in ROWS:
        for s in ("en", "zh"):
            g = S[(S.model == model) & (S.set == s)]
            for k in (0, 1, 2):
                q = g[g.seed == k].set_index(["nfe", "method"]).sim_eres2net
                T = q[("teacher", "teacher")]
                for nfe in ("NFE 2", "NFE 3"):
                    b, r = q.get((nfe, "baseline")), q.get((nfe, "R (ours)"))
                    if b is None or r is None:
                        continue
                    flags.append(dict(model=model, set=s, seed=k, nfe=nfe, R_gt_base=r > b,
                                      R_recovery=(r - b) / (T - b) if T - b > 0.42 else np.nan,
                                      midR_minus_teacher=q.get((nfe, "mid+R (ours)"), np.nan) - T))
    F = pd.DataFrame(flags)
    F.to_csv(ROOT / "results/E8v8_seed_flags.csv", index=False)
    return S, agg, F


XPAIRS = [("en text: zh prompt vs en prompt", "xl_zh2en", "xl_en2en"), ("zh text: en prompt vs zh prompt", "xl_en2zh", "xl_zh2zh"),
          ("ko text: en prompt vs ko prompt", "xl_en2ko", "xl_ko2ko"), ("en text: ko prompt vs en prompt", "xl_ko2en", "xl_en2en")]


def xling():
    import e8v8x_jobs as X
    import scipy.stats as st

    def load(cfg, s):
        f = X.wav_dir(cfg, 0, s) / "_metrics" / "metrics.csv"
        D = pd.read_csv(f).set_index("utt_id")
        D["sim"], D["wer"] = D.sim_eres2net * 100, D.wer * 100
        return D
    rows = []
    for model in ["cv2", "cv3"]:
        for s, _ in X.LISTS:
            T = load(f"x_{model}_n10_def", s)
            for n in (2, 3):
                B, R = load(f"x_{model}_n{n}_def", s), load(f"x_{model}_n{n}_R", s)
                gain = R.sim - B.sim
                rows.append(dict(model=MODEL[model], list=s, nfe=n, teacher=T.sim.mean(), base=B.sim.mean(), R=R.sim.mean(),
                                 loss=T.sim.mean() - B.sim.mean(), gain=gain.mean(), gain_p=st.wilcoxon(gain).pvalue,
                                 rec=gain.mean() / (T.sim.mean() - B.sim.mean()), wer_T=T.wer.mean(), wer_B=B.wer.mean(),
                                 wer_R=R.wer.mean(), dwer_p=st.wilcoxon(R.wer - B.wer).pvalue if (R.wer - B.wer).any() else 1.0,
                                 utmos_B=B.utmos.mean(), utmos_R=R.utmos.mean()))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E8v8_xling.csv", index=False)
    G = []
    for key in ["cv2", "cv3"]:
        f = ROOT / f"results/E8v8_xl_guidance_{key}.csv"
        if f.exists():
            G.append(pd.read_csv(f).groupby("list").rms.mean().rename(MODEL[key]))
    G = pd.concat(G, axis=1) if G else None
    comp = []
    for model in ["CosyVoice2", "CosyVoice3"]:
        for n in (2, 3):
            for lab, cross, same in XPAIRS:
                c, s_ = D[(D.model == model) & (D.nfe == n) & (D.list == cross)].iloc[0], D[(D.model == model) & (D.nfe == n) & (D.list == same)].iloc[0]
                comp.append(dict(model=model, nfe=n, pair=lab, loss_cross=c.loss, loss_same=s_.loss, gain_cross=c.gain,
                                 gain_same=s_.gain, rec_cross=c.rec, rec_same=s_.rec, rec_diff_pp=(c.rec - s_.rec) * 100,
                                 g_cross=G.loc[cross, model] if G is not None else np.nan,
                                 g_same=G.loc[same, model] if G is not None else np.nan))
    return D, pd.DataFrame(comp)


def stream():
    import e8v8s_jobs as Sj
    import scipy.stats as st
    rows = []
    for s, _, _ in Sj.SETS:
        for mode in ("off", "str"):
            T = pd.read_csv(Sj.wav_dir(f"s_{mode}_n10_def", 0, s) / "_metrics/metrics.csv").set_index("utt_id")
            for n in (2, 3):
                B = pd.read_csv(Sj.wav_dir(f"s_{mode}_n{n}_def", 0, s) / "_metrics/metrics.csv").set_index("utt_id")
                R = pd.read_csv(Sj.wav_dir(f"s_{mode}_n{n}_R", 0, s) / "_metrics/metrics.csv").set_index("utt_id")
                gain = (R.sim_eres2net - B.sim_eres2net) * 100
                rows.append(dict(set=s, mode=mode, nfe=n, dur=B.dur_gen_s.mean(), teacher=T.sim_eres2net.mean() * 100,
                                 base=B.sim_eres2net.mean() * 100, R=R.sim_eres2net.mean() * 100, gain=gain.mean(),
                                 gain_p=st.wilcoxon(gain).pvalue, rec=gain.mean() / ((T.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100),
                                 wer_T=T.wer.mean() * 100, wer_B=B.wer.mean() * 100, wer_R=R.wer.mean() * 100,
                                 utmos_T=T.utmos.mean(), utmos_B=B.utmos.mean(), utmos_R=R.utmos.mean()))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E8v8_stream.csv", index=False)
    return D


def robust():
    """§2.3: per prompt condition (length 1/3/10 s, SNR 20/10 dB at 3 s), CosyVoice2 NFE 2 baseline vs R vs teacher;
    SIM vs the clean 10 s crop; 8-seed geometry vs the teacher of the SAME condition (ERes2Net)."""
    import os
    import e8v8r_jobs as Rj
    import scipy.stats as st
    prom = dict(np.load(ROOT / "data/cache/prompt_eres2net.npz"))
    rows = []
    for s, m, _ in Rj.SETS:
        refs = {l.split("|")[0]: l.split("|")[2] for l in (ROOT / f"{m}_ref.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()}
        L = {c: pd.read_csv(Rj.wav_dir(c, 0, s) / "_metrics/metrics.csv").set_index("utt_id") for c in ("r_n2_def", "r_n2_R", "r_n10_def")}
        gain = (L["r_n2_R"].sim_eres2net - L["r_n2_def"].sim_eres2net) * 100
        r = dict(cond=s, teacher=L["r_n10_def"].sim_eres2net.mean() * 100, base=L["r_n2_def"].sim_eres2net.mean() * 100,
                 R=L["r_n2_R"].sim_eres2net.mean() * 100, gain=gain.mean(), gain_p=st.wilcoxon(gain).pvalue,
                 wer_B=L["r_n2_def"].wer.mean() * 100, wer_R=L["r_n2_R"].wer.mean() * 100,
                 utmos_B=L["r_n2_def"].utmos.mean(), utmos_R=L["r_n2_R"].utmos.mean(), utmos_T=L["r_n10_def"].utmos.mean())
        r["rec"] = r["gain"] / (r["teacher"] - r["base"])

        def E(c):
            arr = [np.load(Rj.wav_dir(c, k, s) / "_metrics/emb_eres2net.npz") for k in range(8)]
            return {u: A.unit(np.stack([a_[f"gen/{u}"] for a_ in arr]).astype(float)) for u in refs}
        T, B, Rr = E("r_n10_def"), E("r_n2_def"), E("r_n2_R")
        for tag, X in [("base", B), ("R", Rr)]:
            vr, dc = [], []
            for u in refs:
                UT = T[u]
                c = A.unit(UT.mean(0))
                loo = np.mean([A.cosd(UT[k], A.unit(np.delete(UT, k, 0).mean(0))) for k in range(8)])
                vr.append(np.log(A.cvar(X[u]) / A.cvar(UT)))
                dc.append(np.mean([A.cosd(x, c) for x in X[u]]) - loo)
            r[f"var_ratio_{tag}"], r[f"ddcond_{tag}"] = float(np.exp(np.mean(vr))), float(np.mean(dc))
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E8v8_robust.csv", index=False)
    return D


def latency():
    f = ROOT / "results/E8v8_latency.csv"
    if not f.exists():
        return None
    D = pd.read_csv(f)
    return D.groupby(["cond", "calls"], sort=False)[["first_packet_ms", "first_flow_ms", "first_hift_ms", "llm_first_chunk_ms",
                                                     "decoder_rtf", "sessions", "decoder_share_first"]].agg(["mean", "std"])


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    if what in ("all", "tstar"):
        T, G, C = tstar()
        print(T.pivot_table(index=["model", "nfe", "set"], columns="tstar", values="sim").round(2).to_string())
        print(C.round(3).to_string(index=False))
        if len(G):
            print(G[["model", "t", "gamma", "gamma_rel"]].round(3).to_string(index=False))
    if what in ("all", "seeds"):
        S, agg, F = seeds()
        print(agg.round(2).to_string())
        print(F.round(3).to_string(index=False))
    if what in ("all", "xling"):
        D, C = xling()
        print(D.round(3).to_string(index=False))
        print(C.round(3).to_string(index=False))
    if what in ("all", "robust"):
        print(robust().round(3).to_string(index=False))
    if what in ("all", "stream"):
        print(stream().round(3).to_string(index=False))
    if what in ("all", "latency"):
        L = latency()
        if L is not None:
            print(L.round(3).to_string())
