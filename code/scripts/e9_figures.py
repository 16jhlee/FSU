"""E9 (v7 §3.3) new paper figures; the others are reused (list in reports/E9_figures.md). env_eval.
Fig 1  mechanism: (a) 1-step closed form on the extrapolation axis with measured distances (E8'-9e), (b) support test of
       the first x-prediction with/without CFG (E3b), (c) distance to the teacher mean and seed dependence vs w (E8'-9e).
Fig 3  E3 PCA for one utterance: teacher 8 seeds vs baseline NFE 2 vs R (ours) NFE 2 (ERes2Net embeddings).
Fig 5  hybrid k: teacher's first k cosine steps then one jump (E5 A3), ΔSIM vs teacher.
Fig 9  branch experiment (CosyVoice2, NFE 2 SIM) + first-step guidance size per branch for 3 models.
Outputs reports/fig/E9_fig{1,3,5,9}_*.png
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e8p9_jobs as J9  # noqa: E402
import e8p_analyze as A  # noqa: E402

ROOT = A.ROOT
FIG = ROOT / "reports" / "fig"
INK, INK2, GRID, SURF = A.INK, A.INK2, A.GRID, A.SURF
C_DEF, C_R, C_T = A.C_DEF, A.C_R, "#52514e"


def fig1():
    E = pd.read_csv(ROOT / "results/E8p9e_onestep.csv")
    a = E[E.k >= 0].mean(numeric_only=True)
    sd = E[E.k < 0].groupby("w")[["d_seed", "d_seed_T"]].mean()
    M = pd.read_csv(ROOT / "results/E3b_mechanism_summary.csv").set_index("what")
    M0 = pd.read_csv(ROOT / "results/E3b_mechanism_cfg0_summary.csv").set_index("what")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True, facecolor=SURF,
                             gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    # (a) extrapolation axis (schematic positions, measured distances to the teacher mean)
    ax = axes[0]
    spread = a.d_TT
    ax.axvspan(-spread, spread, color=GRID, alpha=0.9, lw=0)
    ax.text(0, 0.62, f"teacher cloud\n(LOO spread {spread:.2f})", ha="center", fontsize=8.5, color=INK2)
    pts = [(-a.d_T_xu, "$\\bar x_\\emptyset$ (tokens erased)", INK2), (a.d_T_w0, "$\\hat x_1(w{=}0)\\approx\\bar x_c$", C_DEF),
           (a.d_T_w07, "$\\hat x_1(0.7)$ official", C_R), (a.d_T_w15, "$\\hat x_1(1.5)$", "#e34948")]
    for x, lab, col in pts:
        ax.scatter([x], [0.3], s=90, color=col, edgecolor=SURF, lw=2, zorder=3)
        ax.annotate(lab, (x, 0.3), xytext=(0, -26), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
        ax.text(x, 0.40, f"{abs(x):.2f}", ha="center", fontsize=8, color=INK2)
    ax.annotate("", xy=(a.d_T_w15, 0.18), xytext=(a.d_T_w0, 0.18),
                arrowprops=dict(arrowstyle="->", color=C_R, lw=1.5))
    ax.text((a.d_T_w0 + a.d_T_w15) / 2, 0.11, "$w(\\bar x_c-\\bar x_\\emptyset)$", ha="center", fontsize=9, color=C_R)
    ax.axhline(0.3, color=GRID, lw=1, zorder=1)
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-a.d_T_xu - 0.7, a.d_T_w15 + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("mel RMS distance from the teacher's 8-seed mean (signed along the extrapolation axis; schematic)",
                  fontsize=8.5, color=INK2)
    ax.set_title("(a) one Euler step from noise: $\\hat x_1=\\bar x_c+w(\\bar x_c-\\bar x_\\emptyset)$", loc="left",
                 fontsize=10, color=INK)
    A.style(ax)
    ax.grid(False)
    # (b) support test (E3b): nearest-neighbour distance minus the reference 95% quantile (>0 = off support)
    ax = axes[1]
    labs = ["first x-pred\n$\\hat x_0$", "NFE 2\noutput", "on-path pred.\n(ODE state)"]
    keys = ["nn(x0hat) - q95(nn_ref)", "nn(nfe2) - q95(nn_ref)", "nn(xhat_ode) - q95(nn_ref)"]
    v07 = [M.loc[k, "mean"] for k in keys]
    v0 = [M0.loc[k, "mean"] for k in keys]
    x = np.arange(3)
    ax.bar(x - 0.19, v07, 0.36, color=C_R, label="w = 0.7 (official)", edgecolor=SURF, lw=2)
    ax.bar(x + 0.19, v0, 0.36, color=C_DEF, label="w = 0", edgecolor=SURF, lw=2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_ylabel("NN distance − ref. 95% quantile (mel)", fontsize=8.5, color=INK2)
    ax.set_title("(b) off-support test (E3b, CosyVoice2 NFE 2)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    A.style(ax)
    # (c) distance and seed dependence vs w (E8'-9e)
    ax = axes[2]
    ws = [0.0, 0.7, 1.5]
    ax.plot(ws, [a.d_T_w0, a.d_T_w07, a.d_T_w15], "-o", color=C_R, lw=2, ms=7, label="distance to teacher mean")
    ax.plot(ws, [sd.loc[w, "d_seed"] for w in ws], "-s", color=C_DEF, lw=2, ms=7, label="seed-to-seed distance (1-step)")
    ax.axhline(spread, color=C_T, lw=1, ls="--")
    ax.text(0.35, spread, "teacher LOO spread", fontsize=8, color=INK2, ha="center", va="top")
    ax.axhline(sd.loc[0.0, "d_seed_T"], color=C_T, lw=1, ls=":")
    ax.text(0.35, sd.loc[0.0, "d_seed_T"], "teacher seed-to-seed", fontsize=8, color=INK2, ha="center", va="bottom")
    ax.set_xticks(ws)
    ax.set_xlabel("guidance weight w (NFE 1)", fontsize=8.5, color=INK2)
    ax.set_ylabel("mel RMS distance", fontsize=8.5, color=INK2)
    ax.set_title("(c) 1-step output vs w (en 100 × 8 seeds)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    A.style(ax)
    fig.suptitle("Fig 1 — mechanism: the first CFG step extrapolates away from the conditional mean", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(FIG / "E9_fig1_mechanism.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def fig3():
    s = "en"
    meta = {l.split("|")[0]: l.split("|")[2] for l in (ROOT / "data/lists/en_all.meta.lst").read_text(encoding="utf-8").splitlines()
            if l.strip()}
    prom = dict(np.load(ROOT / "data/cache/prompt_eres2net.npz"))

    def E(cfg):
        arr = [np.load(J9.wav_dir(cfg, k, s) / "_metrics" / "emb_eres2net.npz") for k in range(8)]
        return {u: unit(np.stack([a_[f"gen/{u}"] for a_ in arr]).astype(float)) for u in meta}
    T, B, R = E("e82_cv2_all_n10_def"), E("e82_cv2_all_n2_def"), E("e82_cv2_all_n2_R")
    dd = {u: 1 - unit(B[u].mean(0)) @ unit(T[u].mean(0)) for u in meta}
    order = sorted(dd, key=dd.get)
    picks = [order[len(order) // 2], order[int(len(order) * 0.9)]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, facecolor=SURF)
    for ax, u in zip(axes, picks):
        X = np.concatenate([T[u], B[u], R[u]])
        mu = T[u].mean(0)
        _, _, Vt = np.linalg.svd(T[u] - mu, full_matrices=False)
        P = (X - mu) @ Vt[:2].T
        ep = (unit(prom[os.path.normcase(os.path.abspath(meta[u])).replace("\\", "/")]) - mu) @ Vt[:2].T
        ax.scatter(P[:8, 0], P[:8, 1], s=60, facecolor="none", edgecolor=C_T, lw=1.5, label="teacher NFE 10 (8 seeds)")
        ax.scatter(P[8:16, 0], P[8:16, 1], s=60, color=C_DEF, edgecolor=SURF, lw=1.5, label="baseline NFE 2")
        ax.scatter(P[16:, 0], P[16:, 1], s=60, color=C_R, edgecolor=SURF, lw=1.5, label="R (ours) NFE 2")
        for sl, col in [(slice(8, 16), C_DEF), (slice(16, 24), C_R)]:
            c = P[sl].mean(0)
            ax.scatter([c[0]], [c[1]], marker="+", s=140, color=col, lw=2)
        ax.scatter([0], [0], marker="+", s=140, color=C_T, lw=2)
        ax.scatter([ep[0]], [ep[1]], marker="*", s=160, color="#eda100", edgecolor=SURF, lw=1, label="prompt")
        ax.set_title(f"{'median' if u == picks[0] else '90th pct'} baseline bias utterance", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("PC1 of the teacher cloud", fontsize=8.5, color=INK2)
        ax.set_ylabel("PC2", fontsize=8.5, color=INK2)
        A.style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 3 — speaker-embedding cloud (ERes2Net, CosyVoice2 en; + = centre): baseline NFE 2 is shifted, "
                 "R (ours) re-centres", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.savefig(FIG / "E9_fig3_pca.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig5():
    S = pd.read_csv(ROOT / "results/E5_seed0_summary.csv")
    t10 = 1 - np.cos(np.pi / 2 * np.arange(11) / 10)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True, facecolor=SURF)
    for s, col in [("en", C_DEF), ("zh", C_R)]:
        ks = [1, 3, 5, 8]
        v = [S[(S.config == f"cv2_hybrid_k{k}") & (S.set == s)].vsTeacher_sim_eres2net.iloc[0] * 100 for k in ks]
        ax.plot(ks, v, "-o", color=col, lw=2, ms=7, label=s)
        for k, y in zip(ks, v):
            ax.annotate(f"{y:+.2f}", (k, y), xytext=(6, -14) if s == "en" else (-38, 6), textcoords="offset points",
                        fontsize=8, color=col)
    ax.axhspan(-0.42, 0.42, color=GRID, alpha=0.7, lw=0)
    ax.set_xticks([1, 3, 5, 8])
    ax.set_xticklabels([f"k={k}\nt={t10[k]:.2f}" for k in [1, 3, 5, 8]])
    ax.set_xlabel("teacher's first k cosine steps (of 10), then one jump to t=1", fontsize=8.5, color=INK2)
    ax.set_ylabel("ΔSIM vs teacher ×100 (ERes2Net)", fontsize=8.5, color=INK2)
    ax.legend(frameon=False, fontsize=8)
    A.style(ax)
    ax.set_title("Fig 5 — hybrid: speaker identity is settled by t ≈ 0.3 (CosyVoice2)", loc="left", fontsize=10.5, color=INK)
    fig.savefig(FIG / "E9_fig5_hybrid.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig9():
    P = pd.read_csv(ROOT / "results/E8p_pairs.csv")
    Q = P[(P.group == "G3") & (P.set == "en") & (P.nfe == 2)].set_index("branch")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, facecolor=SURF)
    ax = axes[0]
    brs = ["all", "speaker", "token"]
    x = np.arange(3)
    for i, b in enumerate(brs):
        ax.plot([i, i], [Q.loc[b, "def_sim_eres2net"], Q.loc[b, "R_sim_eres2net"]], color=GRID, lw=2, zorder=1)
    ax.scatter(x, [Q.loc[b, "def_sim_eres2net"] for b in brs], s=70, color=C_DEF, edgecolor=SURF, lw=2, zorder=3,
               label="no step rule")
    ax.scatter(x, [Q.loc[b, "R_sim_eres2net"] for b in brs], s=70, color=C_R, edgecolor=SURF, lw=2, zorder=3,
               label="+ R step rule")
    T = Q.loc["all", "T_sim_eres2net"]
    ax.axhline(T, color=C_T, lw=1, ls="--")
    ax.text(2.4, T, "teacher NFE 10", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(["all-null\n(official)", "speaker-null\n(R′ Sel-CFG)", "token-null\n(analysis)"], fontsize=8.5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("SIM-o ×100 (ERes2Net, en)", fontsize=8.5, color=INK2)
    ax.set_title("(a) CosyVoice2 NFE 2: what the unconditional branch erases", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    A.style(ax)
    ax = axes[1]
    files = [("CosyVoice2", "E8p_guidance.csv"), ("CosyVoice3", "E8p8_guidance_cv3.csv"), ("IndexTTS-2", "E8p8_guidance_idx.csv")]
    cols = {"all": C_T, "token": "#6250d6", "speaker": C_R}
    for j, (m, f) in enumerate(files):
        D = pd.read_csv(ROOT / "results" / f).groupby("branch").rms.mean()
        for i, b in enumerate(brs):
            ax.bar(j + (i - 1) * 0.26, D[b], 0.24, color=cols[b], edgecolor=SURF, lw=2,
                   label={"all": "all-null", "token": "token-null", "speaker": "speaker-null"}[b] if j == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels([m for m, _ in files])
    ax.set_ylabel("RMS of $g_1 = w(v_c - v_u)$ at t = 0", fontsize=8.5, color=INK2)
    ax.set_title("(b) first-step guidance size by branch (en 100)", loc="left", fontsize=10, color=INK)
    ax.set_ylim(0, 1.8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    A.style(ax)
    fig.suptitle("Fig 9 — the few-step loss appears when the unconditional branch erases the tokens", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(FIG / "E9_fig9_branch_g1.png", dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig3()
    fig5()
    fig9()
    print("ok")
