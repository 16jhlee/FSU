"""E5 figures.
Fig 3: one utterance (median ERes2Net SIM gap Euler vs teacher), CosyVoice2 NFE4: Euler / A1 / A1+A2-b(APG) /
       A1+A3-early, 8 seeds each, projected on the PCA of the teacher (NFE10) 8-seed cloud; prompt and centre shown.
Fig 4: CFG strength x NFE (CosyVoice2, seed 0): ERes2Net SIM-o and high-frequency ratio vs w for NFE 2 and 10,
       with the A2-c interval-CFG and A2-b APG points.
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
from e5_jobs import ROOT, wav_dir  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def fig3():
    meta = {l.split("|")[0]: l.split("|")[2] for l in
            (ROOT / "data/lists/en_all.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()}
    prom = dict(np.load(ROOT / "data/cache/prompt_eres2net.npz"))
    L = pd.read_csv(ROOT / "results/E5_seed0_long.csv")
    a = L[(L.config == "cv2_euler_n4") & (L.set == "en")].set_index("utt_id").sim_eres2net
    b = L[(L.config == "cv2_euler_n10") & (L.set == "en")].set_index("utt_id").sim_eres2net
    gap = (b - a).dropna()
    u = gap.index[(gap - gap.median()).abs().argsort().iloc[0]]

    def E(name):
        return unit(np.stack([np.load(wav_dir(name, k, "en") / "_metrics" / "emb_eres2net.npz")[f"gen/{u}"]
                              for k in range(8)]).astype(float))
    R = E("cv2_euler_n10")
    mu = R.mean(0)
    _, _, Vt = np.linalg.svd(R - mu, full_matrices=False)
    P = lambda X: (np.atleast_2d(X) - mu) @ Vt[:2].T  # noqa: E731
    p = unit(prom[os.path.normcase(os.path.abspath(meta[u])).replace("\\", "/")])
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    r = P(R)
    ax.scatter(r[:, 0], r[:, 1], s=60, facecolors="none", edgecolors=INK2, lw=1.4, label="teacher NFE10 (8 seeds)")
    for (name, lab), c in zip([("cv2_euler_n4", "Euler NFE4"), ("cv2_a1_n4", "A1 NFE4"),
                               ("cv2_a1apg_n4", "A1 + A2-b (APG) NFE4"), ("cv2_a1_early", "A1 + A3-early NFE4")], C):
        X = E(name)
        q = P(X)
        ax.scatter(q[:, 0], q[:, 1], s=40, color=c, edgecolors=SURF, lw=1.0,
                   label=f"{lab}  (SIM {float((X @ p).mean()):.3f})")
    ax.scatter(*P(unit(R.mean(0)))[0], marker="+", s=180, color=INK, lw=2, label="teacher conditional centre")
    ax.scatter(*P(p)[0], marker="*", s=200, color="#eda100", edgecolors=INK2, lw=0.6,
               label=f"prompt (teacher SIM {float((R @ p).mean()):.3f})")
    ax.set_xlabel("PC1 of teacher cloud")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_title(f"E5 Fig 3 — CosyVoice2, one utterance (median SIM gap), ERes2Net-en", loc="left", fontsize=10)
    style(ax)
    fig.savefig(ROOT / "reports/fig/E5_fig3_pca.png", dpi=150)
    plt.close(fig)


def fig4():
    S = pd.read_csv(ROOT / "results/E5_seed0_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    ws = ["0", "0.3", "0.7", "1.0", "1.5"]
    for ax, (m, ttl) in zip(axes, [("sim_eres2net", "SIM-o · ERes2Net-en"), ("hf_ratio", "4–8 kHz energy ratio")]):
        for n, c, mk in [(2, C[0], "o"), (10, C[1], "s")]:
            for s, ls in [("en", "-"), ("zh", "--")]:
                ys = []
                for w in ws:
                    name = f"cv2_euler_n{n}" if w == "0.7" else f"cv2_cfg{w}_n{n}"
                    r = S[(S.config == name) & (S.set == s)]
                    ys.append(r[m].iloc[0] if len(r) else np.nan)
                ax.plot([float(w) for w in ws], ys, color=c, ls=ls, marker=mk, ms=5, lw=2,
                        label=f"NFE {n} · {s}")
            for name, mk2, lab in [(f"cv2_iv_n{n}", "D", "interval CFG [0.2,0.8]"), (f"cv2_apg_n{n}", "^", "APG")]:
                r = S[(S.config == name) & (S.set == "en")]
                if len(r):
                    ax.scatter([0.7], [r[m].iloc[0]], marker=mk2, s=70, color=c, edgecolors=INK, lw=0.8,
                               zorder=4, label=f"NFE {n} · en · {lab} (w=0.7)")
        ax.set_xlabel("CFG rate w (CosyVoice2 inference_cfg_rate; official 0.7)")
        ax.set_title(ttl, loc="left", fontsize=10)
        style(ax)
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle("E5 Fig 4 — CFG × NFE (CosyVoice2, seed 0, n=300 per set)", x=0.01, ha="left", fontsize=11)
    fig.savefig(ROOT / "reports/fig/E5_fig4_cfg.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig3()
    fig4()
    print("saved")
