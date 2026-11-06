"""E3 figures.
Fig 1: one utterance per panel (A CosyVoice2 NFE2, B ZipVoice-Distill NFE2, C DMOSpeech2 student NFE2),
       ERes2Net-en unit embeddings projected on the 2-D PCA of the reference (teacher) 8-seed cloud;
       conditional centre, global centre and prompt shown. Utterance = the one with median dd_cond.
Fig 2: forest plot, test-en, 5 measures x 7 comparisons, ERes2Net-en (filled) with WavLM / ECAPA (hollow).
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"cv2": "#2a78d6", "zip_": "#eb6834", "zipd": "#1baf7a", "dmoS": "#e87ba4"}
LABEL = {"cv2_nfe2 vs cv2_nfe10": "A CosyVoice2 NFE2 vs 10", "cv2_nfe3 vs cv2_nfe10": "A CosyVoice2 NFE3 vs 10",
         "zip_nfe4 vs zip_nfe16": "B ZipVoice teacher NFE4 vs 16",
         "zipd_nfe2 vs zip_nfe16": "B ZipVoice-Distill NFE2 vs teacher 16",
         "zipd_nfe4 vs zip_nfe16": "B ZipVoice-Distill NFE4 vs teacher 16",
         "dmoS_nfe2 vs dmoT_nfe32": "C DMOSpeech2 student NFE2 vs teacher 32",
         "dmoS_nfe4 vs dmoT_nfe32": "C DMOSpeech2 student NFE4 vs teacher 32"}


def col(comp):
    for k, v in COL.items():
        if comp.startswith(k):
            return v
    return INK2


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def fig1():
    M = pd.read_csv(ROOT / "results/E3_measures.csv")
    E = np.load(ROOT / "results/E3_emb_en.npz")
    meta = {l.split("|")[0]: l.split("|")[2] for l in
            (ROOT / "data/lists/en_all.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()}
    prom = dict(np.load(ROOT / "data/cache/prompt_eres2net.npz"))
    panels = [("cv2_nfe2", "cv2_nfe10", "A  CosyVoice2: NFE2 vs NFE10"),
              ("zipd_nfe2", "zip_nfe16", "B  ZipVoice-Distill NFE2 vs teacher NFE16"),
              ("dmoS_nfe2", "dmoT_nfe32", "C  DMOSpeech2 student NFE2 vs teacher NFE32")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (few, ref, title) in zip(axes, panels):
        g = M[(M.few == few) & (M.ref == ref) & (M.set == "en") & (M.emb == "eres2net")]
        u = g.iloc[(g.dd_cond - g.dd_cond.median()).abs().argsort().iloc[0]].utt
        R = unit(E[f"eres2net/{ref}/en/{u}"].astype(float))
        F = unit(E[f"eres2net/{few}/en/{u}"].astype(float))
        allR = unit(np.stack([E[k] for k in E.files if k.startswith(f"eres2net/{ref}/en/")]).reshape(-1, R.shape[1]))
        gc = unit(allR.mean(0))
        c = unit(R.mean(0))
        p = unit(prom[os.path.normcase(os.path.abspath(meta[u])).replace("\\", "/")])
        mu = R.mean(0)
        _, _, Vt = np.linalg.svd(R - mu, full_matrices=False)
        P = lambda X: (np.atleast_2d(X) - mu) @ Vt[:2].T  # noqa: E731
        a, b, cc, gg, pp = P(R), P(F), P(c), P(gc), P(p)
        ax.scatter(a[:, 0], a[:, 1], s=46, facecolors="none", edgecolors=INK2, lw=1.4, label="reference, 8 seeds")
        ax.scatter(b[:, 0], b[:, 1], s=46, color=col(few), edgecolors=SURF, lw=1.2, label="few-step, 8 seeds")
        ax.scatter(*cc[0], marker="+", s=160, color=INK, lw=2, label="conditional centre")
        ax.scatter(*pp[0], marker="*", s=180, color="#eda100", edgecolors=INK2, lw=0.6, label="prompt")
        ax.scatter(*gg[0], marker="x", s=90, color=INK2, lw=1.6, label="global centre")
        sims_r, sims_f = (R @ p).mean(), (F @ p).mean()
        ax.set_title(f"{title}\nSIM-o to prompt: ref {sims_r:.3f} → few {sims_f:.3f}", loc="left", fontsize=9.5)
        ax.set_xlabel("PC1 of reference cloud")
        ax.set_ylabel("PC2")
        style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("E3 Fig 1 — ERes2Net-en speaker embeddings, one utterance (median Δd_cond), PCA fitted on the "
                 "reference 8-seed cloud", x=0.01, ha="left", fontsize=11)
    fig.savefig(ROOT / "reports/fig/E3_fig1_pca.png", dpi=150)
    plt.close(fig)


def fig2():
    S = pd.read_csv(ROOT / "results/E3_summary.csv")
    S = S[S.set == "en"]
    meas = [("var_ratio", "seed-variance ratio (few/ref)", 1.0, True),
            ("Δd_cond", "Δ dist. to conditional centre", 0.0, False),
            ("Δd_glob", "Δ dist. to global centre", 0.0, False),
            ("norm_ratio", "embedding norm ratio", 1.0, True),
            ("Δd_prompt (few)", "few dist. to prompt − centre's", 0.0, False)]
    comps = list(LABEL)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6), sharey=True, constrained_layout=True)
    for ax, (m, title, ref, logx) in zip(axes, meas):
        for i, cmp in enumerate(comps):
            y = len(comps) - 1 - i
            for e, off, filled in [("wavlm", 0.22, False), ("ecapa", -0.22, False), ("eres2net", 0.0, True)]:
                r = S[(S.comp == cmp) & (S.emb == e) & (S.measure == m)]
                if not len(r):
                    continue
                r = r.iloc[0]
                c = col(cmp)
                ax.plot([r.lo, r.hi], [y + off] * 2, color=c, lw=2 if filled else 1.2, alpha=1 if filled else 0.7)
                ax.scatter(r.est, y + off, s=40 if filled else 26, color=c if filled else SURF, edgecolors=c,
                           lw=1.3, zorder=3)
        ax.axvline(ref, color=INK2, lw=1, ls="--")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_title(title, loc="left", fontsize=9.5)
        style(ax)
    axes[0].set_yticks(range(len(comps)))
    axes[0].set_yticklabels([LABEL[c] for c in comps][::-1], fontsize=8.5)
    fig.suptitle("E3 Fig 2 — paired effects, test-en (n=300, 8 seeds; 95% bootstrap CI). Filled = ERes2Net-en, "
                 "hollow = WavLM (upper) / ECAPA (lower)", x=0.01, ha="left", fontsize=11)
    fig.savefig(ROOT / "reports/fig/E3_fig2_forest.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("saved")
