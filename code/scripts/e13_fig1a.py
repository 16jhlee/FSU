"""E13-9.6: Fig. 1(a) redrawn from measurements (replaces the collinear schematic). CosyVoice2, en 100 x 2 seeds,
mel RMS (results/E8p9e_onestep.csv, E8'-9e):
  (i)  distance to the teacher mean: one-step output with w = 0 (estimate of x_c), w = 0.7, w = 1.5, the unconditional
       mean x_u; reference = the teacher's own leave-one-out spread d_TT
  (ii) seed-to-seed distance of the same outputs; reference = teacher seed-to-seed distance
python scripts/e13_fig1a.py   (env_eval) -> reports/fig/E13_fig1a_onestep.png, results/E13_fig1a.csv
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e8p_analyze as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BAR, REF = "#2a6fdb", "#9aa0a6"


def main():
    D = pd.read_csv(ROOT / "results/E8p9e_onestep.csv")
    one = D[D.w.isna()]
    seed = D[D.w.notna()].groupby("w")[["d_seed", "d_seed_T"]].mean()
    p1 = [("teacher sample (leave-one-out)", one.d_TT.mean(), True), ("one step, w = 0  (≈ x̄_c)", one.d_T_w0.mean(), False),
          ("one step, w = 0.7 (official)", one.d_T_w07.mean(), False), ("one step, w = 1.5", one.d_T_w15.mean(), False),
          ("unconditional mean x̄_∅", one.d_T_xu.mean(), False)]
    p2 = [("teacher (NFE 10)", float(seed.d_seed_T.mean()), True)] + \
         [(f"one step, w = {w:g}", float(seed.loc[w, "d_seed"]), False) for w in seed.index]
    pd.DataFrame([dict(panel="distance to teacher mean", item=a, value=v) for a, v, _ in p1] +
                 [dict(panel="seed-to-seed distance", item=a, value=v) for a, v, _ in p2]).to_csv(ROOT / "results/E13_fig1a.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=A.SURF)
    for ax, panel, title in ((axes[0], p1, "(i) distance to the teacher mean (mel RMS)"), (axes[1], p2, "(ii) seed-to-seed distance (mel RMS)")):
        labs = [a for a, _, _ in panel][::-1]
        vals = [v for _, v, _ in panel][::-1]
        cols = [REF if r else BAR for _, _, r in panel][::-1]
        ax.barh(labs, vals, color=cols, height=0.55)
        ref = [v for _, v, r in panel if r][0]
        ax.axvline(ref, color=REF, lw=1, ls="--")
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.015, i, f"{v:.2f}", va="center", fontsize=8, color=A.INK)
        ax.set_title(title, fontsize=9.5, color=A.INK, loc="left")
        ax.set_xlim(0, max(vals) * 1.15)
        A.style(ax)
        ax.tick_params(labelsize=8)
    fig.suptitle("Fig. 1(a) — one-step output vs teacher (CosyVoice 2, en 100 × 2 seeds; grey = teacher reference)",
                 x=0.01, ha="left", fontsize=10.5, color=A.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "reports/fig/E13_fig1a_onestep.png"
    fig.savefig(out, dpi=170, facecolor=A.SURF)
    print(out)


if __name__ == "__main__":
    main()
