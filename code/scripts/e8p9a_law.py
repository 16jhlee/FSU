"""E8'-9a: does E_tot = sum_{k: CFG on} w_k RMS(v_c - v_u)(t_k) dt_k explain the baseline SIM loss better than E1?
Inputs: results/E8p8_law.csv (points: model, kind, label, w, loss), results/E8p9a_dump_{cv2,cv3,idx}.csv (per-step RMS
at w=0.7, en 100). w-sweep points reuse the w=0.7 RMS (v6/v7: RMS is treated as w-independent) scaled by w.
E1 here = first term of E_tot from the same dump. Outputs: results/E8p9a_law.csv, results/E8p9a_rho.csv,
reports/fig/E8p9a_fig10b_law.png.  env_eval.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e8p_analyze as A  # noqa: E402

ROOT = A.ROOT
KEY = {A.CV2: "cv2", A.CV3: "cv3", A.IDX: "idx"}


def cond_of(model, label):
    m = re.match(r"NFE (\d+)", label)
    if m:
        return f"all_n{m.group(1)}"
    m = re.match(r"(speaker|token) NFE (\d+)", label)
    if m:
        return f"{m.group(1)}_n{m.group(2)}"
    m = re.match(r"w ([\d.]+) NFE (\d+)", label)
    if m:
        return f"all_n{m.group(2)}"
    m = re.match(r"t1 ([\d.]+)", label)
    if m:
        return f"t1_{m.group(1)}"
    raise ValueError(label)


def main():
    L = pd.read_csv(ROOT / "results/E8p8_law.csv")
    steps = {}
    for model, key in KEY.items():
        f = ROOT / f"results/E8p9a_dump_{key}.csv"
        if f.exists():
            D = pd.read_csv(f)
            steps[model] = D.groupby(["cond", "k"]).agg(t=("t", "first"), dt=("dt", "first"), rms=("rms", "mean")).reset_index()
    rows = []
    for _, p in L.iterrows():
        if p.model not in steps:
            continue
        S = steps[p.model]
        c = cond_of(p.model, p.label)
        q = S[S.cond == c].sort_values("k")
        if q.empty:
            continue
        w = p.w
        e_tot = float((w * q.rms * q.dt).sum())  # baseline: every evaluation uses CFG
        e1 = float(w * q.rms.iloc[0] * q.dt.iloc[0])
        rows.append(dict(model=p.model, kind=p.kind, label=p.label, loss=p.loss, E1=e1, E_tot=e_tot, cond=c))
    P = pd.DataFrame(rows)
    P.to_csv(ROOT / "results/E8p9a_law.csv", index=False)
    rho = []
    for model, g in P.groupby("model"):
        for subset, q in [("all points", g), ("excl. token-null", g[~g.label.str.startswith("token")]),
                          ("excl. token-null NFE 10", g[g.label != "token NFE 10"])]:
            r1, r2 = spearmanr(q.E1, q.loss), spearmanr(q.E_tot, q.loss)
            rho.append(dict(model=model, subset=subset, n=len(q), rho_E1=r1.statistic, rho_Etot=r2.statistic))
        # does the w sweep sit on the curve of the other points? monotone (cummax) interpolation of non-w points
        for xcol in ["E1", "E_tot"]:
            o = g[(g.kind != "w sweep") & (g.label != "token NFE 10")].sort_values(xcol)
            ws = g[g.kind == "w sweep"]
            if len(ws) and len(o) > 2:
                curve = np.maximum.accumulate(o.loss.values)
                pred = np.interp(ws[xcol], o[xcol], curve)
                rho.append(dict(model=model, subset=f"w-sweep residual vs {xcol} curve", n=len(ws),
                                resid_mean=float((ws.loss.values - pred).mean()),
                                resid_abs=float(np.abs(ws.loss.values - pred).mean())))
            t1 = g[g.label == "t1 0.1"]
            if len(t1) and len(o) > 2:
                oo = o[o.label != "t1 0.1"]
                pred = np.interp(t1[xcol], oo[xcol], np.maximum.accumulate(oo.loss.values))
                rho.append(dict(model=model, subset=f"t1=0.1 residual vs {xcol} curve", n=1,
                                resid_mean=float(t1.loss.values[0] - pred[0])))
    R = pd.DataFrame(rho)
    R.to_csv(ROOT / "results/E8p9a_rho.csv", index=False)
    fig(P, R)
    pd.set_option("display.width", 220)
    print(P.round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))


MARK = {"NFE (official schedule)": "o", "branch: speaker-null (R′ Sel-CFG)": "^", "branch: token-null": "v",
        "w sweep": "D", "t1 sweep (NFE 2)": "s"}


def fig(P, R):
    models = [m for m in [A.CV2, A.CV3, A.IDX] if m in set(P.model)]
    fig, axes = plt.subplots(2, len(models), figsize=(5.3 * len(models), 8), constrained_layout=True,
                             facecolor=A.SURF, squeeze=False)
    for j, model in enumerate(models):
        g = P[P.model == model]
        for i, xcol in enumerate(["E1", "E_tot"]):
            ax = axes[i, j]
            for kind, mk in MARK.items():
                q = g[g.kind == kind]
                if not q.empty:
                    ax.scatter(q[xcol], q.loss, marker=mk, s=50, color=A.MODEL_C[model], edgecolor=A.SURF, lw=1.5,
                               zorder=3, label=kind)
            for _, r in g[g.label.isin(["t1 0.1", "token NFE 10", "w 1.5 NFE 2"])].iterrows():
                ax.annotate(r.label, (r[xcol], r.loss), xytext=(4, 3), textcoords="offset points", fontsize=7.5,
                            color=A.INK2)
            rr = R[(R.model == model) & (R.subset == "all points")]
            rho = rr[f"rho_{'E1' if xcol == 'E1' else 'Etot'}"].iloc[0]
            ax.set_title(f"{model} — {'E1 (first step)' if xcol == 'E1' else 'E_tot (all CFG steps)'}   ρ = {rho:.2f}",
                         loc="left", fontsize=10, color=A.INK)
            ax.axhspan(-A.EPS["eres2net"], A.EPS["eres2net"], color=A.GRID, alpha=0.7, lw=0)
            ax.set_xlabel(f"{xcol} = Σ w·RMS(v_c − v_u)·Δt" if xcol == "E_tot" else "E1 = w·RMS(v_c − v_u)(t=0)·Δt1",
                          fontsize=8.5, color=A.INK2)
            A.style(ax)
            if j == 0:
                ax.set_ylabel("baseline SIM loss vs teacher ×100 (en)", fontsize=9, color=A.INK2)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Fig 10-b — first-step term E1 vs all-step guidance size E_tot (grey band = ±ε)", x=0.01, ha="left",
                 fontsize=11, color=A.INK)
    fig.savefig(ROOT / "reports/fig/E8p9a_fig10b_law.png", dpi=150, facecolor=A.SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()
