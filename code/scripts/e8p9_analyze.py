"""E8'-9 b/c/d analysis. env_eval.  python scripts/e8p9_analyze.py [b|c|d|all]
9b: calls vs (SIM, UTMOS) Pareto (Fig 11) for CosyVoice2/3: Euler baseline / R (ours) NFE 1-10, mid+R (+1/+2 final Euler
    steps), external baselines (E8'-7 best NFE-2 settings), teacher; + hf_ratio / flux_var (spectral artefact proxies).
9c: loss vs NFE for {baseline, R (ours), w=0} (IndexTTS-2, CosyVoice2), 8-seed geometry at NFE 4/8, and the two-component
    table: (i) first-step extrapolation = R - baseline at NFE 2, (ii) truncation = teacher - R at NFE 2.
9d: CosyVoice2 G6 recovery recomputed with the teacher at the same w (NFE 10).
Outputs results/E8p9b_pareto.csv, E8p9c_nfe.csv, E8p9c_components.csv, E8p9c_geometry.csv, E8p9d_recovery.csv,
reports/fig/E8p9b_fig11_pareto.png, reports/fig/E8p9c_fig12_two_components.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e8p9_jobs as J9  # noqa: E402
import e8p7_jobs as J7  # noqa: E402
import e8p_analyze as A  # noqa: E402

ROOT = A.ROOT
CV2, CV3, IDX = A.CV2, A.CV3, A.IDX


def W(name, k, s):
    if name.startswith("e87:"):
        return J7.wav_dir(name[4:], k, s)
    return J9.wav_dir(name, k, s)


A.wdir = W


def acoustic(cfg, s):
    f = W(cfg, 0, s) / "_metrics" / "acoustic.csv"
    return pd.read_csv(f).set_index("utt_id") if f.exists() else None


def summary(cfg, s):
    D = A.load(cfg, s)
    if D is None:
        return None
    r = {m: D[m].mean() * (100 if m.startswith("sim") else 1) for m in ["sim_eres2net", "sim_wavlm", "sim_ecapa", "wer", "utmos"]}
    ac = acoustic(cfg, s)
    if ac is not None:
        r["hf_ratio"], r["flux_var"] = ac.hf_ratio.mean(), ac.flux_var.mean()
    return r


# ---------------- 9b ----------------
PARETO = {
    CV2: [("Euler baseline", n, 2 * n, c) for n, c in [(1, "e81_cv2_n1_def"), (2, "e82_cv2_all_n2_def"), (3, "e82_cv2_all_n3_def"),
                                                        (4, "e7:cv2_default_n4"), (10, "e82_cv2_all_n10_def")]] +
         [("R (ours)", n, k, c) for n, k, c in [(1, 1, "e81_cv2_n1_R"), (2, 3, "e82_cv2_all_n2_R"), (3, 5, "e82_cv2_all_n3_R"),
                                                (4, 6, "e7:cv2_n4_m0011"), (10, 17, "e82_cv2_all_n10_R")]] +
         [("mid+R (ours)", 1, 3, "A_cv2_midR"), ("mid+R +1 Euler", 2, 5, "b_cv2_midR1"), ("mid+R +2 Euler", 3, 7, "b_cv2_midR2"),
          ("MAMBO-G(4)", 2, 4, "e87:cv2_mambo4_n2"), ("CFG-Zero* s*", 2, 4, "e87:cv2_s_n2"),
          ("Sel-CFG def_text", 2, 4, "e87:cv2_defltext_n2"), ("R′ (Sel-CFG input_text)", 2, 4, "e82_cv2_speaker_n2_def")],
    CV3: [("Euler baseline", n, 2 * n, f"e81_cv3_n{n}_def") for n in [1, 2, 3, 4, 10]] +
         [("R (ours)", n, k, f"e81_cv3_n{n}_R") for n, k in [(1, 1), (2, 3), (3, 5), (4, 6), (10, 17)]] +
         [("mid+R (ours)", 1, 3, "A_cv3_midR"), ("mid+R +1 Euler", 2, 5, "b_cv3_midR1"), ("mid+R +2 Euler", 3, 7, "b_cv3_midR2"),
          ("MAMBO-G(4)", 2, 4, "e87:cv3_mambo4_n2"), ("CFG-Zero* s*", 2, 4, "e87:cv3_s_n2"),
          ("Sel-CFG def_text", 2, 4, "e87:cv3_defltext_n2"), ("R′ (Sel-CFG input_text)", 2, 4, "B_cv3_Rp_n2")]}


def part_b():
    rows = []
    for model, pts in PARETO.items():
        for s in ["en", "zh"]:
            for meth, n, calls, cfg in pts:
                r = summary(cfg, s)
                if r:
                    rows.append(dict(model=model, set=s, method=meth, steps=n, calls=calls, config=cfg, **r))
    Pb = pd.DataFrame(rows)
    Pb.to_csv(ROOT / "results/E8p9b_pareto.csv", index=False)
    COL = {"Euler baseline": A.C_DEF, "R (ours)": A.C_R, "mid+R (ours)": "#1baf7a", "mid+R +1 Euler": "#1baf7a",
           "mid+R +2 Euler": "#1baf7a", "MAMBO-G(4)": "#6250d6", "CFG-Zero* s*": "#eda100", "Sel-CFG def_text": "#e87ba4",
           "R′ (Sel-CFG input_text)": "#e34948"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True, facecolor=A.SURF)
    for j, model in enumerate([CV2, CV3]):
        for i, m in enumerate(["sim_eres2net", "utmos"]):
            ax = axes[i, j]
            g = Pb[(Pb.model == model) & (Pb.set == "en")]
            for meth in ["Euler baseline", "R (ours)"]:
                q = g[g.method == meth].sort_values("calls")
                ax.plot(q.calls, q[m], "-o", color=COL[meth], lw=2, ms=6, label=meth, zorder=2)
            q = g[g.method.str.startswith("mid+R")].sort_values("calls")
            ax.plot(q.calls, q[m], "-s", color=COL["mid+R (ours)"], lw=2, ms=7, label="mid+R (ours) (+k Euler)", zorder=3)
            for meth in ["MAMBO-G(4)", "CFG-Zero* s*", "Sel-CFG def_text", "R′ (Sel-CFG input_text)"]:
                q = g[g.method == meth]
                ax.scatter(q.calls, q[m], marker="D", s=46, color=COL[meth], edgecolor=A.SURF, lw=1.5, zorder=4,
                           label=f"{meth} NFE 2")
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 10, 17, 20])
            ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "10", "17", "20"])
            ax.set_xlabel("network evaluations (calls)", fontsize=9, color=A.INK2)
            ax.set_ylabel("SIM-o ×100 (ERes2Net, en)" if m.startswith("sim") else "UTMOS (en)", fontsize=9, color=A.INK2)
            ax.set_title(f"{model} — {'SIM' if m.startswith('sim') else 'UTMOS'}", loc="left", fontsize=10, color=A.INK)
            A.style(ax)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Fig 11 — calls vs (SIM, UTMOS): Euler baseline, R (ours), mid+R (ours) (+k final Euler steps), external "
                 "baselines", x=0.01, ha="left", fontsize=11, color=A.INK)
    fig.savefig(ROOT / "reports/fig/E8p9b_fig11_pareto.png", dpi=150, facecolor=A.SURF)
    plt.close(fig)
    return Pb


# ---------------- 9c ----------------
TEACHER = {CV2: "e82_cv2_all_n10_def", IDX: "e81_idx_n25_def"}


def part_c():
    rows = []
    for model, key in [(CV2, "cv2"), (IDX, "idx")]:
        for s in ["en", "zh"]:
            T = A.load(TEACHER[model], s)
            for cond in ["def", "R", "w0"]:
                for n in J9.GRID + ([10] if model == CV2 else []):
                    cfg = f"c_{key}_{cond}_n{n}"
                    if n == 10:
                        cfg = {"def": "e82_cv2_all_n10_def", "R": "e82_cv2_all_n10_R", "w0": "e5:cv2_cfg0_n10"}[cond]
                    D = A.load(cfg, s)
                    if D is None or T is None:
                        continue
                    rows.append(dict(model=model, set=s, cond=cond, nfe=n, sim=D.sim_eres2net.mean() * 100,
                                     loss=(T.sim_eres2net.mean() - D.sim_eres2net.mean()) * 100, wer=D.wer.mean(),
                                     utmos=D.utmos.mean()))
    C = pd.DataFrame(rows)
    C.to_csv(ROOT / "results/E8p9c_nfe.csv", index=False)
    comp = []
    for model in [CV2, IDX]:
        for s in ["en", "zh"]:
            g = C[(C.model == model) & (C.set == s)].set_index(["cond", "nfe"])
            for n in [2, 3, 4]:
                if ("def", n) in g.index and ("R", n) in g.index:
                    comp.append(dict(model=model, set=s, nfe=n, total=g.loc[("def", n), "loss"],
                                     i_first_step=g.loc[("def", n), "loss"] - g.loc[("R", n), "loss"],
                                     ii_truncation=g.loc[("R", n), "loss"],
                                     w0_loss=g.loc[("w0", n), "loss"] if ("w0", n) in g.index else np.nan))
    Cp = pd.DataFrame(comp)
    Cp.to_csv(ROOT / "results/E8p9c_components.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, facecolor=A.SURF, sharey=True)
    for ax, model in zip(axes, [CV2, IDX]):
        g = C[(C.model == model) & (C.set == "en")]
        for cond, col, lab in [("def", A.C_DEF, "baseline"), ("R", A.C_R, "R (ours)"), ("w0", A.INK2, "w = 0 (no CFG)")]:
            q = g[g.cond == cond].sort_values("nfe")
            ax.plot(q.nfe, q.loss, "-o", color=col, lw=2, ms=6, label=lab)
        ax.axhspan(-A.EPS["eres2net"], A.EPS["eres2net"], color=A.GRID, alpha=0.7, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 4, 6, 8, 10, 12, 25])
        ax.set_xticklabels(["2", "3", "4", "6", "8", "10", "12", "25"])
        ax.set_xlabel("NFE", fontsize=9, color=A.INK2)
        ax.set_title(model, loc="left", fontsize=10, color=A.INK)
        A.style(ax)
    axes[0].set_ylabel("SIM loss vs teacher ×100 (en)", fontsize=9, color=A.INK2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 12 — two components: (i) first-step extrapolation = baseline − R, (ii) truncation = R − teacher",
                 x=0.01, ha="left", fontsize=11, color=A.INK)
    fig.savefig(ROOT / "reports/fig/E8p9c_fig12_two_components.png", dpi=150, facecolor=A.SURF)
    plt.close(fig)
    geo = [(CV2, f"NFE {n} R vs baseline", f"c_cv2_def_n{n}", f"c_cv2_R_n{n}") for n in (4, 8)] + \
          [(IDX, f"NFE {n} R vs baseline", f"c_idx_def_n{n}", f"c_idx_R_n{n}") for n in (4, 8)]
    G = A.geometry(geo, W, "results/E8p9c_geometry.csv")
    return C, Cp, G


# ---------------- 9d ----------------
def part_d():
    TW = {0.3: "e5:cv2_cfg0.3_n10", 0.5: "d_cv2_w0.5_n10", 0.7: "e82_cv2_all_n10_def", 1.0: "e5:cv2_cfg1.0_n10",
          1.5: "e5:cv2_cfg1.5_n10"}
    rows = []
    for s in ["en", "zh"]:
        T07 = A.load(TW[0.7], s)
        for w, tcfg in TW.items():
            Tw = A.load(tcfg, s)
            for n in [2, 3]:
                if w == 0.7:
                    b, r = f"e82_cv2_all_n{n}_def", f"e82_cv2_all_n{n}_R"
                else:
                    b, r = f"e84_cv2_w{w}_n{n}_def", f"e84_cv2_w{w}_n{n}_R"
                B, R = A.load(b, s), A.load(r, s)
                if Tw is None or B is None or R is None:
                    continue
                gain = (R.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100
                gw = (Tw.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100
                g7 = (T07.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100
                rows.append(dict(set=s, w=w, nfe=n, teacher_w_sim=Tw.sim_eres2net.mean() * 100, gain=gain,
                                 gap_teacher_w=gw, rec_teacher_w=gain / gw if gw > A.EPS["eres2net"] else np.nan,
                                 gap_teacher_07=g7, rec_teacher_07=gain / g7 if g7 > A.EPS["eres2net"] else np.nan))
    Dd = pd.DataFrame(rows)
    Dd.to_csv(ROOT / "results/E8p9d_recovery.csv", index=False)
    return Dd


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 300)
    if what in ("all", "b"):
        print(part_b().round(3).to_string(index=False))
    if what in ("all", "c"):
        C, Cp, G = part_c()
        print(C.round(2).to_string(index=False))
        print(Cp.round(2).to_string(index=False))
        print(G[(G.emb == "eres2net") & G.measure.isin(["var_ratio", "ddcond"])].round(4).to_string(index=False))
    if what in ("all", "d"):
        print(part_d().round(3).to_string(index=False))
