"""E8'-8 analysis (v6 §2): operating points, H8-1..H8-4, geometry (table 4), Fig 10 (unified first-step law). env_eval.
python scripts/e8p8_analyze.py [ops|law|geometry|all]

All comparisons paired by utterance (seed 0; geometry 8 seeds), SIM x100 (ERes2Net-en main, WavLM-SV / ECAPA), WER/CER %.
E1 (H8-4) = w * RMS(v_c - v_u at t=0, noise state) * Δt1, from the block-D guidance dumps measured at w=0.7:
E1 = (w/0.7) * rms(g1) * Δt1 per model and unconditional branch (g1 RMS over the generated region, en 100 utterances).
y = SIM loss of the condition vs the model's teacher (official NFE, official branch, w=0.7), en 300.
Outputs: results/E8p8_ops.csv, results/E8p8_hyp.csv, results/E8p8_law.csv, results/E8p8_geometry.csv,
         reports/fig/E8p8_fig10_unified_law.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e8p8_jobs as J8  # noqa: E402
import e8p_analyze as A  # noqa: E402
from e7_jobs import wav_dir as e7_wav_dir  # noqa: E402

ROOT = A.ROOT
CV2, CV3, IDX = A.CV2, A.CV3, A.IDX
EMBS, EPS = A.EMBS, A.EPS


def W(name, k, s):
    return e7_wav_dir(name[3:], k, s) if name.startswith("e7:") else J8.wav_dir(name, k, s)


A.wdir = W  # A.load / A.pair resolve E8'-8 names too

# condition -> (config per set group, calls)
SEED = ["en", "zh", "hard"]
OPS = {
    CV2: {"teacher": ("e82_cv2_all_n10_def", 20), "baseline NFE2": ("e82_cv2_all_n2_def", 4), "R (ours)": ("e82_cv2_all_n2_R", 3),
          "R′ (Sel-CFG input_text)": ("e82_cv2_speaker_n2_def", 4), "R′+R": ("e82_cv2_speaker_n2_R", 3), "mid+R (ours)": ("A_cv2_midR", 3),
          "baseline NFE3": ("e82_cv2_all_n3_def", 6), "R′ NFE3": ("e82_cv2_speaker_n3_def", 6),
          "baseline NFE10": ("e82_cv2_all_n10_def", 20), "R′ NFE10": ("e82_cv2_speaker_n10_def", 20)},
    CV3: {"teacher": ("e81_cv3_n10_def", 20), "baseline NFE2": ("e81_cv3_n2_def", 4), "R (ours)": ("e81_cv3_n2_R", 3),
          "R′ (Sel-CFG input_text)": ("B_cv3_Rp_n2", 4), "R′+R": ("B_cv3_RpR_n2", 3), "mid+R (ours)": ("A_cv3_midR", 3),
          "baseline NFE3": ("e81_cv3_n3_def", 6), "R′ NFE3": ("B_cv3_Rp_n3", 6),
          "baseline NFE10": ("e81_cv3_n10_def", 20), "R′ NFE10": ("B_cv3_Rp_n10", 20)},
    IDX: {"teacher": ("e81_idx_n25_def", 50), "baseline NFE2": ("e81_idx_n2_def", 4), "R (ours)": ("e81_idx_n2_R", 3),
          "R′ (Sel-CFG input_text)": ("B_idx_Rp_n2", 4), "R′+R": ("B_idx_RpR_n2", 3), "mid+R (ours)": ("A_idx_midR", 3),
          "baseline NFE3": ("e81_idx_n3_def", 6), "R′ NFE3": ("B_idx_Rp_n3", 6),
          "baseline NFE10": ("B_idx_base_n10", 20), "R′ NFE10": ("B_idx_Rp_n10", 20),
          "[0,0.3,1] baseline": ("C_idx_t03_def", 4), "[0,0.3,1] R (ours)": ("C_idx_t03_R", 3)},
}
MM_OPS = {CV2: {"teacher": ("e85_cv2_n10_def", 20), "baseline NFE2": ("e85_cv2_n2_def", 4), "R (ours)": ("e85_cv2_n2_R", 3),
                "mid+R (ours)": ("A_cv2_midR", 3)},
          CV3: {"teacher": ("e85_cv3_n10_def", 20), "baseline NFE2": ("e85_cv3_n2_def", 4), "R (ours)": ("e85_cv3_n2_R", 3),
                "mid+R (ours)": ("A_cv3_midR", 3)}}
METRICS = [f"sim_{e}" for e in EMBS] + ["wer", "wer_sv", "utmos"]


def ops_for(model, s):
    if s in ("mm_ko", "mm_ja"):
        return MM_OPS.get(model, {})
    return OPS[model]


def cmp(model, s, a, b):
    """paired b - a (SIM x100)."""
    O = ops_for(model, s)
    D, R = A.load(O[a][0], s), A.load(O[b][0], s)
    if D is None or R is None:
        return None
    c = D.index.intersection(R.index)
    r = dict(model=model, set=s, a=a, b=b, n=len(c))
    for m in METRICS:
        if m in D and m in R:
            sc = 100 if m.startswith("sim") else 1
            est, lo, hi, p = A.stat((R.loc[c, m] - D.loc[c, m]) * sc)
            r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"], r[f"d_{m}_p"] = est, lo, hi, p
    return r


def ops():
    rows, hyp = [], []
    for model in OPS:
        for s in SEED + ["mm_ko", "mm_ja"]:
            O = ops_for(model, s)
            for cond, (cfg, calls) in O.items():
                D = A.load(cfg, s)
                if D is None:
                    continue
                r = dict(model=model, set=s, method=cond, config=cfg, calls=calls, n=len(D))
                for m in METRICS:
                    if m in D:
                        r[m] = D[m].mean() * (100 if m.startswith("sim") else 1)
                rows.append(r)
            # hypotheses
            for a, b, tag in [("teacher", "mid+R (ours)", "H8-1"), ("teacher", "R′ (Sel-CFG input_text)", "H8-2 loss"),
                              ("baseline NFE3", "R′ NFE3", "H8-2 NFE3"), ("baseline NFE10", "R′ NFE10", "H8-2 NFE10"),
                              ("R′ (Sel-CFG input_text)", "R′+R", "H8-2 R′+R"), ("teacher", "R (ours)", "ref R"),
                              ("baseline NFE2", "R (ours)", "ref R gain"), ("baseline NFE2", "mid+R (ours)", "mid+R gain"),
                              ("[0,0.3,1] baseline", "[0,0.3,1] R (ours)", "H8-3 t03 gain"),
                              ("teacher", "[0,0.3,1] baseline", "H8-3 t03 base loss")]:
                if a in O and b in O:
                    r = cmp(model, s, a, b)
                    if r:
                        r["hyp"] = tag
                        hyp.append(r)
    Ops, H = pd.DataFrame(rows), pd.DataFrame(hyp)
    # recovery for H8-3 / mid+R: (cond - baseline) / (teacher - baseline)
    for model in OPS:
        for s in SEED:
            g = Ops[(Ops.model == model) & (Ops.set == s)].set_index("method")
            for cond, base in [("R (ours)", "baseline NFE2"), ("mid+R (ours)", "baseline NFE2"),
                               ("[0,0.3,1] R (ours)", "[0,0.3,1] baseline"), ("R′ (Sel-CFG input_text)", "baseline NFE2")]:
                if {cond, base, "teacher"} <= set(g.index):
                    for e in EMBS:
                        gap = g.loc["teacher", f"sim_{e}"] - g.loc[base, f"sim_{e}"]
                        Ops.loc[(Ops.model == model) & (Ops.set == s) & (Ops.method == cond), f"rec_{e}"] = \
                            (g.loc[cond, f"sim_{e}"] - g.loc[base, f"sim_{e}"]) / gap if gap > EPS[e] else np.nan
    Ops.to_csv(ROOT / "results/E8p8_ops.csv", index=False)
    H.to_csv(ROOT / "results/E8p8_hyp.csv", index=False)
    return Ops, H


# ---------------- H8-4 unified law ----------------
GUIDE = {CV2: "results/E8p_guidance.csv", CV3: "results/E8p8_guidance_cv3.csv", IDX: "results/E8p8_guidance_idx.csv"}
TEACHER = {CV2: "e82_cv2_all_n10_def", CV3: "e81_cv3_n10_def", IDX: "e81_idx_n25_def"}


def rms(model, br):
    f = ROOT / GUIDE[model]
    if not f.exists():
        return np.nan
    D = pd.read_csv(f)
    return D[D.branch == br].rms.mean()


def law():
    pts = []

    def add(model, kind, br, w, dt1, cfg, label):
        T, D = A.load(TEACHER[model], "en"), A.load(cfg, "en")
        if T is None or D is None:
            return
        pts.append(dict(model=model, kind=kind, branch=br, w=w, dt1=dt1, config=cfg, label=label,
                        E1=w / 0.7 * rms(model, br) * dt1,
                        loss=(T.sim_eres2net.mean() - D.sim_eres2net.mean()) * 100,
                        loss_wavlm=(T.sim_wavlm.mean() - D.sim_wavlm.mean()) * 100,
                        loss_ecapa=(T.sim_ecapa.mean() - D.sim_ecapa.mean()) * 100))
    cos_dt1 = {n: 1 - np.cos(np.pi / 2 / n) for n in [1, 2, 3, 4, 10]}
    for n in [1, 2, 3, 4, 10]:
        cfg = {1: "e81_cv2_n1_def", 4: "e7:cv2_default_n4"}.get(n, f"e82_cv2_all_n{n}_def")
        add(CV2, "NFE (official schedule)", "all", 0.7, cos_dt1[n], cfg, f"NFE {n}")
        add(CV3, "NFE (official schedule)", "all", 0.7, cos_dt1[n], f"e81_cv3_n{n}_def", f"NFE {n}")
    for n in [1, 2, 3, 4, 25]:
        add(IDX, "NFE (official schedule)", "all", 0.7, 1 / n, f"e81_idx_n{n}_def", f"NFE {n}")
    add(IDX, "NFE (official schedule)", "all", 0.7, 0.1, "B_idx_base_n10", "NFE 10")
    for br, kind in [("speaker", "branch: speaker-null (R′ Sel-CFG)"), ("token", "branch: token-null")]:
        for n in [2, 3, 10]:
            add(CV2, kind, br, 0.7, cos_dt1[n], f"e82_cv2_{br}_n{n}_def", f"{br} NFE {n}")
    for n in [2, 3, 10]:
        add(CV3, "branch: speaker-null (R′ Sel-CFG)", "speaker", 0.7, cos_dt1[n], f"B_cv3_Rp_n{n}", f"speaker NFE {n}")
        add(IDX, "branch: speaker-null (R′ Sel-CFG)", "speaker", 0.7, 1 / n, f"B_idx_Rp_n{n}", f"speaker NFE {n}")
    for n in [2, 3]:
        for w in [0.3, 0.5, 1.0, 1.5]:
            add(CV2, "w sweep", "all", w, cos_dt1[n], f"e84_cv2_w{w}_n{n}_def", f"w {w} NFE {n}")
    for t1 in [0.1, 0.2, 0.3, 0.5, 0.7]:
        add(CV2, "t1 sweep (NFE 2)", "all", 0.7, t1, f"e83_cv2_t{t1}_def", f"t1 {t1}")
        add(CV3, "t1 sweep (NFE 2)", "all", 0.7, t1, f"e83_cv3_t{t1}_def", f"t1 {t1}")
    add(IDX, "t1 sweep (NFE 2)", "all", 0.7, 0.3, "C_idx_t03_def", "t1 0.3")
    Lw = pd.DataFrame(pts)
    Lw.to_csv(ROOT / "results/E8p8_law.csv", index=False)
    rho = []
    for model, g in Lw.groupby("model"):
        for subset, q in [("all points", g), ("excl. token-null", g[g.branch != "token"]),
                          ("excl. t1=0.1", g[g.label != "t1 0.1"])]:
            r = spearmanr(q.E1, q.loss)
            rho.append(dict(model=model, subset=subset, n=len(q), rho=r.statistic, p=r.pvalue))
    R = pd.DataFrame(rho)
    fig10(Lw, R)
    return Lw, R


MARK = {"NFE (official schedule)": "o", "branch: speaker-null (R′ Sel-CFG)": "^", "branch: token-null": "v", "w sweep": "D",
        "t1 sweep (NFE 2)": "s"}


def fig10(Lw, R):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True, facecolor=A.SURF, sharey=True)
    for ax, model in zip(axes, [CV2, CV3, IDX]):
        g = Lw[Lw.model == model]
        for kind, mk in MARK.items():
            q = g[g.kind == kind]
            if q.empty:
                continue
            ax.scatter(q.E1, q.loss, marker=mk, s=54, color=A.MODEL_C[model], edgecolor=A.SURF, lw=1.5, zorder=3, label=kind)
        for _, r in g[g.label.isin(["t1 0.1", "token NFE 10", "NFE 1"])].iterrows():
            ax.annotate(r.label, (r.E1, r.loss), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=A.INK2)
        rr = R[(R.model == model) & (R.subset == "all points")]
        rx = R[(R.model == model) & (R.subset == "excl. t1=0.1")]
        if len(rr):
            ax.set_title(f"{model}   ρ = {rr.rho.iloc[0]:.2f} (all), {rx.rho.iloc[0]:.2f} (excl. t1=0.1)", loc="left",
                         fontsize=10, color=A.INK)
        ax.axhspan(-EPS["eres2net"], EPS["eres2net"], color=A.GRID, alpha=0.7, lw=0)
        ax.set_xlabel("E1 = w · ‖v_c − v_u‖ (t=0, RMS) · Δt1", fontsize=9, color=A.INK2)
        A.style(ax)
    axes[0].set_ylabel("baseline SIM loss vs teacher ×100 (en)", fontsize=9, color=A.INK2)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Fig 10 — unified first-step law: loss vs first-step extrapolation size (grey band = ±ε)",
                 x=0.01, ha="left", fontsize=11, color=A.INK)
    fig.savefig(ROOT / "reports/fig/E8p8_fig10_unified_law.png", dpi=150, facecolor=A.SURF)
    plt.close(fig)


GEOM8 = [(CV2, "R (ours)", "e82_cv2_all_n2_def", "e82_cv2_all_n2_R"),
         (CV2, "R′ (Sel-CFG input_text)", "e82_cv2_all_n2_def", "e82_cv2_speaker_n2_def"),
         (CV2, "mid+R (ours)", "e82_cv2_all_n2_def", "A_cv2_midR"),
         (CV3, "R (ours)", "e81_cv3_n2_def", "e81_cv3_n2_R"),
         (CV3, "R′ (Sel-CFG input_text)", "e81_cv3_n2_def", "B_cv3_Rp_n2"),
         (CV3, "mid+R (ours)", "e81_cv3_n2_def", "A_cv3_midR"),
         (IDX, "R (ours)", "e81_idx_n2_def", "e81_idx_n2_R"),
         (IDX, "R′ (Sel-CFG input_text)", "e81_idx_n2_def", "B_idx_Rp_n2"),
         (IDX, "mid+R (ours)", "e81_idx_n2_def", "A_idx_midR"),
         (IDX, "[0,0.3,1] R vs [0,0.3,1] baseline", "C_idx_t03_def", "C_idx_t03_R"),
         (IDX, "[0,0.3,1] baseline vs NFE2 baseline", "e81_idx_n2_def", "C_idx_t03_def")]


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    if what in ("all", "ops"):
        Ops, H = ops()
        print(Ops[["model", "set", "method", "calls", "sim_eres2net", "sim_wavlm", "sim_ecapa", "wer", "wer_sv", "utmos",
                   "rec_eres2net"]].round(2).to_string(index=False))
        print(H[["hyp", "model", "set", "a", "b", "d_sim_eres2net", "d_sim_eres2net_lo", "d_sim_eres2net_hi", "d_sim_wavlm",
                 "d_sim_ecapa", "d_wer", "d_wer_p", "d_utmos", "d_utmos_p"]].round(3).to_string(index=False))
    if what in ("all", "law"):
        Lw, R = law()
        print(Lw[["model", "kind", "label", "E1", "loss"]].round(3).to_string(index=False))
        print(R.round(3).to_string(index=False))
    if what in ("all", "geometry"):
        O = A.geometry(GEOM8, W, "results/E8p8_geometry.csv")
        print(O[O.emb == "eres2net"].round(4).to_string(index=False))
