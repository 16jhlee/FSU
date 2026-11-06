"""E8' analysis (v5 G1-G7 + E8'-6 geometry). env_eval.
python scripts/e8p_analyze.py [pairs|geometry|figs|all]

Every comparison is paired by utterance at the same NFE: d_X = X(R) - X(default), SIM x100 (ERes2Net-en main,
WavLM-SV / ECAPA alongside), WER/CER in %. recovery = d_sim_eres2net / (SIM(teacher) - SIM(default)), teacher = each
model's official default NFE with default CFG (same set, same seed 0).
R = CFG off for evaluations at t < 0.1 (repo time, t=0 noise) == --cfg_interval 0.1,1. Calls = network evaluations
(a CFG evaluation counts 2: cond + uncond rows of one batch). Masks/Δt1 are computed from each model's schedule.
Outputs: results/E8p_pairs.csv, results/E8p_geometry.csv, reports/fig/E8p_fig{7,8,9}_*.png
"""
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as J  # noqa: E402
from e7_jobs import wav_dir as e7_wav_dir  # noqa: E402

ROOT = J.ROOT
EMBS = ["eres2net", "wavlm", "ecapa"]
EPS = {"eres2net": 0.42, "ecapa": 0.71, "wavlm": 1.06}
rng = np.random.default_rng(0)
CV2, CV3, IDX, F5, ZIP = "CosyVoice2", "CosyVoice3", "IndexTTS-2", "F5-TTS", "ZipVoice"
TEACHER = {CV2: "e82_cv2_all_n10_def", CV3: "e81_cv3_n10_def", IDX: "e81_idx_n25_def", F5: "e81_f5_n32_def",
           ZIP: "e7:zip_default_n16"}


def wdir(name, k, s):
    return e7_wav_dir(name[3:], k, s) if name.startswith("e7:") else J.wav_dir(name, k, s)


# ---------------- schedules, masks, calls ----------------
def sched(model, n, t_span=None):
    if t_span:
        return [float(x) for x in t_span.split(",")]
    u = np.linspace(0, 1, n + 1)
    if model in (CV2, CV3, F5):  # cosine (F5: sway -1 == 1 - cos(pi/2 u))
        return list(1 - np.cos(np.pi / 2 * u))
    if model == ZIP:  # t-shift 0.5
        return list(0.5 * u / (1 - 0.5 * u))
    return list(u)  # IndexTTS-2 s2mel: linspace


def eval_points(ts, solver="euler"):
    pts = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        if solver == "euler":
            pts.append(t0)
        elif solver == "midpoint":
            pts += [t0, (t0 + t1) / 2]
        else:  # heun
            pts += [t0, t1]
    return pts


def calls(pts, R):
    return sum(1 if (R and p < 0.1) else 2 for p in pts)


def mask(pts):
    return "".join("0" if p < 0.1 else "1" for p in pts)


# ---------------- loading / stats ----------------
_cache = {}


def load(name, s):
    key = (name, s)
    if key not in _cache:
        f = wdir(name, 0, s) / "_metrics" / "metrics.csv"
        D = None
        if f.exists():
            D = pd.read_csv(f).set_index("utt_id")
            for c in ("wer", "wer_sv"):
                if c in D:
                    D[c] = D[c] * 100
        _cache[key] = D
    return _cache[key]


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (5000, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5]), (wilcoxon(x).pvalue if np.any(x != 0) else 1.0)


def pair(group, model, label, s, dname, rname, pts, extra=None, teacher=None):
    D, R = load(dname, s), load(rname, s)
    T = load(teacher or TEACHER[model], s)
    if D is None or R is None:
        return None
    c = D.index.intersection(R.index)
    r = dict(group=group, model=model, label=label, set=s, n=len(c), default=dname, R=rname,
             mask_R=mask(pts), calls_def=calls(pts, False), calls_R=calls(pts, True), dt1=pts[1] if len(pts) > 1 else 1.0)
    r.update(extra or {})
    metrics = [f"sim_{e}" for e in EMBS] + ["wer", "wer_sv", "utmos"]
    for m in metrics:
        if m not in D or m not in R:
            continue
        sc = 100 if m.startswith("sim") else 1
        r[f"def_{m}"], r[f"R_{m}"] = D.loc[c, m].mean() * sc, R.loc[c, m].mean() * sc
        est, lo, hi, p = stat((R.loc[c, m] - D.loc[c, m]) * sc)
        r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"], r[f"d_{m}_p"] = est, lo, hi, p
        if T is not None and m in T:
            r[f"T_{m}"] = T[m].mean() * sc
    if T is not None:
        for e in EMBS:
            gap = r.get(f"T_sim_{e}", np.nan) - r[f"def_sim_{e}"]
            r[f"gap_{e}"] = gap
            r[f"rec_{e}"] = r[f"d_sim_{e}"] / gap if gap > EPS[e] else np.nan
    return r


def pairs():
    rows = []
    add = lambda r: rows.append(r) if r else None  # noqa: E731
    seed = [s for s, _, _ in J.SEEDTTS]
    # G1 token-conditioned decoders (+ hard = G7 difficulty)
    for s in seed:
        for n in [1, 2, 3, 4, 10]:
            pts = eval_points(sched(CV2, n))
            if n == 1:
                d, r = "e81_cv2_n1_def", "e81_cv2_n1_R"
            elif n == 4:
                d, r = "e7:cv2_default_n4", "e7:cv2_n4_m0011"
            else:
                d, r = f"e82_cv2_all_n{n}_def", f"e82_cv2_all_n{n}_R"
            add(pair("G1", CV2, f"NFE {n}", s, d, r, pts, dict(nfe=n)))
            add(pair("G1", CV3, f"NFE {n}", s, f"e81_cv3_n{n}_def", f"e81_cv3_n{n}_R", eval_points(sched(CV3, n)), dict(nfe=n)))
        for n in [1, 2, 3, 4, 25]:
            add(pair("G1", IDX, f"NFE {n}", s, f"e81_idx_n{n}_def", f"e81_idx_n{n}_R", eval_points(sched(IDX, n)), dict(nfe=n)))
        # G2 text-conditioned controls
        for n in [4, 8, 16, 32]:
            add(pair("G2", F5, f"NFE {n}", s, f"e81_f5_n{n}_def", f"e81_f5_n{n}_R", eval_points(sched(F5, n)), dict(nfe=n)))
        for n, m in [(2, "01"), (4, "0111"), (8, "00111111"), (16, "0001111111111111")]:
            add(pair("G2", ZIP, f"NFE {n}", s, f"e7:zip_default_n{n}", f"e7:zip_n{n}_m{m}", eval_points(sched(ZIP, n)), dict(nfe=n)))
        # G3 branch
        for br in ["all", "speaker", "token"]:
            for n in [2, 3, 10]:
                add(pair("G3", CV2, f"{br} NFE {n}", s, f"e82_cv2_{br}_n{n}_def", f"e82_cv2_{br}_n{n}_R",
                         eval_points(sched(CV2, n)), dict(nfe=n, branch=br)))
        # G4 first-step size (NFE 2 schedule [0, t1, 1])
        for mdl, tag in [(CV2, "cv2"), (CV3, "cv3")]:
            for t1 in [0.1, 0.2, 0.3, 0.5, 0.7]:
                pts = eval_points(sched(mdl, 2, f"0,{t1},1"))
                add(pair("G4", mdl, f"t1={t1}", s, f"e83_{tag}_t{t1}_def", f"e83_{tag}_t{t1}_R", pts, dict(nfe=2, t1=t1)))
        # G5 solvers
        for sol in ["midpoint", "heun"]:
            for st in [1, 2]:
                pts = eval_points(sched(CV2, st), sol)
                add(pair("G5", CV2, f"{sol} {st} step", s, f"e84_cv2_{sol}_s{st}_def", f"e84_cv2_{sol}_s{st}_R", pts,
                         dict(solver=sol, steps=st, nfe=2 * st)))
        # G6 guidance strength (teacher = official w 0.7 NFE 10)
        for n in [2, 3]:
            for w in ["0.3", "0.5", "1.0", "1.5"]:
                add(pair("G6", CV2, f"w={w} NFE {n}", s, f"e84_cv2_w{w}_n{n}_def", f"e84_cv2_w{w}_n{n}_R",
                         eval_points(sched(CV2, n)), dict(nfe=n, w=float(w))))
    # G7 languages
    for s, _, _ in J.MM:
        for mdl, tag in [(CV2, "cv2"), (CV3, "cv3")]:
            for n in [2, 3]:
                add(pair("G7", mdl, f"NFE {n}", s, f"e85_{tag}_n{n}_def", f"e85_{tag}_n{n}_R", eval_points(sched(mdl, n)),
                         dict(nfe=n), teacher=f"e85_{tag}_n10_def"))
    P = pd.DataFrame(rows)
    lab = pd.read_csv(ROOT / "results/method_labels.csv").set_index("config").method  # v6 §0 labels
    P["method_default"], P["method_R"] = P["default"].map(lab), P["R"].map(lab)
    P.to_csv(ROOT / "results/E8p_pairs.csv", index=False)
    return P


# ---------------- E8'-6 geometry (8 seeds, NFE 2, en/zh) ----------------
def pkey(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosd(a, b):
    return 1 - unit(a) @ unit(b)


def cvar(U):
    return len(U) / (len(U) - 1) * np.mean(np.sum((U - U.mean(0)) ** 2, axis=1))


GEOM = [(CV2, "all", "e82_cv2_all_n2_def", "e82_cv2_all_n2_R"),
        (CV2, "speaker", "e82_cv2_speaker_n2_def", "e82_cv2_speaker_n2_R"),
        (CV2, "token", "e82_cv2_token_n2_def", "e82_cv2_token_n2_R"),
        (CV3, "all", "e81_cv3_n2_def", "e81_cv3_n2_R"),
        (IDX, "all", "e81_idx_n2_def", "e81_idx_n2_R")]


def geometry(geom=None, wdir_fn=None, out_csv="results/E8p_geometry.csv"):
    """geom: [(model, label, default_config, R_config)]; each pair vs the model's teacher 8 seeds (en/zh)."""
    geom, wdir_ = geom or GEOM, wdir_fn or wdir
    out = []
    for s, m, _ in J.SEEDTTS[:2]:
        meta = {l.split("|")[0]: l.split("|")[2] for l in (ROOT / (m + ".meta.lst")).read_text(encoding="utf-8").splitlines()
                if l.strip()}
        for e in EMBS:
            prom = dict(np.load(ROOT / "data/cache" / f"prompt_{e}.npz"))
            cache = {}

            def E(name):
                if name not in cache:
                    arr = [np.load(wdir_(name, k, s) / "_metrics" / f"emb_{e}.npz") for k in range(8)]
                    cache[name] = {u: np.stack([a[f"gen/{u}"] for a in arr]).astype(float) for u in meta}
                return cache[name]
            for model, br, dname, rname in geom:
                tname = TEACHER[model]
                try:
                    Rf, Df, Tf = E(rname), E(dname), E(tname)
                except FileNotFoundError as ex:
                    print("skip geometry", model, br, s, e, ex)
                    continue
                g = unit(unit(np.concatenate(list(Tf.values()))).mean(0))
                recs = []
                for u in Tf:
                    UT = unit(Tf[u])
                    c = unit(UT.mean(0))
                    loo = np.mean([cosd(UT[k], unit(np.delete(UT, k, 0).mean(0))) for k in range(8)])
                    ep = prom[pkey(meta[u])]
                    rec = {}
                    for tag, X in [("def", Df[u]), ("R", Rf[u])]:
                        U = unit(X)
                        rec[f"{tag}_var_ratio"] = np.log(cvar(U) / cvar(UT))
                        rec[f"{tag}_ddcond"] = np.mean([cosd(x, c) for x in U]) - loo
                        rec[f"{tag}_ddglob"] = np.mean([cosd(x, g) for x in U]) - np.mean([cosd(x, g) for x in UT])
                        rec[f"{tag}_norm_ratio"] = np.log(np.linalg.norm(X, axis=1).mean() / np.linalg.norm(Tf[u], axis=1).mean())
                        rec[f"{tag}_dprompt"] = np.mean([cosd(x, ep) for x in U]) - cosd(c, ep)
                    recs.append(rec)
                G = pd.DataFrame(recs)
                for meas in ["var_ratio", "ddcond", "ddglob", "norm_ratio", "dprompt"]:
                    a, b = G[f"R_{meas}"], G[f"def_{meas}"]
                    f = np.exp if meas in ("var_ratio", "norm_ratio") else (lambda z: z)
                    est, lo, hi, p = stat(a - b)
                    out.append(dict(model=model, branch=br, set=s, emb=e, measure=meas, default=f(b.mean()), R=f(a.mean()),
                                    diff=est, lo=lo, hi=hi, p=p, n=len(G)))
    O = pd.DataFrame(out)
    O.to_csv(ROOT / out_csv, index=False)
    return O


# ---------------- figures ----------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_DEF, C_R, C_T = "#2a78d6", "#eb6834", "#52514e"
MODEL_C = {CV2: "#2a78d6", CV3: "#1baf7a", IDX: "#6250d6", F5: "#eda100", ZIP: "#e87ba4"}


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig7(P):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True, facecolor=SURF)
    for ax, model in zip(axes[:3], [CV2, CV3, IDX]):
        Q = P[(P.group == "G1") & (P.model == model) & (P.set == "en")].sort_values("nfe")
        x = np.arange(len(Q))
        ax.plot(x, Q.def_sim_eres2net, "-o", color=C_DEF, lw=2, ms=6, label="baseline")
        ax.plot(x, Q.R_sim_eres2net, "-o", color=C_R, lw=2, ms=6, label="R (ours)")
        ax.axhline(Q.T_sim_eres2net.iloc[0], color=C_T, lw=1, ls="--")
        ax.text(len(Q) - 1, Q.T_sim_eres2net.iloc[0], "teacher", color=INK2, fontsize=8, ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n{a}→{b}" for n, a, b in zip(Q.nfe, Q.calls_def, Q.calls_R)], fontsize=8)
        ax.set_title(model, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("NFE (calls default→R)", fontsize=8.5, color=INK2)
        style(ax)
    axes[0].set_ylabel("SIM-o ×100 (ERes2Net-en, en)", fontsize=9, color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[3]
    G = P[(P.group == "G7") & (P.nfe == 2)]
    H = P[(P.group == "G1") & (P.set == "hard") & (P.nfe == 2) & P.model.isin([CV2, CV3])]
    Q = pd.concat([G, H])
    sets = ["mm_ko", "mm_ja", "mm_de", "hard"]
    for j, model in enumerate([CV2, CV3]):
        y = [Q[(Q.model == model) & (Q.set == s)].d_sim_eres2net.mean() for s in sets]
        lo = [Q[(Q.model == model) & (Q.set == s)].d_sim_eres2net_lo.mean() for s in sets]
        hi = [Q[(Q.model == model) & (Q.set == s)].d_sim_eres2net_hi.mean() for s in sets]
        xx = np.arange(len(sets)) + (j - 0.5) * 0.36
        ax.bar(xx, y, 0.34, color=MODEL_C[model], label=model, edgecolor=SURF, lw=2)
        ax.errorbar(xx, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="none", ecolor=INK2, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(["ko", "ja", "de", "Seed hard"])
    ax.set_title("Languages, NFE 2: ΔSIM (R (ours) − baseline)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.suptitle("Fig 7 — R (ours) across token-conditioned decoders (seed 0, paired)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(ROOT / "reports/fig/E8p_fig7_models.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig8(P):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True, facecolor=SURF)
    Q = P[(P.set == "en") & (((P.group == "G1") & P.model.isin([CV2, CV3, IDX])) | (P.group == "G4"))]
    for model in [CV2, CV3, IDX]:
        M = Q[Q.model == model]
        for grp, mk in [("G1", "o"), ("G4", "s")]:
            X = M[M.group == grp]
            if X.empty:
                continue
            ax.scatter(X.dt1, X.d_sim_eres2net, s=46, marker=mk, color=MODEL_C[model], edgecolor=SURF, lw=1.5, zorder=3,
                       label=f"{model} ({'official schedules' if grp == 'G1' else 'NFE 2, t1 moved'})")
    ax.axhspan(-EPS["eres2net"], EPS["eres2net"], color=GRID, alpha=0.7, lw=0)
    ax.axvline(0.05, color=INK2, lw=0.8, ls=":")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Δt1 = length of the first step (repo time)", fontsize=9, color=INK2)
    ax.set_ylabel("ΔSIM ×100, R (ours) − baseline (en)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper left")
    style(ax)
    ax.set_title("Fig 8 — first-step size law (grey band = ±ε)", loc="left", fontsize=10.5, color=INK)
    fig.savefig(ROOT / "reports/fig/E8p_fig8_first_step.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig9(P):
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.9), constrained_layout=True, facecolor=SURF)
    Q = P[(P.group == "G3") & (P.set == "en")]
    brs = ["all", "speaker", "token"]
    ax = axes[0]  # absolute SIM at NFE 2: default vs R, teacher line
    X = [Q[(Q.branch == b) & (Q.nfe == 2)].iloc[0] for b in brs]
    xx = np.arange(3)
    for i, x in enumerate(X):  # dot plot (truncated axis -> no bars)
        ax.plot([i, i], [x.def_sim_eres2net, x.R_sim_eres2net], color=GRID, lw=2, zorder=1)
    ax.scatter(xx, [x.def_sim_eres2net for x in X], s=70, color=C_DEF, label="no step rule", zorder=3, edgecolor=SURF, lw=2)
    ax.scatter(xx, [x.R_sim_eres2net for x in X], s=70, color=C_R, label="+ R step rule", zorder=3, edgecolor=SURF, lw=2)
    T = X[0].T_sim_eres2net
    ax.axhline(T, color=C_T, lw=1, ls="--")
    ax.text(2.45, T, "teacher (NFE 10)", color=INK2, fontsize=8, ha="right", va="bottom")
    lo = min(x.def_sim_eres2net for x in X)
    ax.set_ylim(lo - 1.5, T + 1)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(xx)
    ax.set_xticklabels(["all-null\nbaseline / R (ours)", "speaker-null\nR′ (Sel-CFG) / R′+R", "token-null\n(analysis)"], fontsize=8)
    ax.set_title("SIM ×100 at NFE 2", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    style(ax)
    for ax, (m, lab) in zip(axes[1:], [("d_sim_eres2net", "ΔSIM ×100 (R − default)"), ("d_wer", "ΔWER % (R − default)")]):
        for j, n in enumerate([2, 3, 10]):
            X = [Q[(Q.branch == b) & (Q.nfe == n)] for b in brs]
            y = [x[m].iloc[0] if len(x) else np.nan for x in X]
            lo = [x[m + "_lo"].iloc[0] if len(x) else np.nan for x in X]
            hi = [x[m + "_hi"].iloc[0] if len(x) else np.nan for x in X]
            xx = np.arange(3) + (j - 1) * 0.27
            ax.bar(xx, y, 0.25, color=["#2a78d6", "#eb6834", "#1baf7a"][j], label=f"NFE {n}", edgecolor=SURF, lw=2)
            ax.errorbar(xx, y, yerr=[np.subtract(y, lo), np.subtract(hi, y)], fmt="none", ecolor=INK2, lw=1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["all-null\n(baseline branch)", "speaker-null\n(R′ branch)", "token-null\n(analysis)"])
        ax.set_title(lab, loc="left", fontsize=10, color=INK)
        style(ax)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig 9 — what the unconditional branch erases decides R (CosyVoice2, en, seed 0)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.savefig(ROOT / "reports/fig/E8p_fig9_branch.png", dpi=150, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    if what in ("all", "pairs", "figs"):
        P = pairs()
        cols = ["group", "model", "label", "set", "n", "mask_R", "calls_def", "calls_R", "def_sim_eres2net", "T_sim_eres2net",
                "d_sim_eres2net", "d_sim_eres2net_p", "rec_eres2net", "d_sim_wavlm", "d_sim_ecapa", "def_wer", "d_wer",
                "d_wer_p", "d_wer_sv", "d_utmos"]
        if what != "figs":
            print(P[[c for c in cols if c in P]].round(3).to_string(index=False))
        fig7(P)
        fig8(P)
        fig9(P)
    if what in ("all", "geometry"):
        O = geometry()
        print(O[O.emb == "eres2net"].round(4).to_string(index=False))
