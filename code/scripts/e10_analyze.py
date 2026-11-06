"""E10 analysis (v10).  python scripts/e10_analyze.py [fig61|holm|fullset|wsign|fill|all]   (env_eval)
  fig61   E10-5.1 unified observation: NFE 2 methods of Table 7' (CosyVoice2/3, en/zh), x = 8-seed Δd_cond (ERes2Net,
          E8p7/E8p8/E8p10 geometry), y = recovery (ERes2Net); auxiliary x = |w1| * RMS(v_c - v_u)|t=0 (branch RMS from E8'-8
          block D; MAMBO-G from the logged r at t=0; CFG-Zero* s* and JRR: no single first-step scale -> not plotted).
          Spearman(Δd_cond, recovery). -> results/E10_fig6_1.csv, reports/fig/E10_fig6_1_unified.png
  holm    E10-5.2 Holm correction per table (Table 7' cells: Δ vs baseline, Δ vs R; SIM ERes2Net, en and zh)
          -> results/E10_holm_table7.csv
  fullset E10-2 -> results/E10_full_table1.csv (+ Holm), results/E10_full_subset_check.csv
  wsign   E10-3 -> results/E10_wsign.csv, results/E10_wsign_geometry.csv, reports/fig/E10_fig14_wsign.png
  fill    E10-4 -> results/E10_fill.csv, results/E10_fill_geometry.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e10_jobs as J  # noqa: E402
import e8p_analyze as A  # noqa: E402

ROOT = J.ROOT
FIG = ROOT / "reports" / "fig"
CV2, CV3, IDX = "CosyVoice2", "CosyVoice3", "IndexTTS-2"
MKEY = {"cv2": CV2, "cv3": CV3, "idx": IDX}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
rng = np.random.default_rng(0)


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def holm(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, run = len(idx), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


# ---------------- E10-5.1 ----------------
def guidance_rms():
    """RMS(v_c - v_u) at t=0 per branch (E8'-8 block D stores w * RMS with w = 0.7; en 100)."""
    out = {}
    for model, f in [(CV2, "E8p_guidance"), (CV3, "E8p8_guidance_cv3")]:
        G = pd.read_csv(ROOT / f"results/{f}.csv")
        out[model] = {b: G[G.branch == b].rms.mean() / 0.7 for b in ("all", "speaker", "token")}
    return out


def fig61():
    G = pd.concat([pd.read_csv(ROOT / f"results/{f}.csv").assign(src=f) for f in
                   ("E8p8_geometry", "E8p7_geometry", "E8p10_geometry")])
    G = G[(G.emb == "eres2net") & (G.measure == "ddcond")]
    T7 = pd.concat([pd.read_csv(ROOT / "results/E8p7_table7.csv"), pd.read_csv(ROOT / "results/E8p10_table.csv")])
    T7 = T7[T7.nfe == 2]
    T1 = pd.read_csv(ROOT / "results/E9_table1.csv")
    rms = guidance_rms()
    tv = pd.read_csv(ROOT / "results/E8p10_tv_scales.csv").set_index(["nfe", "schedule"]).w_first
    mr = pd.read_csv(ROOT / "results/E8p7_mambo_r.csv")
    mr = mr[(mr.nfe == 2) & (mr.t == 0)].r.values
    # method -> (geometry label, recovery label in Table 7' / Table 1, first-step |w| x branch, marker kind)
    METH = [("baseline", None, "baseline", ("all", 0.7), "base"),
            ("R (ours)", "R (ours)", "R (ours)", ("all", 0.0), "ours"),
            ("mid+R (ours)", "mid+R (ours)", "T1:mid+R (ours)", ("all", 0.0), "ours"),
            ("R′ (Sel-CFG input_text)", "R′ (Sel-CFG input_text)", "R′ (Sel-CFG input_text)", ("speaker", 0.7), "ext"),
            ("R′+R", "Sel-CFG def_text+R = R′+R", "Sel-CFG def_text+R", ("all", 0.0), "ext"),
            ("Sel-CFG def_text", "Sel-CFG def_text", "Sel-CFG def_text", ("all", 0.7), "ext"),
            ("MAMBO-G(4)", "MAMBO-G(4)", "MAMBO-G(4)", ("all", float(np.mean(0.7 * np.exp(-4 * mr)))), "ext"),
            ("CFG-Zero* s*", "CFG-Zero* s*", "CFG-Zero* s*", None, "ext"),
            ("TFR", "TFR", "TFR", ("all", 0.0), "ext"),
            ("PCI", "PCI", "PCI", ("all", 0.7), "ext"),
            ("TV-CFG stepup", "TV-CFG stepup", "TV-CFG stepup", ("all", abs(tv[(2, "stepup")])), "neg"),
            ("TV-CFG symmetric (official)", "TV-CFG symmetric", "TV-CFG symmetric", ("all", abs(tv[(2, "symmetric")])), "collapse"),
            ("JRR-CFG", "JRR-CFG", "JRR-CFG", None, "ext"),
            ("R (ours) + PCI", "R (ours) + PCI", "R (ours) + PCI", ("all", 0.0), "ours"),
            ("R (ours) + TFR", "R (ours) + TFR", "R (ours) + TFR", ("all", 0.0), "ours")]
    rows = []
    for model in (CV2, CV3):
        for s in ("en", "zh"):
            g = G[(G.model == model) & (G.set == s)]
            base_dd = g["default"].iloc[0]
            for name, glab, rlab, wx, kind in METH:
                if glab is None:
                    dd = base_dd
                else:
                    r = g[g.branch.isin([glab, glab.replace("R′+R", "R′+R")])]
                    if r.empty and glab == "R (ours)":
                        r = g[g.branch == "R (ours) = Kynk [0.1,0.9]"]
                    if r.empty:
                        continue
                    dd = r.R.iloc[0]
                if rlab == "baseline":
                    rec = 0.0
                elif rlab.startswith("T1:"):
                    t = T1[(T1.model == model) & (T1.NFE.astype(str) == "2") & (T1.method == rlab[3:])]
                    rec = t[f"rec {s}"].iloc[0] / 100 if len(t) else np.nan
                else:
                    t = T7[(T7.model == model) & (T7.method == rlab) & (T7.set == s)]
                    rec = t.recovery.iloc[0] if len(t) else np.nan
                aux = np.nan if wx is None else wx[1] * rms[model][wx[0]]
                rows.append(dict(model=model, set=s, method=name, kind=kind, ddcond=dd, recovery=rec, w1_abs=None if wx is None else wx[1],
                                 branch=None if wx is None else wx[0], aux_x=aux))
    D = pd.DataFrame(rows)
    # E10-3 / E10-4 points (if analysed): w1 sweep and R′+R CV3 are added as separate markers
    W = ROOT / "results/E10_wsign_geometry_summary.csv"
    if W.exists():
        Wg = pd.read_csv(W)
        for _, r in Wg.iterrows():
            D.loc[len(D)] = dict(model=r.model, set=r.set, method=f"w1 = {r.w1:g} (E10-3)", kind="wsweep", ddcond=r.ddcond,
                                 recovery=r.recovery, w1_abs=abs(r.w1), branch="all", aux_x=abs(r.w1) * rms[r.model]["all"])
    F = ROOT / "results/E10_fill_geometry_summary.csv"
    if F.exists():
        Fg = pd.read_csv(F)
        for _, r in Fg[Fg.model == CV3].iterrows():
            if not ((D.model == CV3) & (D.set == r.set) & (D.method == "R′+R")).any():
                D.loc[len(D)] = dict(model=CV3, set=r.set, method="R′+R", kind="ext", ddcond=r.ddcond, recovery=r.recovery,
                                     w1_abs=0.0, branch="all", aux_x=0.0)
    D.to_csv(ROOT / "results/E10_fig6_1.csv", index=False)
    st = []
    for model in (CV2, CV3):
        for s in ("en", "zh", "pooled"):
            for incl in ("Table 7' methods (no collapse)", "+ w1 sweep (E10-3)"):
                d = D[(D.model == model) & ((D.set == s) if s != "pooled" else True) & (D.kind != "collapse")]
                d = d[d.kind != "wsweep"] if incl.startswith("Table") else d
                d = d.dropna(subset=["recovery"])
                if len(d) >= 4:
                    rho, p = spearmanr(d.ddcond, d.recovery)
                    da = d.dropna(subset=["aux_x"])
                    rho2, p2 = spearmanr(da.aux_x, da.recovery) if len(da) >= 4 else (np.nan, np.nan)
                    st.append(dict(model=model, set=s, points=incl, n=len(d), rho_ddcond=rho, p=p, n_aux=len(da), rho_aux=rho2, p_aux=p2))
    S = pd.DataFrame(st)
    S.to_csv(ROOT / "results/E10_fig6_1_spearman.csv", index=False)
    print(S.round(3).to_string())
    KC = {"base": ("#52514e", "s"), "ours": ("#eb6834", "o"), "ext": ("#2a78d6", "o"), "neg": ("#1baf7a", "v"),
          "collapse": ("#c4314b", "X"), "wsweep": ("#6250d6", "D")}
    KL = {"base": "baseline", "ours": "R (ours) family", "ext": "external rules", "neg": "w1 < 0 (TV-CFG stepup)",
          "collapse": "collapse (TV-CFG symmetric official; off-scale)", "wsweep": "E10-3 w1 sweep"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for j, model in enumerate((CV2, CV3)):
        for i, xcol in enumerate(("ddcond", "aux_x")):
            ax = axes[i, j]
            style(ax)
            d = D[D.model == model]
            for kind, (c, mk) in KC.items():
                e = d[d.kind == kind].dropna(subset=[xcol, "recovery"])
                if kind == "collapse":
                    for _, r in e.iterrows():
                        ax.annotate(f"{r.method} {r.set}: x={r[xcol]:.2f}, rec={100 * r.recovery:.0f}%", xy=(0.98, 0.04 + 0.05 * (r.set == "zh")),
                                    xycoords="axes fraction", ha="right", fontsize=7, color=c)
                    continue
                for s, fill in (("en", c), ("zh", "none")):
                    q = e[e.set == s]
                    ax.scatter(q[xcol], 100 * q.recovery, s=42, marker=mk, facecolors=fill, edgecolors=c, lw=1.4,
                               label=f"{KL[kind]} ({s})" if (i == 0 and j == 0) else None, zorder=3)
            for _, r in d[(d.set == "en") & (d.kind.isin(["ours", "ext", "neg", "base"]))].dropna(subset=[xcol, "recovery"]).iterrows():
                ax.annotate(r.method, (r[xcol], 100 * r.recovery), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2)
            ax.axhline(0, color=INK2, lw=0.8)
            s_ = S[(S.model == model) & (S.set == "pooled") & (S.points.str.startswith("Table"))]
            rho = s_.rho_ddcond.iloc[0] if xcol == "ddcond" and len(s_) else (s_.rho_aux.iloc[0] if len(s_) else np.nan)
            ax.set_title(f"{model} NFE 2 — Spearman ρ (en+zh) = {rho:+.2f}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("8-seed Δd_cond (ERes2Net)" if xcol == "ddcond" else "first-step effective guidance |w₁|·RMS(v_c − v_u)|ₜ₌₀",
                          fontsize=9, color=INK2)
            ax.set_ylabel("recovery of the NFE 2 SIM loss (%)", fontsize=9, color=INK2)
            if xcol == "ddcond":
                xmax = max(0.08, d[~d.kind.isin(["collapse", "wsweep"])].ddcond.max() * 1.1)
                ax.set_xlim(-0.003, xmax)
                off = d[(d.kind == "wsweep") & ((d.ddcond > xmax) | (d.recovery < -0.4))]
                if len(off):
                    ax.text(0.98, 0.14, f"E10-3 points off-scale (w₁ ≤ −0.7): {len(off)}", transform=ax.transAxes,
                            ha="right", fontsize=7, color="#6250d6")
            ax.set_ylim(-40, 110)
    fig.legend(loc="lower center", ncol=4, fontsize=7.5, frameon=False)
    fig.text(0.01, 0.995, "Filled = en, open = zh. Top: x = 8-seed Δd_cond; bottom: x = first-step effective guidance "
             "(CFG-Zero* s* and JRR have no single first-step scale and are omitted there).", fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "E10_fig6_1_unified.png", dpi=160, facecolor="white")
    print(FIG / "E10_fig6_1_unified.png")
    return D, S


# ---------------- E10-5.2 Holm ----------------
def holm_table7():
    T = pd.concat([pd.read_csv(ROOT / "results/E8p7_table7.csv").assign(src="E8′-7"),
                   pd.read_csv(ROOT / "results/E8p10_table.csv").assign(src="E8′-10")])
    T = T.drop_duplicates(subset=["model", "nfe", "method", "set"], keep="first")
    cells = []
    for _, r in T.iterrows():
        for cmp_ in ("vs_base", "vs_R"):
            p, d = r.get(f"{cmp_}_d_sim_eres2net_p"), r.get(f"{cmp_}_d_sim_eres2net")
            if pd.isna(p) or (cmp_ == "vs_base" and r.method == "baseline") or (cmp_ == "vs_R" and r.method == "R (ours)"):
                continue
            if d == 0 and p == 1.0:  # alias of the reference itself (e.g. Kynk == R)
                continue
            cells.append(dict(model=r.model, nfe=r.nfe, method=r.method, set=r.set, comparison=cmp_, d=d, p=p, src=r.src))
    C = pd.DataFrame(cells)
    C["p_holm"] = holm(C.p.values)
    C["star_raw"], C["star_holm"] = C.p < 0.05, C.p_holm < 0.05
    C["changed"] = C.star_raw != C.star_holm
    C.to_csv(ROOT / "results/E10_holm_table7.csv", index=False)
    print(f"Table 7': {len(C)} cells, raw * {C.star_raw.sum()}, Holm * {C.star_holm.sum()}, changed {C.changed.sum()}")
    print(C[C.changed].to_string())
    return C


# ---------------- shared paired comparison ----------------
def load_metrics(d):
    f = d / "_metrics" / "metrics.csv"
    if not f.exists():
        return None
    D = pd.read_csv(f).set_index("utt_id")
    for extra in ("objq.csv", "objq_dnsmos.csv"):  # E10-1′ objective quality predictors
        g = d / "_metrics" / extra
        if g.exists():
            x = pd.read_csv(g).set_index("utt_id")
            D = D.join(x[[c for c in x.columns if c not in D.columns]], how="left")
    for c in ("wer", "wer_wh"):
        if c in D:
            D[c] = D[c] * 100
    for c in [c for c in D.columns if c.startswith("sim_")]:
        D[c] = D[c] * 100
    return D


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (5000, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5]), (wilcoxon(x).pvalue if np.any(x != 0) else 1.0)


METRICS = ["sim_eres2net", "sim_wavlm", "sim_ecapa", "wer", "wer_wh", "utmos", "utmosv2", "dnsmos_ovrl", "dnsmos_sig",
           "dnsmos_bak", "nisqa_mos", "nisqa_tts"]


def compare(X, B, T, prefix_base="vs_base", prefix_T="vs_teacher"):
    r = {}
    c = X.index.intersection(B.index) if B is not None else X.index
    r["n"] = len(c)
    for m in METRICS:
        if m not in X:
            continue
        r[m] = X.loc[c, m].mean()
        for ref, pre in ((B, prefix_base), (T, prefix_T)):
            if ref is not None and m in ref and ref is not X:
                est, lo, hi, p = stat(X.loc[c, m] - ref.loc[c, m])
                r.update({f"{pre}_d_{m}": est, f"{pre}_d_{m}_lo": lo, f"{pre}_d_{m}_hi": hi, f"{pre}_d_{m}_p": p})
    return r


# ---------------- E10-2 full sets ----------------
SUBSET_REF = {  # Table 1 subset configs (same conditions, 300 / 300 / 100)
    "cv2": {"base_n2": "e82_cv2_all_n2_def", "R_n2": "e82_cv2_all_n2_R", "midR": "A_cv2_midR", "base_n3": "e82_cv2_all_n3_def",
            "R_n3": "e82_cv2_all_n3_R", "teacher": "e82_cv2_all_n10_def"},
    "cv3": {"base_n2": "e81_cv3_n2_def", "R_n2": "e81_cv3_n2_R", "midR": "A_cv3_midR", "base_n3": "e81_cv3_n3_def",
            "R_n3": "e81_cv3_n3_R", "teacher": "e81_cv3_n10_def"},
    "idx": {"base_n2": "e81_idx_n2_def", "R_n2": "e81_idx_n2_R", "midR": "A_idx_midR", "base_n3": "e81_idx_n3_def",
            "R_n3": "e81_idx_n3_R", "teacher": "e81_idx_n25_def"},
}
TEACHER_CALLS = {"cv2": 20, "cv3": 20, "idx": 50}


def fullset():
    rows, chk = [], []
    for model in ("cv2", "cv3", "idx"):
        for s in ("en", "zh", "hard", "lspc"):
            Dm = {key: load_metrics(J.wav_dir(f"f_{model}_{key}", 0, s)) if (f"f_{model}_{key}", s) in J.ALIAS or
                  (J.wav_dir(f"f_{model}_{key}", 0, s) / "_done").exists() else None for key, *_ in J.COND}
            if all(v is None for v in Dm.values()):
                continue
            T = Dm["teacher"]
            for key, label, n, _ in J.COND:
                X = Dm[key]
                if X is None:
                    continue
                nfe = J.TEACHER_NFE[model] if key == "teacher" else n
                base = Dm["base_n2"] if key in ("base_n2", "R_n2", "midR") else Dm["base_n3"] if key in ("base_n3", "R_n3") else None
                r = dict(model=MKEY[model], set=s, cond=key, method=label, nfe=nfe,
                         calls=TEACHER_CALLS[model] if key == "teacher" else J.CALLS[key])
                r.update(compare(X, base if base is not X else None, T if T is not X else None))
                if base is not None and T is not None and base is not X:
                    gap = T.sim_eres2net.mean() - base.sim_eres2net.mean()
                    r["teacher_minus_base"] = gap
                    r["recovery"] = (X.sim_eres2net.mean() - base.sim_eres2net.mean()) / gap if gap > 0.42 else np.nan
                if T is not None and "utmos" in X:
                    r["utmos_minus_teacher"] = X.utmos.mean() - T.utmos.mean()
                rows.append(r)
                # subset consistency: same utterances in the full run vs the subset run (determinism + S1 subset value)
                if s in ("en", "zh", "hard"):
                    S = load_metrics(A.wdir(SUBSET_REF[model][key], 0, s) if not SUBSET_REF[model][key].startswith("A_")
                                     else J.J9.wav_dir(SUBSET_REF[model][key], 0, s))
                    if S is not None:
                        c = S.index.intersection(X.index)
                        chk.append(dict(model=MKEY[model], set=s, cond=key, n_common=len(c),
                                        max_abs_d_sim=float(np.max(np.abs(S.loc[c, "sim_eres2net"] - X.loc[c, "sim_eres2net"]))),
                                        sim_subset_run=S.loc[c, "sim_eres2net"].mean(), sim_full_run_on_subset=X.loc[c, "sim_eres2net"].mean(),
                                        sim_full=X.sim_eres2net.mean()))
    F = pd.DataFrame(rows)
    if F.empty:
        print("no full-set results yet")
        return F
    pc = [c for c in F.columns if c.startswith("vs_base_d_sim_eres2net_p") or c.startswith("vs_teacher_d_sim_eres2net_p")]
    for c in pc:
        F[c.replace("_p", "_p_holm")] = holm(F[c].values)
    F.to_csv(ROOT / "results/E10_full_table1.csv", index=False)
    if chk:
        pd.DataFrame(chk).to_csv(ROOT / "results/E10_full_subset_check.csv", index=False)
    print(F[["model", "set", "cond", "n", "sim_eres2net", "wer", "utmos", "recovery"]].round(3).to_string())
    return F


# ---------------- geometry on a list of utterances (seed 0 from the 300 folder, seeds 1-7 from the 100 / 300 folder) ----
def emb_stack(name, s, e, utts, wdir):
    arr = []
    for k in range(8):
        s_k = s if k == 0 else s
        d = wdir(name, k, s_k)
        if not (d / "_metrics" / f"emb_{e}.npz").exists():
            alt = {"en": "en100", "zh": "zh100"}[s]
            d = wdir(name, k, alt)
        arr.append(np.load(d / "_metrics" / f"emb_{e}.npz"))
    return {u: np.stack([a[f"gen/{u}"] for a in arr]).astype(float) for u in utts}


def geometry_on(pairs, utt_lists, wdir, emb="eres2net"):
    """pairs: [(model, label, config)]; teacher = the model's teacher 8 seeds; measures as e8p_analyze.geometry()."""
    out = []
    for s, lst in utt_lists.items():
        meta = {l.split("|")[0]: l.split("|")[2] for l in (ROOT / (lst + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()}
        utts = list(meta)
        prom = dict(np.load(ROOT / "data/cache" / f"prompt_{emb}.npz"))
        cache = {}

        def E(name):
            if name not in cache:
                cache[name] = emb_stack(name, s, emb, utts, wdir)
            return cache[name]
        for model, label, cfg in pairs:
            try:
                X, Tf = E(cfg), E(A.TEACHER[model])
            except (FileNotFoundError, KeyError) as ex:
                print("skip geometry", model, label, s, ex)
                continue
            recs = []
            for u in utts:
                UT = A.unit(Tf[u])
                c = A.unit(UT.mean(0))
                loo = np.mean([A.cosd(UT[k], A.unit(np.delete(UT, k, 0).mean(0))) for k in range(8)])
                ep = prom[A.pkey(meta[u])]
                U = A.unit(X[u])
                recs.append(dict(var_ratio=np.log(A.cvar(U) / A.cvar(UT)), ddcond=np.mean([A.cosd(x, c) for x in U]) - loo,
                                 norm_ratio=np.log(np.linalg.norm(X[u], axis=1).mean() / np.linalg.norm(Tf[u], axis=1).mean()),
                                 dprompt=np.mean([A.cosd(x, ep) for x in U]) - A.cosd(c, ep)))
            G = pd.DataFrame(recs)
            for meas in G.columns:
                v = G[meas]
                est, lo, hi, p = stat(v)
                f = np.exp if meas in ("var_ratio", "norm_ratio") else (lambda z: z)
                out.append(dict(model=model, label=label, config=cfg, set=s, emb=emb, measure=meas, value=f(v.mean()),
                                lo=f(lo), hi=f(hi), n=len(G)))
    return pd.DataFrame(out)


# ---------------- E10-3 ----------------
def ws_config(model, w):
    if (model, w) in J.WS_EXIST:
        return J.WS_EXIST[(model, w)]
    return J.WS_NAME[(model, w)]


def ws_wdir(name, k, s):
    """seed 0: 300 folders; seeds 1-7: E10 100 folders for new configs, the existing 300 folders for reused runs."""
    if name.startswith("ws_"):
        if name == "ws_cv3_w-0.43" and k == 0:
            return J.J10.wav_dir("cv3_tvu_n2", 0, s)
        return J.wav_dir(name, k, s)
    if name.startswith("e8p10:"):
        if name == "e8p10:cv3_tvu_n2" and k >= 1:
            return J.wav_dir("ws_cv3_w-0.43", k, s)
        return J.J10.wav_dir(name[6:], k, s.replace("100", ""))
    return A.wdir(name, k, s.replace("100", "")) if not name.startswith("A_") else J.J9.wav_dir(name, k, s.replace("100", ""))


def wsign():
    rows = []
    for model in ("cv2", "cv3"):
        M = MKEY[model]
        base = {s: load_metrics(ws_wdir(ws_config(model, 0.7), 0, s)) for s in ("en", "zh")}
        T = {s: load_metrics(A.wdir(A.TEACHER[M], 0, s)) for s in ("en", "zh")}
        R = {s: load_metrics(ws_wdir(ws_config(model, 0.0), 0, s)) for s in ("en", "zh")}
        for w in J.W1:
            cfg = ws_config(model, w)
            for s in ("en", "zh"):
                X = load_metrics(ws_wdir(cfg, 0, s))
                if X is None:
                    continue
                r = dict(model=M, w1=w if w != -0.43 else -0.4333, set=s, config=cfg)
                r.update(compare(X, base[s] if cfg != ws_config(model, 0.7) else None, T[s]))
                if R[s] is not None and cfg != ws_config(model, 0.0):
                    est, lo, hi, p = stat(X.sim_eres2net - R[s].loc[X.index, "sim_eres2net"])
                    r.update(vs_R_d_sim_eres2net=est, vs_R_d_sim_eres2net_lo=lo, vs_R_d_sim_eres2net_hi=hi, vs_R_d_sim_eres2net_p=p)
                gap = T[s].sim_eres2net.mean() - base[s].sim_eres2net.mean()
                r["recovery"] = (X.sim_eres2net.mean() - base[s].sim_eres2net.mean()) / gap
                rows.append(r)
    W = pd.DataFrame(rows)
    W.to_csv(ROOT / "results/E10_wsign.csv", index=False)
    pairs = [(MKEY[m], f"w1={w}", ws_config(m, w)) for m in ("cv2", "cv3") for w in J.W1]
    G = geometry_on(pairs, {"en": "data/lists/en_100", "zh": "data/lists/zh_100"}, ws_wdir)
    if not G.empty:
        G["w1"] = G.label.str[3:].astype(float).replace(-0.43, -0.4333)
        G.to_csv(ROOT / "results/E10_wsign_geometry.csv", index=False)
        sm = G[G.measure == "ddcond"][["model", "set", "w1", "value"]].rename(columns={"value": "ddcond"})
        sm = sm.merge(W[["model", "set", "w1", "recovery"]], on=["model", "set", "w1"], how="left")
        sm.to_csv(ROOT / "results/E10_wsign_geometry_summary.csv", index=False)
    print(W[["model", "set", "w1", "sim_eres2net", "recovery", "wer", "utmos"]].round(3).to_string())
    return W, G


def fig14(W, G):
    dist = [pd.read_csv(f).assign(model=MKEY[f.stem.split("_")[-1]]) for f in ROOT.glob("results/E10_wsign_dist_*.csv")]
    Dd = pd.concat(dist) if dist else None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    C = {(CV2, "en"): ("#2a78d6", "-"), (CV2, "zh"): ("#2a78d6", "--"), (CV3, "en"): ("#1baf7a", "-"), (CV3, "zh"): ("#1baf7a", "--")}
    for (m, s), (c, ls) in C.items():
        w = W[(W.model == m) & (W.set == s)].sort_values("w1")
        if len(w):
            axes[0].plot(w.w1, w.sim_eres2net, ls=ls, color=c, marker="o", ms=5, lw=2, label=f"{m} {s}")
        if G is not None and not G.empty:
            g = G[(G.model == m) & (G.set == s) & (G.measure == "ddcond")].sort_values("w1")
            axes[1].plot(g.w1, g.value, ls=ls, color=c, marker="o", ms=5, lw=2, label=f"{m} {s}")
    if Dd is not None:
        for m, c in ((CV2, "#2a78d6"), (CV3, "#1baf7a")):
            d = Dd[Dd.model == m].groupby("w1").agg(dist=("dist_over_t1", "mean"), g=("g_rms", "mean")).reset_index()
            axes[2].plot(d.w1, d.dist, color=c, marker="o", ms=5, lw=2, label=f"{m} en 50")
            axes[2].plot(d.w1, d.dist.min() + d.g.iloc[0] * np.abs(d.w1), color=c, lw=1, ls=":",
                         label=f"{m}: min + |w₁|·‖g_c‖ (‖g_c‖={d.g.iloc[0]:.2f})")
    for ax, t in zip(axes, ["SIM ERes2Net ×100 (seed 0, 300 utt.)", "8-seed Δd_cond (100 utt.)", "dist(z_t₁, teacher path set) / t₁"]):
        style(ax)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.axvline(0.7, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel("first-step w₁ (second step w = 0.7)", fontsize=9, color=INK2)
        ax.set_title(t, fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "E10_fig14_wsign.png", dpi=160, facecolor="white")
    print(FIG / "E10_fig14_wsign.png")


# ---------------- E10-4 ----------------
FILL = [  # (model, label, nfe, config, calls, baseline config, K)
    (CV3, "R′+R", 3, "fill_cv3_RpR_n3", 5, "e81_cv3_n3_def", "e81_cv3_n3_R"),
    (IDX, "R′+R", 3, "fill_idx_RpR_n3", 5, "e81_idx_n3_def", "e81_idx_n3_R"),
    (CV2, "R′+R", 3, "e82_cv2_speaker_n3_R", 5, "e82_cv2_all_n3_def", "e82_cv2_all_n3_R"),
    (IDX, "mid+R +1 Euler (ours)", 2, "fill_idx_midR1", 5, "e81_idx_n3_def", "e81_idx_n3_R"),
    (CV3, "mid+R +1 Euler (ours)", 2, "b_cv3_midR1", 5, "e81_cv3_n3_def", "e81_cv3_n3_R"),
    (CV2, "mid+R +1 Euler (ours)", 2, "b_cv2_midR1", 5, "e82_cv2_all_n3_def", "e82_cv2_all_n3_R"),
]
SEED3 = [(CV2, "R′ (Sel-CFG input_text)", "x_cv2_Rp_n2", "e82_cv2_all_n2_def"), (CV2, "R′+R", "x_cv2_RpR_n2", "e82_cv2_all_n2_def"),
         (CV3, "R′ (Sel-CFG input_text)", "x_cv3_Rp_n2", "e81_cv3_n2_def"), (CV3, "R′+R", "fill_cv3_RpR_n2", "e81_cv3_n2_def"),
         (IDX, "R′ (Sel-CFG input_text)", "x_idx_Rp_n2", "e81_idx_n2_def"), (IDX, "R′+R", "fill_idx_RpR_n2", "e81_idx_n2_def"),
         (CV2, "R (ours)", "e82_cv2_all_n2_R", "e82_cv2_all_n2_def"), (CV3, "R (ours)", "e81_cv3_n2_R", "e81_cv3_n2_def"),
         (IDX, "R (ours)", "e81_idx_n2_R", "e81_idx_n2_def"), (CV2, "baseline", "e82_cv2_all_n2_def", None),
         (CV3, "baseline", "e81_cv3_n2_def", None), (IDX, "baseline", "e81_idx_n2_def", None)]


def fwdir(name, k, s):
    if name in J.CONFIGS or (name, "k0") in J.ALIAS:
        return J.wav_dir(name, k, s)
    return J.J9.wav_dir(name, k, s) if not name.startswith("e7:") else A.wdir(name, k, s)


def fill():
    rows = []
    for M, label, nfe, cfg, calls, bcfg, rcfg in FILL:
        for s in ("en", "zh"):
            X, B, Rr, T = (load_metrics(fwdir(c, 0, s)) for c in (cfg, bcfg, rcfg, A.TEACHER[M]))
            if X is None:
                continue
            r = dict(kind="seed0", model=M, method=label, nfe=nfe, config=cfg, calls=calls, set=s)
            r.update(compare(X, B, T))
            est, lo, hi, p = stat(X.sim_eres2net - Rr.loc[X.index, "sim_eres2net"])
            r.update(vs_R_d_sim_eres2net=est, vs_R_d_sim_eres2net_lo=lo, vs_R_d_sim_eres2net_hi=hi, vs_R_d_sim_eres2net_p=p,
                     R_sim_eres2net=Rr.sim_eres2net.mean(), R_utmos=Rr.utmos.mean(), utmos_minus_teacher=X.utmos.mean() - T.utmos.mean())
            gap = T.sim_eres2net.mean() - B.sim_eres2net.mean()
            r["recovery"] = (X.sim_eres2net.mean() - B.sim_eres2net.mean()) / gap
            rows.append(r)
    for M, label, cfg, bcfg in SEED3:
        for s in ("en", "zh"):
            per = []
            for k in (0, 1, 2):
                X = load_metrics(fwdir(cfg, k, s))
                B = load_metrics(fwdir(bcfg, k, s)) if bcfg else None
                T = load_metrics(A.wdir(A.TEACHER[M], 0, s))
                if X is None or "wer" not in X:
                    per = None
                    break
                rec = (X.sim_eres2net.mean() - B.sim_eres2net.mean()) / (T.sim_eres2net.mean() - B.sim_eres2net.mean()) if B is not None and "sim_eres2net" in B else np.nan
                per.append(dict(sim=X.sim_eres2net.mean(), wer=X.wer.mean(), utmos=X.utmos.mean(), rec=rec))
            if not per:
                continue
            P_ = pd.DataFrame(per)
            r = dict(kind="3seed", model=M, method=label, nfe=2, config=cfg, set=s)
            for c in P_.columns:
                r[f"{c}_mean"], r[f"{c}_sd"] = P_[c].mean(), P_[c].std(ddof=1)
                for k in range(3):
                    r[f"{c}_k{k}"] = P_[c].iloc[k]
            rows.append(r)
    Fd = pd.DataFrame(rows)
    Fd.to_csv(ROOT / "results/E10_fill.csv", index=False)
    pairs = [(CV3, "R′+R", "fill_cv3_RpR_n2"), (IDX, "R′+R", "fill_idx_RpR_n2"), (CV2, "R′+R", "e82_cv2_speaker_n2_R"),
             (CV2, "R (ours)", "e82_cv2_all_n2_R"), (CV3, "R (ours)", "e81_cv3_n2_R"), (IDX, "R (ours)", "e81_idx_n2_R"),
             (CV2, "R′ (Sel-CFG input_text)", "e82_cv2_speaker_n2_def"), (CV3, "R′ (Sel-CFG input_text)", "B_cv3_Rp_n2"),
             (IDX, "R′ (Sel-CFG input_text)", "B_idx_Rp_n2"), (CV2, "baseline", "e82_cv2_all_n2_def"),
             (CV3, "baseline", "e81_cv3_n2_def"), (IDX, "baseline", "e81_idx_n2_def")]
    G = geometry_on(pairs, {"en": "data/lists/en_all", "zh": "data/lists/zh_all"}, fwdir)
    if not G.empty:
        G.to_csv(ROOT / "results/E10_fill_geometry.csv", index=False)
        sm = G[(G.measure == "ddcond") & (G.label == "R′+R")][["model", "set", "value"]].rename(columns={"value": "ddcond"})
        recs = []
        for _, r in sm.iterrows():
            q = Fd[(Fd.kind == "3seed") & (Fd.model == r.model) & (Fd.method == "R′+R") & (Fd.set == r.set)]
            recs.append(q.rec_k0.iloc[0] if len(q) else np.nan)
        sm["recovery"] = recs
        sm.to_csv(ROOT / "results/E10_fill_geometry_summary.csv", index=False)
    print(Fd.round(3).to_string())
    return Fd, G


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("holm", "all"):
        holm_table7()
    if what in ("fullset", "all"):
        fullset()
    if what in ("wsign", "all"):
        W, G = wsign()
        fig14(W, G)
    if what in ("fill", "all"):
        fill()
    if what in ("fig61", "all"):
        fig61()
