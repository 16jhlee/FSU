"""E14 (v10.6) analysis, one function per block; each writes results/E14_<block>.csv and prints the pre-registered verdicts.
python scripts/e14_analyze.py [glmfull mask10 w0zh onestep seeds3 branches idxcf thru bitcheck ...]   (env_eval; default all)
SIM = ERes2Net x100 (WavLM / ECAPA alongside), WER/CER %, eps = 0.42. Paired over utterances (bootstrap 10k CI, Wilcoxon,
Holm within the block's table), as v10.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e11_jobs as J11  # noqa: E402
import e11d_jobs as J11D  # noqa: E402
import e13_jobs as J13  # noqa: E402
import e14_jobs as J  # noqa: E402

ROOT = J.ROOT
RES = ROOT / "results"
EPS = 0.42
EMBS = ["eres2net", "wavlm", "ecapa"]
SIMS = [f"sim_{e}" for e in EMBS]
OBJ = ["utmos", "utmosv2", "dnsmos_ovrl", "nisqa_mos", "nisqa_tts"]
rng = np.random.default_rng(0)
pd.set_option("display.width", 260)


def boot(x, q=(2.5, 97.5), n=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    b = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(b, q)


def paired(X, Y, col):
    c = X.index.intersection(Y.index)
    d = (X.loc[c, col] - Y.loc[c, col]).dropna()
    m, lo, hi = boot(d)
    return dict(d=m, lo=lo, hi=hi, p=(wilcoxon(d).pvalue if np.any(d != 0) else 1.0), n=len(d))


def M(wd, name, k, s):
    return EA.load_metrics(wd(name, k, s))


def with_objq(X, d):
    """join objective-quality predictors (objq_metrics.py: _metrics/objq.csv, objq_dnsmos.csv) when present"""
    if X is None:
        return X
    for f in ("objq.csv", "objq_dnsmos.csv"):
        p = Path(d) / "_metrics" / f
        if p.exists():
            Q = pd.read_csv(p).set_index("utt_id")
            X = X.join(Q[[c for c in Q.columns if c not in X.columns]], how="left")
    return X


# ---------------- E14-1 ----------------
def glmfull():
    srcs = {  # cond -> (wav_dir fn, config, set map)
        "teacher": (J11D.wav_dir, "d_teacher", {"en": "en", "zh": "zh", "hard": "hard"}),
        "allnull_n2": (None, None, None), "allnullR_n2": (None, None, None),
        "allnull_midR": (J.wav_dir, "allnull_midR", {"en": "en_full", "zh": "zh_full", "hard": "hard"}),
        "base_n2": (J11D.wav_dir, "d_base_n2", {"en": "en", "zh": "zh", "hard": "hard"}),
        "midR": (J11D.wav_dir, "d_midR", {"en": "en", "zh": "zh", "hard": "hard"}),
    }

    def load(cond, s):
        if cond in ("allnull_n2", "allnullR_n2"):
            wd, cfg, ss = (J.wav_dir, cond, "hard") if s == "hard" else (J11D.wav_dir, f"d_{cond}", s)
        else:
            wd, cfg, m = srcs[cond]
            ss = m[s]
        return with_objq(M(wd, cfg, 0, ss), wd(cfg, 0, ss))
    rows, sig = [], []
    for s in ("en", "zh", "hard"):
        D = {c: load(c, s) for c in srcs}
        T, A = D["teacher"], D["allnull_n2"]
        for c in ("allnull_n2", "allnullR_n2", "allnull_midR", "teacher"):
            X = D[c]
            if X is None:
                continue
            r = dict(set=s, cond=c, n=len(X), **{m: X[m].mean() for m in SIMS + ["wer"] + OBJ if m in X})
            if c != "teacher":
                r["loss_vs_teacher"] = T.sim_eres2net.mean() - X.sim_eres2net.mean()
                for m in SIMS + ["wer"] + OBJ:
                    if m in X and m in T:
                        p = paired(X, T, m)
                        r.update({f"vs_teacher_d_{m}": p["d"], f"vs_teacher_d_{m}_lo": p["lo"], f"vs_teacher_d_{m}_hi": p["hi"]})
            if c in ("allnullR_n2", "allnull_midR") and A is not None:
                gap = T.sim_eres2net.mean() - A.sim_eres2net.mean()
                r["recovery_vs_allnull"] = (X.sim_eres2net.mean() - A.sim_eres2net.mean()) / gap
                for m in SIMS + ["wer"]:
                    p = paired(X, A, m)
                    r.update({f"vs_allnull_d_{m}": p["d"], f"vs_allnull_d_{m}_lo": p["lo"], f"vs_allnull_d_{m}_hi": p["hi"],
                              f"vs_allnull_d_{m}_p": p["p"]})
                    sig.append((len(rows), f"vs_allnull_d_{m}_p"))
                c_ = X.index.intersection(A.index)
                d = (X.loc[c_, "wer"] - A.loc[c_, "wer"]).dropna()
                est, lo, hi = boot(d, (5, 95))
                r.update(wer_tost_ci90_lo=lo, wer_tost_ci90_hi=hi, wer_equivalent=bool(lo > -0.3 and hi < 0.3),
                         wer_noninferior=bool(hi < 0.3))
            rows.append(r)
    O = pd.DataFrame(rows)
    for col in {c for _, c in sig}:  # Holm within the table (Table VII unit), per metric
        O[col + "_holm"] = EA.holm(O[col].values) if O[col].notna().any() else np.nan
    # all-null teacher (E13-5, en/zh 300 subset) as the secondary reference
    aux = []
    for s in ("en", "zh"):
        Ta = M(J13.wav_dir, "ga_T", 0, s)
        X = load("allnull_midR", s)
        if Ta is not None and X is not None:
            c = Ta.index.intersection(X.index)
            aux.append(dict(set=s, n=len(c), sim_allnull_teacher=Ta.loc[c, "sim_eres2net"].mean(),
                            sim_midR_on_subset=X.loc[c, "sim_eres2net"].mean()))
    O.to_csv(RES / "E14_glm_allnull_full.csv", index=False)
    pd.DataFrame(aux).to_csv(RES / "E14_glm_allnull_full_auxteacher.csv", index=False)
    print(O[[c for c in ["set", "cond", "n", "sim_eres2net", "loss_vs_teacher", "recovery_vs_allnull", "wer", "vs_allnull_d_wer",
                         "vs_allnull_d_wer_p_holm", "wer_noninferior"] + OBJ if c in O]].round(3).to_string(index=False))
    h = O[(O.set == "hard")].set_index("cond")
    if "allnull_n2" in h.index:
        print("G1 hard all-null loss > 2.0:", round(h.loc["allnull_n2", "loss_vs_teacher"], 2))
    if "allnullR_n2" in h.index:
        print("G2 hard recovery >= 70%:", round(h.loc["allnullR_n2", "recovery_vs_allnull"] * 100, 1))
    mr = O[O.cond == "allnull_midR"].set_index("set")
    if len(mr):
        print("G3 midR >= teacher - eps:", (-mr.loss_vs_teacher >= -EPS).to_dict())
        if "vs_allnull_d_wer_p_holm" in mr:
            print("G4 WER not significantly worse:", ((mr.vs_allnull_d_wer <= 0) | (mr.vs_allnull_d_wer_p_holm >= 0.05)).to_dict())
        q = [f"vs_teacher_d_{m}" for m in OBJ if f"vs_teacher_d_{m}" in mr]
        print("G5 |objq - teacher| <= 0.15:", mr[q].abs().le(0.15).to_dict())
    print(pd.DataFrame(aux).round(3).to_string(index=False))


# ---------------- E14-2 ----------------
GA = {"11": ("ga_11", J13.wav_dir), "01": ("ga_01", J13.wav_dir), "10": ("ga_10", J.wav_dir), "00": ("ga_00", J13.wav_dir),
      "h": ("ga_h", J13.wav_dir)}


def cos100(a, b):
    return 100 * (a @ b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b))


def mask10():
    sim_rows, pairs, geo = [], [], []
    for s, lst in (("en", "data/lists/en_100"), ("zh", "data/lists/zh_100")):
        meta = {l.split("|")[0]: l.split("|")[2] for l in (ROOT / (lst + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()}
        utts = list(meta)
        for e in EMBS:
            prom = dict(np.load(ROOT / "data/cache" / f"prompt_{e}.npz"))
            S, X = {}, {}
            for c, (cfg, wd) in GA.items():
                X[c] = EA.emb_stack(cfg, s, e, utts, wd)
                S[c] = np.stack([cos100(X[c][u], prom[EA.A.pkey(meta[u])]) for u in utts])
                for i, u in enumerate(utts):
                    for k in range(8):
                        sim_rows.append(dict(cond=c, set=s, emb=e, utt=u, seed=k, sim=S[c][i, k]))
            for a, b in (("10", "00"), ("10", "11"), ("01", "10"), ("00", "11"), ("01", "11")):
                dd = S[a] - S[b]
                m_ = dd.mean(1)
                bb = rng.choice(m_, (10000, len(m_))).mean(1)
                ps = dd.mean(0)
                pairs.append(dict(set=s, emb=e, comparison=f"{a} - {b}", d=m_.mean(), lo=np.percentile(bb, 2.5),
                                  hi=np.percentile(bb, 97.5), seeds_negative=int((ps < 0).sum()), seeds_positive=int((ps > 0).sum()),
                                  seed_min=ps.min(), seed_max=ps.max()))
            # centre geometry vs the all-null teacher ga_T (E13-1 / E13-1b definitions)
            T = EA.emb_stack("ga_T", s, e, utts, J13.wav_dir)
            sh = {}
            for c in ("11", "10"):
                dc, sh[c] = [], {}
                for u in utts:
                    Tu, Xu = EA.A.unit(T[u]), EA.A.unit(X[c][u])
                    mt, mf = Tu.mean(0), Xu.mean(0)
                    st2 = np.mean(np.sum((Tu - mt) ** 2, 1))
                    sf2 = np.mean(np.sum((Xu - mf) ** 2, 1))
                    d2 = np.sum((mf - mt) ** 2) / st2
                    dc.append(np.sqrt(max(0.0, d2 - (sf2 + st2) / (8 * st2))))
                    sh[c][u] = mf - mt
                m_, lo, hi = boot(dc)
                geo.append(dict(set=s, emb=e, cond=c, measure="delta_corr", value=m_, lo=lo, hi=hi))
            cs = [float(sh["10"][u] @ sh["11"][u] / np.linalg.norm(sh["10"][u]) / np.linalg.norm(sh["11"][u])) for u in utts]
            m_, lo, hi = boot(cs)
            geo.append(dict(set=s, emb=e, cond="10", measure="cos_to_base_shift", value=m_, lo=lo, hi=hi))
            print("mask10", s, e, flush=True)
    D = pd.DataFrame(sim_rows)
    D.to_csv(RES / "E14_glm_mask10_seeds.csv", index=False)
    P, G = pd.DataFrame(pairs), pd.DataFrame(geo)
    Sm = D.groupby(["cond", "set", "emb", "seed"]).sim.mean().groupby(["cond", "set", "emb"]).agg(["mean", "std"]).reset_index()
    Sm.to_csv(RES / "E14_glm_mask10_summary.csv", index=False)
    P.to_csv(RES / "E14_glm_mask10.csv", index=False)
    G.to_csv(RES / "E14_glm_mask10_geometry.csv", index=False)
    print(Sm[Sm.emb == "eres2net"].round(2).to_string(index=False))
    print(P.round(3).to_string(index=False))
    print(G.round(3).to_string(index=False))
    for s in ("en", "zh"):
        p = P[(P.set == s) & (P.emb == "eres2net")].set_index("comparison")
        g = G[(G.set == s) & (G.emb == "eres2net")]
        dc = g[g.measure == "delta_corr"].set_index("cond").value
        print(s, "M1 10<00 seeds:", p.loc["10 - 00", "seeds_negative"], "| M2 10<11 seeds:", p.loc["10 - 11", "seeds_negative"],
              "| M3 delta_corr 10 >= 11:", round(dc["10"], 3), ">=", round(dc["11"], 3),
              "| M4 cos:", round(g[g.measure == "cos_to_base_shift"].value.iloc[0], 3))


# ---------------- E14-4 / E14-5 ----------------
def w0zh():
    rows = []
    for n in (2, 3, 4, 8, 10):
        name, wd = (f"g_w0zh_n{n}", J.wav_dir) if n != 10 else ("g_n10_w0", J.wav_dir)
        X = M(wd, name, 0, "zh")
        if X is not None:
            rows.append(dict(nfe=n, n=len(X), **{m: X[m].mean() for m in SIMS + ["wer"] if m in X}))
    O = pd.DataFrame(rows)
    E = pd.read_csv(ROOT / "results/E11_w0.csv")
    O.to_csv(RES / "E14_glm_w0curve_zh.csv", index=False)
    print(O.round(3).to_string(index=False))
    if len(O) == 5:
        s = O.set_index("nfe").sim_eres2net
        d = s[10] - s[2]
        print(f"W1 SIM(10) - SIM(2) = {d:+.2f} in [0, 1.2]: {0 <= d <= 1.2}; en (E11) {E.sim.iloc[-1] - E.sim.iloc[0]:+.2f}")
        print(f"W2 range over N>=4 = {s[[4, 8, 10]].max() - s[[4, 8, 10]].min():.2f} <= eps; monotone: {bool(np.all(np.diff(s.values) >= 0))}")


def onestep():
    rows = []
    for s in ("en", "zh"):
        X, B, T = M(J.wav_dir, "g_w0_n1", 0, s), M(J11.wav_dir, "g_base_n1", 0, s), M(J11.wav_dir, "g_base_n10", 0, s)
        if X is None:
            continue
        p = paired(X, B, "sim_eres2net")
        gap = T.sim_eres2net.mean() - B.sim_eres2net.mean()
        rows.append(dict(set=s, sim_w0_n1=X.sim_eres2net.mean(), sim_w07_n1=B.sim_eres2net.mean(), sim_teacher=T.sim_eres2net.mean(),
                         d=p["d"], lo=p["lo"], hi=p["hi"], p=p["p"], recovery=(X.sim_eres2net.mean() - B.sim_eres2net.mean()) / gap,
                         wer_w0_n1=X.wer.mean(), wer_w07_n1=B.wer.mean(), utmos_w0_n1=X.utmos.mean(), utmos_w07_n1=B.utmos.mean(),
                         **{f"{m}_w0_n1": X[m].mean() for m in SIMS[1:]}))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "E14_glm_onestep.csv", index=False)
    print(O.round(3).to_string(index=False))
    if len(O):
        print("O1 SIM(w0) > SIM(w0.7):", (O.d > 0).tolist(), "| O2 recovery in [0.20, 0.65]:", O.recovery.between(0.2, 0.65).tolist())


# ---------------- E14-10 ----------------
def seeds3():
    rows = []
    for key in ("base_n2", "R_n2", "midR"):
        for s in ("en", "zh"):
            for k in (0, 1, 2):
                X = M(J11.wav_dir, f"g_{key}", 0, s) if k == 0 else M(J.wav_dir, f"gs3_{key}", k, s)
                if X is not None:
                    rows.append(dict(cond=key, set=s, seed=k, n=len(X), **{m: X[m].mean() for m in SIMS + ["wer", "utmos"] if m in X}))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "E14_glm_seeds.csv", index=False)
    if not len(O):
        return
    S = O.groupby(["cond", "set"])[SIMS + ["wer", "utmos"]].agg(["mean", "std"])
    print(O.round(3).to_string(index=False))
    print(S.round(3).to_string())
    W = O.pivot_table(index=["set", "seed"], columns="cond", values="sim_eres2net")
    print("S1 max seed SD:", round(O.groupby(["cond", "set"]).sim_eres2net.std().max(), 3),
          "| S2 R > base all seeds:", bool((W.R_n2 > W.base_n2).all()), "| midR > R all seeds:", bool((W.midR > W.R_n2).all()))
    S.to_csv(RES / "E14_glm_seeds_summary.csv")


# ---------------- E14-6 ----------------
def branches():
    rows = []
    E10 = {"F5-TTS": "E10_rho_f5", "ZipVoice": "E10_rho_zip"}
    for model, key in (("F5-TTS", "f5"), ("ZipVoice", "zip")):
        p = RES / f"E14_textcond_branches_{key}.csv"
        if not p.exists():
            continue
        D = pd.read_csv(p)
        ref = pd.read_csv(RES / f"{E10[model]}.csv").g_rms.mean()
        g = {}
        for b, x in D.groupby("branch"):
            u = x.groupby("utt").g_rms.mean()
            m_, lo, hi = boot(u)
            g[b] = m_
            rows.append(dict(model=model, branch=b, value=m_, lo=lo, hi=hi, n_utt=len(u), e10_7_official=ref))
        rows.append(dict(model=model, branch="audionull/allnull", value=g["audionull"] / g["allnull"]))
        rows.append(dict(model=model, branch="textnull/allnull", value=g["textnull"] / g["allnull"]))
        print(model, {k: round(v, 3) for k, v in g.items()}, "E10-7 official-branch g_rms", round(ref, 3))
        print("  B1 allnull >= textnull > audionull:", g["allnull"] >= g["textnull"] > g["audionull"],
              "| B2 audionull/allnull <= 0.6:", round(g["audionull"] / g["allnull"], 3))
        if model == "ZipVoice":
            print("  B3 (as registered) 2 x textnull within 10% of 0.85:", round(2 * g["textnull"], 3), abs(2 * g["textnull"] / 0.85 - 1) <= 0.1,
                  "| pure textnull vs E10-7 (same quantity):", round(g["textnull"], 3), "vs", round(ref, 3))
        else:
            print("  allnull vs E10-7 (same quantity):", round(g["allnull"], 3), "vs", round(ref, 3))
    pd.DataFrame(rows).to_csv(RES / "E14_textcond_branches.csv", index=False)


# ---------------- E14-7 ----------------
def idxcf():
    p = RES / "E14_idx_closedform.csv"
    if not p.exists():
        return
    D = pd.read_csv(p)
    rows = []
    for sd, g in D.groupby("seeds"):
        rows.append(dict(seeds=sd, n=len(g), cos_mean=g.cos_slope_gest.mean(), cos_min=g.cos_slope_gest.min(),
                         rms_slope=g.rms_slope.mean(), rms_gest=g.rms_gest.mean(), lin_resid_max=g.lin_resid.max(),
                         rel_diff=g.rms_slope.mean() / g.rms_gest.mean() - 1))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "E14_idx_closedform_summary.csv", index=False)
    print(O.round(4).to_string(index=False))
    o = O.set_index("seeds").loc["shared"]
    print("A1 cos mean >= 0.9:", round(o.cos_mean, 3), "| A2 |slope - g_est| <= 10%:", round(o.rel_diff * 100, 1), "%")


# ---------------- E14-9 ----------------
def thru():
    p = RES / "E14_midR_throughput.csv"
    if not p.exists():
        return
    D = pd.read_csv(p)
    W = D.pivot_table(index="concurrency", columns="cond", values="throughput")
    W["midR_over_R"] = W["mid+R (ours)"] / W["R (ours) NFE2"]
    E = pd.read_csv(RES / "E10_concurrency.csv")
    W["R_E10_6"] = E[E.cond == "R (ours) NFE2"].set_index("concurrency").throughput
    W["midR_over_R_E10_6"] = W["mid+R (ours)"] / W["R_E10_6"]
    print(W.round(3).to_string())
    print("L1 c=1 within 5%:", abs(W.loc[1, "midR_over_R"] - 1) <= 0.05, "| L2 c=4, 16:", (abs(W.loc[[4, 16], "midR_over_R"] - 1) <= 0.05).tolist())
    W.to_csv(RES / "E14_midR_throughput_summary.csv")


# ---------------- determinism checks ----------------
def bitcheck():
    rows = []
    pairs = [(J.wav_dir("gachk_11", 1, "en100"), J13.wav_dir("ga_11", 1, "en100")),
             (J.wav_dir("gachk_00", 1, "en100"), J13.wav_dir("ga_00", 1, "en100")),
             (J.wav_dir("gs3chk_base_n2", 1, "en100"), J11.wav_dir("g_base_n2", 1, "en100"))]
    for key in ("base_n2", "R_n2", "midR"):  # seed-1 300 runs vs the E11 seed-1 en100 folders on their 100 utterances
        pairs.append((J.wav_dir(f"gs3_{key}", 1, "en"), J11.wav_dir(f"g_{key}", 1, "en100")))
    for a, b in pairs:
        if not (Path(a) / "_done").exists():
            continue
        names = sorted(f.name for f in Path(b).glob("*.wav"))
        # compare the data chunk only: libsndfile's float-WAV PEAK chunk carries a write timestamp (header bytes differ)
        def data(p):
            x = p.read_bytes()
            return x[x.index(b"data"):]
        same = sum(data(Path(a) / n) == data(Path(b) / n) for n in names if (Path(a) / n).exists())
        rows.append(dict(new=str(a), ref=str(b), n_ref=len(names), n_identical=same))
    O = pd.DataFrame(rows)
    O.to_csv(RES / "E14_bitcheck.csv", index=False)
    print(O.to_string(index=False))


BLOCKS = dict(glmfull=glmfull, mask10=mask10, w0zh=w0zh, onestep=onestep, seeds3=seeds3, branches=branches, idxcf=idxcf, thru=thru,
              bitcheck=bitcheck)

if __name__ == "__main__":
    for b in (sys.argv[1:] or BLOCKS):
        print(f"===== {b}")
        BLOCKS[b]()
