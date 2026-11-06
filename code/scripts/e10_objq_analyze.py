"""E10-1′ analysis: objective quality next to UTMOS.   python scripts/e10_objq_analyze.py   (env_eval)
Predictors (columns): utmos (UTMOS22, existing), utmosv2, dnsmos_ovrl / dnsmos_sig / dnsmos_bak (P.835), nisqa_mos, nisqa_tts.
Outputs
  results/E10_objq_eps.csv   resolution ε per predictor: same condition re-synthesised with 8 noise seeds (teacher, 3 models,
                             en/zh). eps_utt = mean over utterances of the per-utterance SD across seeds (utterance-level,
                             like the embedder ε); eps_sys = SD across seeds of the 300-utterance mean.
  results/E10_objq.csv       every re-scored folder (table tag, config, set): mean of each predictor
  results/E10_objq_table1.csv  Table 1 rows (subset): means + paired Δ vs teacher and vs baseline (bootstrap CI, Wilcoxon, Holm)
  reports/E10_objq.md        tables + pre-registered Q1-Q3
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import e8p9_jobs as J9  # noqa: E402
import e10_objq_jobs as O  # noqa: E402
from e9_tables import T1  # noqa: E402

ROOT = J9.ROOT
PRED = ["utmos", "utmosv2", "dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "nisqa_mos", "nisqa_tts"]
MAIN3 = ["utmosv2", "dnsmos_ovrl", "nisqa_mos"]  # Q2 / Q3: the three new predictors
rng = np.random.default_rng(0)


def load(d):
    d = Path(d) / "_metrics"
    parts = []
    for f in ("metrics.csv", "objq.csv", "objq_dnsmos.csv"):
        if (d / f).exists():
            x = pd.read_csv(d / f).set_index("utt_id")
            parts.append(x[[c for c in x.columns if c in PRED]])
    if not parts:
        return None
    D = pd.concat(parts, axis=1)
    return D.loc[:, ~D.columns.duplicated()]


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan,) * 4
    bs = rng.choice(x, (5000, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5]), (wilcoxon(x).pvalue if np.any(x != 0) else 1.0)


def holm(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - r) * p[i]))
        out[i] = run
    return out


def eps():
    rows = []
    for c, model in zip(O.TEACHERS, ["CosyVoice2", "CosyVoice3", "IndexTTS-2"]):
        for s in ("en", "zh"):
            Ds = [load(J9.wav_dir(c, k, s)) for k in range(8)]
            if any(D is None for D in Ds):
                continue
            for p in PRED:
                if not all(p in D for D in Ds):
                    continue
                u = Ds[0].index
                M = np.stack([D.loc[u, p].values for D in Ds])  # 8 x n
                rows.append(dict(model=model, set=s, predictor=p, n_utt=M.shape[1], eps_utt=M.std(0, ddof=1).mean(),
                                 eps_sys=M.mean(1).std(ddof=1), mean=M.mean()))
    E = pd.DataFrame(rows)
    E.to_csv(ROOT / "results/E10_objq_eps.csv", index=False)
    return E


def folders():
    rows = []
    for m, d, t in O.jobs():
        D = load(d)
        if D is None:
            continue
        r = dict(tag=t, folder=str(Path(d).relative_to(ROOT)) if str(d).startswith(str(ROOT)) else d, n=len(D))
        for p in PRED:
            if p in D:
                r[p] = D[p].mean()
        rows.append(r)
    F = pd.DataFrame(rows)
    F.to_csv(ROOT / "results/E10_objq.csv", index=False)
    return F


def table1():
    rows = []
    for model, d in T1.items():
        tcfg = d["teacher"][0]
        for s in ("en", "zh"):
            T = load(J9.wav_dir(tcfg, 0, s))
            for n, items in d["rows"].items():
                B = load(J9.wav_dir(items[0][1], 0, s))
                for meth, cfg, calls in items + [("teacher", tcfg, d["teacher"][1])]:
                    X = load(J9.wav_dir(cfg, 0, s))
                    if X is None:
                        continue
                    r = dict(model=model, NFE=n if meth != "teacher" else "teacher", method=meth.replace("*", ""), calls=calls,
                             set=s, n=len(X))
                    for p in PRED:
                        if p not in X:
                            continue
                        r[p] = X[p].mean()
                        for ref, tag in ((T, "vsT"), (B, "vsB")):
                            if ref is None or p not in ref or (tag == "vsT" and meth == "teacher") or (tag == "vsB" and cfg == items[0][1]):
                                continue
                            c = X.index.intersection(ref.index)
                            est, lo, hi, pv = stat(X.loc[c, p] - ref.loc[c, p])
                            r.update({f"{tag}_{p}": est, f"{tag}_{p}_lo": lo, f"{tag}_{p}_hi": hi, f"{tag}_{p}_p": pv})
                    rows.append(r)
                    if meth == "teacher":
                        break
            # teacher row once per model/set
    D = pd.DataFrame(rows).drop_duplicates(subset=["model", "NFE", "method", "set"])
    for c in [c for c in D.columns if c.endswith("_p")]:
        D[c + "_holm"] = holm(D[c].values)
    D.to_csv(ROOT / "results/E10_objq_table1.csv", index=False)
    return D


def table7():
    """Table 7′ (E8′-7 + E8′-10): per (model, nfe, method, set) means of every predictor + paired Δ vs baseline and vs
    R (ours) at the same NFE (Holm over the table)."""
    import e8p7_jobs as J7
    import e8p10_jobs as J10
    T = pd.concat([pd.read_csv(ROOT / "results/E8p7_table7.csv").assign(src="E8′-7", res=lambda x: "7"),
                   pd.read_csv(ROOT / "results/E8p10_table.csv").assign(src="E8′-10", res=lambda x: "10")])
    T = T.drop_duplicates(subset=["model", "nfe", "method", "set"], keep="first")
    wd = lambda r: (J7.wav_dir if r.res == "7" else J10.wav_dir)(r.config, 0, r.set)  # noqa: E731
    cache = {}
    for _, r in T.iterrows():
        cache[(r.model, r.nfe, r.method, r.set)] = load(wd(r))
    rows = []
    for (model, nfe, method, s), X in cache.items():
        if X is None:
            continue
        r = dict(model=model, NFE=nfe, method=method, set=s, n=len(X))
        for ref_m, tag in (("baseline", "vsB"), ("R (ours)", "vsR")):
            ref = cache.get((model, nfe, ref_m, s))
            for p in PRED:
                if p not in X:
                    continue
                r[p] = X[p].mean()
                if ref is None or p not in ref or method == ref_m:
                    continue
                c = X.index.intersection(ref.index)
                est, lo, hi, pv = stat(X.loc[c, p] - ref.loc[c, p])
                r.update({f"{tag}_{p}": est, f"{tag}_{p}_p": pv})
        rows.append(r)
    D = pd.DataFrame(rows)
    for c in [c for c in D.columns if c.endswith("_p")]:
        D[c + "_holm"] = holm(D[c].values)
    D.to_csv(ROOT / "results/E10_objq_table7.csv", index=False)
    return D


def other_tables():
    """Table 9 / v8 folders: mean of every predictor per folder (tags T9, v8seed, v8stream, v8robust) — joined to the
    existing per-folder rows of those tables by folder path."""
    F = pd.read_csv(ROOT / "results/E10_objq.csv")
    F[F.tag.isin(["T9", "v8seed", "v8stream", "v8robust"])].to_csv(ROOT / "results/E10_objq_t9_v8.csv", index=False)
    return F


def verdicts(D, E):
    out = []
    epsm = E.groupby(["model", "predictor"]).eps_utt.mean() if not E.empty else None
    for model in ("CosyVoice2", "CosyVoice3", "IndexTTS-2"):
        for s in ("en", "zh"):
            g = D[(D.model == model) & (D.set == s)]
            R = g[(g.method == "R (ours)") & (g.NFE.astype(str) == "2")]
            M = g[g.method == "mid+R (ours)"]
            if R.empty:
                continue
            R = R.iloc[0]
            ut = R.get("vsT_utmos", np.nan)
            q1 = []
            for p in MAIN3:
                v = R.get(f"vsT_{p}", np.nan)
                e = epsm.get((model, p), np.nan) if epsm is not None else np.nan
                q1.append(f"{p} {v:+.3f} (ε {e:.3f}) {'✓' if abs(v) <= e else '✗'}{'' if np.sign(v) == np.sign(ut) else ' (sign ≠ UTMOS)'}")
            q2 = []
            if len(M):
                M = M.iloc[0]
                q2 = [p for p in MAIN3 if M.get(f"vsT_{p}", 0) < 0 and M.get(f"vsT_{p}_p_holm", 1) < 0.05]
            q3 = [p for p in MAIN3 if R.get(f"vsB_{p}", 0) > 0 and R.get(f"vsB_{p}_p_holm", 1) < 0.05]
            out.append(dict(model=model, set=s, UTMOS_R_minus_T=ut, Q1=" ; ".join(q1),
                            Q2_sig_negative=",".join(q2), Q2=("supported" if len(q2) >= 2 else "refuted") if len(M) else "",
                            Q3_sig_positive=",".join(q3), Q3="supported" if len(q3) == 3 else "refuted"))
    return pd.DataFrame(out)


def main():
    E = eps()
    F = folders()
    D = table1()
    T7 = table7()
    other_tables()
    print(T7[["model", "NFE", "method", "set"] + [c for c in ("utmos", "utmosv2", "dnsmos_ovrl", "nisqa_mos") if c in T7]].round(3).to_string())
    V = verdicts(D, E)
    V.to_csv(ROOT / "results/E10_objq_verdicts.csv", index=False)
    pd.set_option("display.width", 250)
    print(E.round(3).to_string())
    print(D[["model", "NFE", "method", "set"] + [p for p in PRED if p in D]].round(3).to_string())
    print(V.to_string())
    print(F.groupby("tag").size())


if __name__ == "__main__":
    main()
