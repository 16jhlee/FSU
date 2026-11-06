"""E5 analysis (v3 §3.5).

1. Seed-0 table (all configs, en/zh): WER, SIM-o (WavLM/ECAPA/ERes2Net), SIM-r (family vocoder resynthesised prompt),
   UTMOS, spectral (hf_ratio, flux_var), NFE / network calls; paired deltas vs the Euler baseline of the same
   total NFE and vs the teacher reference; recovery = (SIM_alg - SIM_euler) / (SIM_teacher - SIM_euler).
2. Geometry (8-seed configs): E3 measures vs the teacher reference cloud (ERes2Net-en primary), next to the same
   measures for the Euler baseline, and their paired difference (alg - euler).
Outputs results/E5_seed0_long.csv, E5_seed0_summary.csv, E5_geometry.csv, E5_geometry_summary.csv.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from e5_jobs import BASE, CONFIGS, ROOT, SETS, nfe_calls, wav_dir  # noqa: E402

EMBS = ["wavlm", "ecapa", "eres2net"]
TEACHER = {"A": "cv2_euler_n10", "B": "zip_euler_n16"}
BASELINE = {"cv2_a1_n3": "cv2_euler_n3", "cv2_a1_n4": "cv2_euler_n4", "cv2_a1apg_n3": "cv2_euler_n3",
            "cv2_a1apg_n4": "cv2_euler_n4", "cv2_a1_early": "cv2_euler_n4", "cv2_rs07": "cv2_euler_n3",
            "cv2_rs05": "cv2_euler_n3", "zip_a1_n4": "zip_euler_n4", "zip_a1_early": "zip_euler_n4",
            "zipd_a1_n3": "zipd_euler_n3", "zipd_a1_n4": "zipd_euler_n4", "zipd_rs07": "zipd_euler_n3",
            "zipd_rs05": "zipd_euler_n3", "zipd_euler_n3": "zipd_euler_n4", "cv2_euler_n4": "cv2_euler_n3"}
for n in [2, 10]:
    for w in ["0", "0.3", "1.0", "1.5"]:
        BASELINE[f"cv2_cfg{w}_n{n}"] = f"cv2_euler_n{n}"
for n in [2, 3, 10]:
    BASELINE[f"cv2_apg_n{n}"] = BASELINE[f"cv2_iv_n{n}"] = f"cv2_euler_n{n}"
for s in ["early", "uniform", "late"]:
    BASELINE[f"cv2_euler_{s}"], BASELINE[f"zip_euler_{s}"] = "cv2_euler_n4", "zip_euler_n4"
    if s != "early":
        BASELINE[f"cv2_a1_{s}"], BASELINE[f"zip_a1_{s}"] = "cv2_a1_n4", "zip_a1_n4"
for kk in [1, 3, 5, 8]:
    BASELINE[f"cv2_hybrid_k{kk}"] = "cv2_euler_n10"
rng = np.random.default_rng(0)


def fam(name):
    return "A" if name.startswith("cv2") else "B"


def pkey(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def stat(x, log=False):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (10000, len(x))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    if log:
        return dict(est=np.exp(x.mean()), lo=np.exp(lo), hi=np.exp(hi), p=p, n=len(x))
    return dict(est=x.mean(), lo=lo, hi=hi, p=p, n=len(x))


META = {s: {l.split("|")[0]: l.split("|")[2] for l in (ROOT / (m + ".meta.lst")).read_text(encoding="utf-8").splitlines()
            if l.strip()} for s, m, _ in SETS}
RES = {(v, e): dict(np.load(ROOT / "data/cache" / f"prompt_resynth_{v}_{e}.npz")) for v in ["hift", "vocos"] for e in EMBS}
PROM = {e: dict(np.load(ROOT / "data/cache" / f"prompt_{e}.npz")) for e in EMBS}


def load_seed0(name, s):
    d = wav_dir(name, 0, s)
    f = d / "_metrics" / "metrics.csv"
    if not f.exists() or not ((d / "_e5done").exists() or name in BASE):
        return None
    df = pd.read_csv(f)
    ac = d / "_metrics" / "acoustic.csv"
    if ac.exists():
        df = df.merge(pd.read_csv(ac)[["utt_id", "hf_ratio", "flux_var"]], on="utt_id", how="left")
    voc = "hift" if fam(name) == "A" else "vocos"
    for e in EMBS:
        g = np.load(d / "_metrics" / f"emb_{e}.npz")
        df[f"simr_{e}"] = [float(unit(g[f"gen/{u}"]) @ unit(RES[(voc, e)][pkey(META[s][u])])) for u in df.utt_id]
    df["wer"] *= 100
    return df.set_index("utt_id")


def seed0():
    names = list(BASE) + list(CONFIGS)
    cache = {(n, s): load_seed0(n, s) for n in names for s, _, _ in SETS}
    mets = ["wer"] + [f"sim_{e}" for e in EMBS] + [f"simr_{e}" for e in EMBS] + ["utmos", "hf_ratio", "flux_var"]
    rows, long = [], []
    for n in names:
        for s, _, _ in SETS:
            D = cache[(n, s)]
            if D is None:
                continue
            long.append(D.assign(config=n, set=s).reset_index())
            nf, calls = nfe_calls(n)
            r = dict(config=n, set=s, nfe=nf, calls=calls, n=len(D), **{m: D[m].mean() for m in mets if m in D})
            for tag, ref in [("vsEuler", BASELINE.get(n)), ("vsTeacher", TEACHER[fam(n)])]:
                R = cache.get((ref, s)) if ref else None
                if R is None or ref == n:
                    continue
                c = D.index.intersection(R.index)
                for m in mets:
                    if m in D and m in R:
                        st = stat(D.loc[c, m] - R.loc[c, m])
                        r[f"{tag}_{m}"], r[f"{tag}_{m}_lo"], r[f"{tag}_{m}_hi"], r[f"{tag}_{m}_p"] = \
                            st["est"], st["lo"], st["hi"], st["p"]
            b, t = BASELINE.get(n), TEACHER[fam(n)]
            if b and cache.get((b, s)) is not None and cache.get((t, s)) is not None:
                for e in EMBS:
                    gap = cache[(t, s)][f"sim_{e}"].mean() - cache[(b, s)][f"sim_{e}"].mean()
                    r[f"recovery_{e}"] = (D[f"sim_{e}"].mean() - cache[(b, s)][f"sim_{e}"].mean()) / gap \
                        if abs(gap) > 1e-9 else np.nan
            rows.append(r)
    pd.concat(long).to_csv(ROOT / "results/E5_seed0_long.csv", index=False)
    S = pd.DataFrame(rows)
    S.to_csv(ROOT / "results/E5_seed0_summary.csv", index=False)
    return S


def cvar(U):
    return len(U) / (len(U) - 1) * np.mean(np.sum((U - U.mean(0)) ** 2, axis=1))


def cosd(a, b):
    return 1 - unit(a) @ unit(b)


def geometry():
    eight = [n for n, c in CONFIGS.items() if len(c["K"]) == 8]
    pairs = [(n, BASELINE[n], TEACHER[fam(n)]) for n in eight if n in BASELINE]
    rows = []
    need = {x for p in pairs for x in p}

    def emb(name, s, e):
        ok = all(((wav_dir(name, k, s) / "_e5done").exists() or name in BASE) for k in range(8))
        if not ok:
            return None
        arr = [np.load(wav_dir(name, k, s) / "_metrics" / f"emb_{e}.npz") for k in range(8)]
        return {u: np.stack([a[f"gen/{u}"] for a in arr]).astype(np.float64) for u in META[s]}

    for s, _, _ in SETS:
        for e in EMBS:
            E = {n: emb(n, s, e) for n in need}
            for alg, base, ref in pairs:
                if E.get(alg) is None or E.get(base) is None or E.get(ref) is None:
                    continue
                R = E[ref]
                g = unit(unit(np.concatenate(list(R.values()))).mean(0))
                for u in R:
                    UR = unit(R[u])
                    c = unit(UR.mean(0))
                    loo = np.mean([cosd(UR[k], unit(np.delete(UR, k, 0).mean(0))) for k in range(8)])
                    ep = PROM[e][pkey(META[s][u])]
                    rec = dict(alg=alg, base=base, ref=ref, set=s, emb=e, utt=u)
                    for tag, X in [("alg", E[alg][u]), ("base", E[base][u])]:
                        U = unit(X)
                        rec[f"{tag}_lvr"] = np.log(cvar(U) / cvar(UR))
                        rec[f"{tag}_ddcond"] = np.mean([cosd(x, c) for x in U]) - loo
                        rec[f"{tag}_ddglob"] = np.mean([cosd(x, g) for x in U]) - np.mean([cosd(x, g) for x in UR])
                        rec[f"{tag}_lnorm"] = np.log(np.linalg.norm(X, axis=1).mean() / np.linalg.norm(R[u], axis=1).mean())
                        rec[f"{tag}_dprompt"] = np.mean([cosd(x, ep) for x in U]) - cosd(c, ep)
                        rec[f"{tag}_sim"] = np.mean(U @ unit(ep))
                    rows.append(rec)
    G = pd.DataFrame(rows)
    G.to_csv(ROOT / "results/E5_geometry.csv", index=False)
    out = []
    for (alg, s, e), g in G.groupby(["alg", "set", "emb"], sort=False):
        for m, log in [("lvr", True), ("ddcond", False), ("ddglob", False), ("lnorm", True), ("dprompt", False),
                       ("sim", False)]:
            a, b = stat(g[f"alg_{m}"], log), stat(g[f"base_{m}"], log)
            dlt = stat(g[f"alg_{m}"] - g[f"base_{m}"])
            out.append(dict(alg=alg, base=g.base.iloc[0], set=s, emb=e, measure=m, alg_est=a["est"], alg_lo=a["lo"],
                            alg_hi=a["hi"], base_est=b["est"], diff=dlt["est"], diff_lo=dlt["lo"], diff_hi=dlt["hi"],
                            diff_p=dlt["p"]))
    O = pd.DataFrame(out)
    O.to_csv(ROOT / "results/E5_geometry_summary.csv", index=False)
    return O


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    if what in ("all", "seed0"):
        S = seed0()
        cols = ["config", "set", "nfe", "calls", "wer", "sim_eres2net", "simr_eres2net", "utmos", "hf_ratio",
                "vsEuler_sim_eres2net", "vsEuler_sim_eres2net_p", "vsEuler_wer", "recovery_eres2net"]
        print(S[[c for c in cols if c in S]].round(4).to_string(index=False))
    if what in ("all", "geometry"):
        O = geometry()
        print(O[O.emb == "eres2net"].round(4).to_string(index=False))
