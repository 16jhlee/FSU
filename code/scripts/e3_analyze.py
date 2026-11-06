"""E3 analysis (v2 §2.1 / §2.3): mode averaging (H1) vs over-smoothing (H0').

Per embedder e, comparison (few F vs reference R), set s, utterance u, with seeds k = 0..K-1
(only seeds complete for every config are used):
  U = L2-normalised raw embeddings, c_u = normalised mean of the K reference vectors (conditional centre)
  var_ratio   : V(U_F) / V(U_R), V = unbiased mean squared distance to centroid; averaged as log, then exp
  d_cond      : mean_k cosdist(U_F[k], c_u)  vs  teacher leave-one-out mean_k cosdist(U_R[k], c_u^(-k))
  d_glob      : mean_k cosdist(U_F[k], g)    vs  mean_k cosdist(U_R[k], g),  g = normalised mean of all
                reference vectors of the set (unweighted); aux g_p = mean of prompt embeddings
  norm        : mean_k ||E_F[k]|| / mean_k ||E_R[k]|| (raw, before normalisation)
  d_prompt    : mean_k cosdist(U_F[k], e_p) - cosdist(c_u, e_p)   (H1: ~0, H0': > 0)
                also reported: reference samples mean_k cosdist(U_R[k], e_p) - cosdist(c_u, e_p)
Prosody: across-seed variance ratio (F/R) of f0_mean_st, f0_sd_st, rate_active.
Smearing: mean over seeds of hf_ratio, flux_var, F minus R; Spearman with per-utt dSIM.
Stats: paired over utterances, bootstrap 95% CI (10k), Wilcoxon, paired Cliff's delta (mean sign).
Outputs: results/E3_measures.csv (utterance level), results/E3_summary.csv, results/E3_emb_{en,zh}.npz.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from e3_jobs import COMPARISONS, CONFIGS, K, ROOT, SETS, wav_dir  # noqa: E402

EMBS = ["wavlm", "ecapa", "eres2net"]
rng = np.random.default_rng(0)


def pkey(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosdist(a, b):
    return 1.0 - unit(a) @ unit(b)


def cvar(U):
    return len(U) / (len(U) - 1) * np.mean(np.sum((U - U.mean(0)) ** 2, axis=1))


def complete_seeds():
    ok = []
    for k in K:
        if all((wav_dir(n, k, s) / "_e3done").exists() for n in CONFIGS for s, _, _ in SETS):
            ok.append(k)
    return ok


def stats(x, log=False):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (10000, len(x))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    try:
        p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    except ValueError:
        p = np.nan
    m = x.mean()
    if log:
        return dict(est=np.exp(m), lo=np.exp(lo), hi=np.exp(hi), p=p, delta=float(np.sign(x).mean()), n=len(x))
    return dict(est=m, lo=lo, hi=hi, p=p, delta=float(np.sign(x).mean()), n=len(x))


def main():
    ks = complete_seeds()
    print("complete seeds:", ks, flush=True)
    assert len(ks) >= 2
    meta = {s: {l.split("|")[0]: l.split("|")[2] for l in (ROOT / (m + ".meta.lst")).read_text(encoding="utf-8").splitlines()
                if l.strip()} for s, m, _ in SETS}
    # load embeddings: emb[e][(cfg, s)] -> dict utt -> array [K, D]
    emb, store = {}, {}
    for e in EMBS:
        emb[e] = {}
        for n in CONFIGS:
            for s, _, _ in SETS:
                arrs = [np.load(wav_dir(n, k, s) / "_metrics" / f"emb_{e}.npz") for k in ks]
                d = {u: np.stack([a[f"gen/{u}"] for a in arrs]).astype(np.float64) for u in meta[s]}
                emb[e][(n, s)] = d
                for u, v in d.items():
                    store[f"{e}/{n}/{s}/{u}"] = v.astype(np.float32)
    # split by test set so each file stays under GitHub's 100 MB limit (no LFS)
    for s, _, _ in SETS:
        np.savez(ROOT / "results" / f"E3_emb_{s}.npz", **{k: v for k, v in store.items() if k.split("/")[2] == s})
    prom = {e: dict(np.load(ROOT / "data" / "cache" / f"prompt_{e}.npz")) for e in EMBS}
    ac = {(n, s): pd.concat([pd.read_csv(wav_dir(n, k, s) / "_metrics" / "acoustic.csv").assign(k=k) for k in ks])
          for n in CONFIGS for s, _, _ in SETS}

    rows = []
    for few, ref in COMPARISONS:
        for s, _, _ in SETS:
            for e in EMBS:
                R, F = emb[e][(ref, s)], emb[e][(few, s)]
                g = unit(unit(np.concatenate(list(R.values()))).mean(0))
                gp = unit(np.mean([unit(prom[e][pkey(meta[s][u])]) for u in R], axis=0))
                for u in R:
                    UR, UF = unit(R[u]), unit(F[u])
                    c = unit(UR.mean(0))
                    loo = [cosdist(UR[k], unit(np.delete(UR, k, 0).mean(0))) for k in range(len(UR))]
                    ep = prom[e][pkey(meta[s][u])]
                    rows.append(dict(
                        comp=f"{few} vs {ref}", few=few, ref=ref, set=s, emb=e, utt=u,
                        log_var_ratio=np.log(cvar(UF) / cvar(UR)),
                        d_cond_few=np.mean([cosdist(x, c) for x in UF]), d_cond_ref=np.mean(loo),
                        d_glob_few=np.mean([cosdist(x, g) for x in UF]), d_glob_ref=np.mean([cosdist(x, g) for x in UR]),
                        d_globp_few=np.mean([cosdist(x, gp) for x in UF]), d_globp_ref=np.mean([cosdist(x, gp) for x in UR]),
                        log_norm_ratio=np.log(np.linalg.norm(F[u], axis=1).mean() / np.linalg.norm(R[u], axis=1).mean()),
                        d_prompt_few=np.mean([cosdist(x, ep) for x in UF]) - cosdist(c, ep),
                        d_prompt_ref=np.mean([cosdist(x, ep) for x in UR]) - cosdist(c, ep),
                        dsim=np.mean(UF @ unit(ep)) - np.mean(UR @ unit(ep))))
    M = pd.DataFrame(rows)
    M["dd_cond"] = M.d_cond_few - M.d_cond_ref
    M["dd_glob"] = M.d_glob_few - M.d_glob_ref
    M["dd_globp"] = M.d_globp_few - M.d_globp_ref
    M.to_csv(ROOT / "results" / "E3_measures.csv", index=False)

    # acoustic per utterance
    arows = []
    for few, ref in COMPARISONS:
        for s, _, _ in SETS:
            A, B = ac[(few, s)], ac[(ref, s)]
            va, vb = A.groupby("utt_id").var(numeric_only=True), B.groupby("utt_id").var(numeric_only=True)
            ma, mb = A.groupby("utt_id").mean(numeric_only=True), B.groupby("utt_id").mean(numeric_only=True)
            for u in va.index.intersection(vb.index):
                r = dict(comp=f"{few} vs {ref}", set=s, utt=u)
                for f in ["f0_mean_st", "f0_sd_st", "rate_active"]:
                    r[f"lvr_{f}"] = np.log((va.loc[u, f] + 1e-9) / (vb.loc[u, f] + 1e-9))
                for f in ["hf_ratio", "flux_var"]:
                    r[f"d_{f}"] = ma.loc[u, f] - mb.loc[u, f]
                arows.append(r)
    A = pd.DataFrame(arows)
    A.to_csv(ROOT / "results" / "E3_acoustic_measures.csv", index=False)

    summ = []
    for (comp, s, e), g in M.groupby(["comp", "set", "emb"], sort=False):
        for meas, col, log in [("var_ratio", "log_var_ratio", True), ("Δd_cond", "dd_cond", False),
                               ("Δd_glob", "dd_glob", False), ("Δd_glob(prompt mean)", "dd_globp", False),
                               ("norm_ratio", "log_norm_ratio", True), ("Δd_prompt (few)", "d_prompt_few", False),
                               ("Δd_prompt (ref)", "d_prompt_ref", False), ("ΔSIM", "dsim", False)]:
            summ.append(dict(comp=comp, set=s, emb=e, measure=meas, **stats(g[col], log)))
    for (comp, s), g in A.groupby(["comp", "set"], sort=False):
        for meas in ["lvr_f0_mean_st", "lvr_f0_sd_st", "lvr_rate_active"]:
            summ.append(dict(comp=comp, set=s, emb="-", measure="var_ratio " + meas[4:], **stats(g[meas], True)))
        for meas in ["d_hf_ratio", "d_flux_var"]:
            summ.append(dict(comp=comp, set=s, emb="-", measure=meas, **stats(g[meas])))
            for e in EMBS:
                ds = M[(M.comp == comp) & (M.set == s) & (M.emb == e)].set_index("utt").dsim
                j = g.set_index("utt")[meas].reindex(ds.index)
                summ.append(dict(comp=comp, set=s, emb=e, measure=f"rho(ΔSIM,{meas})",
                                 est=spearmanr(ds, j, nan_policy="omit").correlation))
    S = pd.DataFrame(summ)
    S.to_csv(ROOT / "results" / "E3_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    show = S[S.set == "en"].copy()
    show["v"] = [f"{r.est:+.4f}" if pd.isna(r.lo) else f"{r.est:.4f} [{r.lo:.4f},{r.hi:.4f}]{'*' if r.p < 0.05 else ''}"
                 for r in show.itertuples()]
    print(show.pivot_table(index=["comp", "measure"], columns="emb", values="v", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
