"""E7 analysis: prescription rule (v4 §1.3).
Outputs results/E7_seed0_summary.csv, E7_geometry_summary.csv, reports/fig/E7_fig5_heatmap.png and prints the
E7-P1..P3 inputs. ΔSIM = ERes2Net-en SIM-o x100 vs the default (all-step CFG) run at the same NFE, paired by utt."""
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
from e7_jobs import CONFIGS, REQUEST, ROOT, SETS, TEACHER, VARIANT_TO_CONFIG, mask, VARIANTS, wav_dir  # noqa: E402

EMBS = ["wavlm", "ecapa", "eres2net"]
rng = np.random.default_rng(0)
MODELS = {"cv2": ("A", [2, 3, 4, 10], "cv2_default_n{}"), "zip": ("B", [2, 4, 8, 16], "zip_default_n{}")}


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (10000, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5]), (wilcoxon(x).pvalue if np.any(x != 0) else 1.0)


def load(name, s):
    d = wav_dir(name, 0, s)
    f = d / "_metrics" / "metrics.csv"
    if not f.exists():
        return None
    D = pd.read_csv(f).set_index("utt_id")
    ac = d / "_metrics" / "acoustic.csv"
    if ac.exists():
        D = D.join(pd.read_csv(ac).set_index("utt_id")[["hf_ratio"]])
    D["wer"] = D["wer"] * 100
    return D


def seed0():
    rows = []
    for model, (fam, nfes, dflt) in MODELS.items():
        variants = sorted({v for (m, n), vs in REQUEST.items() if m == model for v in vs}) + \
            (["ramp0.2", "ramp0.4"] if model == "cv2" else [])
        for s, _, _ in SETS:
            T = load(TEACHER[fam], s)
            for n in nfes:
                B = load(dflt.format(n), s)
                if B is None:
                    continue
                for v in ["default"] + variants:
                    name = dflt.format(n) if v == "default" else VARIANT_TO_CONFIG.get((model, n, v))
                    if name is None:
                        continue
                    D = load(name, s)
                    if D is None:
                        continue
                    r = dict(model=model, fam=fam, nfe=n, variant=v, config=name, set=s, n_utt=len(D),
                             mask="".join(map(str, mask(model, n, VARIANTS[v]))) if v in VARIANTS else
                             ("1" * n if v == "default" else "ramp"))
                    for m in ["wer", "utmos", "hf_ratio"] + [f"sim_{e}" for e in EMBS]:
                        if m in D:
                            r[m] = D[m].mean()
                    c = D.index.intersection(B.index)
                    for m in ["wer"] + [f"sim_{e}" for e in EMBS]:
                        est, lo, hi, p = stat((D.loc[c, m] - B.loc[c, m]) * (1 if m == "wer" else 100))
                        r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"], r[f"d_{m}_p"] = est, lo, hi, p
                    if T is not None:
                        gap = (T["sim_eres2net"].mean() - B["sim_eres2net"].mean()) * 100
                        r["teacher_gap"] = gap
                        r["vs_teacher"] = (D["sim_eres2net"].mean() - T["sim_eres2net"].mean()) * 100
                        r["recovery"] = r["d_sim_eres2net"] / gap if gap > 1e-6 else np.nan
                    rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(ROOT / "results/E7_seed0_summary.csv", index=False)
    return S


def heatmap(S):
    cols = ["first", "iv0.1-1", "iv0.2-1", "iv0.3-1", "iv0.2-0.8"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
    for ax, model in zip(axes, ["cv2", "zip"]):
        fam, nfes, _ = MODELS[model]
        M = np.full((len(nfes), len(cols)), np.nan)
        for i, n in enumerate(nfes):
            for j, v in enumerate(cols):
                r = S[(S.model == model) & (S.nfe == n) & (S.variant == v) & (S.set == "en")]
                if len(r):
                    M[i, j] = r.d_sim_eres2net.iloc[0]
        vmax = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(nfes)):
            for j in range(len(cols)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(M[i, j]) > 0.6 * vmax else "#0b0b0b")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(["first step\nw=0", "CFG only\n[0.1,1]", "[0.2,1]", "[0.3,1]", "[0.2,0.8]"], fontsize=8.5)
        ax.set_yticks(range(len(nfes)))
        ax.set_yticklabels([f"NFE {n}" for n in nfes])
        ax.set_title(f"{'A CosyVoice2' if model == 'cv2' else 'B ZipVoice teacher'} — ΔSIM vs all-step CFG (×100, en)",
                     loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("E7 Fig 5 — which steps get CFG (repo time, t=0 noise; ERes2Net-en SIM-o, seed 0, n=300)",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(ROOT / "reports/fig/E7_fig5_heatmap.png", dpi=150)
    plt.close(fig)


def pkey(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosd(a, b):
    return 1 - unit(a) @ unit(b)


def cvar(U):
    return len(U) / (len(U) - 1) * np.mean(np.sum((U - U.mean(0)) ** 2, axis=1))


def geometry():
    core = [(VARIANT_TO_CONFIG[("zip", 4, "first")], "zip_default_n4", "zip_default_n16"),
            (VARIANT_TO_CONFIG[("zip", 4, "iv0.2-0.8")], "zip_default_n4", "zip_default_n16"),
            (VARIANT_TO_CONFIG[("cv2", 2, "first")], "cv2_default_n2", "cv2_default_n10"),
            (VARIANT_TO_CONFIG[("cv2", 3, "first")], "cv2_default_n3", "cv2_default_n10")]
    meta = {s: {l.split("|")[0]: l.split("|")[2] for l in (ROOT / (m + ".meta.lst")).read_text(encoding="utf-8").splitlines()
                if l.strip()} for s, m, _ in SETS[:2]}
    out = []
    for s, _, _ in SETS[:2]:
        for e in EMBS:
            prom = dict(np.load(ROOT / "data/cache" / f"prompt_{e}.npz"))

            def E(name):
                arr = [np.load(wav_dir(name, k, s) / "_metrics" / f"emb_{e}.npz") for k in range(8)]
                return {u: np.stack([a[f"gen/{u}"] for a in arr]).astype(float) for u in meta[s]}
            cache = {}
            for alg, base, ref in core:
                for x in (alg, base, ref):
                    if x not in cache:
                        cache[x] = E(x)
                R = cache[ref]
                g = unit(unit(np.concatenate(list(R.values()))).mean(0))
                recs = []
                for u in R:
                    UR = unit(R[u])
                    c = unit(UR.mean(0))
                    loo = np.mean([cosd(UR[k], unit(np.delete(UR, k, 0).mean(0))) for k in range(8)])
                    ep = prom[pkey(meta[s][u])]
                    rec = {}
                    for tag, X in [("alg", cache[alg][u]), ("base", cache[base][u])]:
                        U = unit(X)
                        rec[f"{tag}_lvr"] = np.log(cvar(U) / cvar(UR))
                        rec[f"{tag}_ddcond"] = np.mean([cosd(x, c) for x in U]) - loo
                        rec[f"{tag}_ddglob"] = np.mean([cosd(x, g) for x in U]) - np.mean([cosd(x, g) for x in UR])
                        rec[f"{tag}_lnorm"] = np.log(np.linalg.norm(X, axis=1).mean() / np.linalg.norm(R[u], axis=1).mean())
                        rec[f"{tag}_dprompt"] = np.mean([cosd(x, ep) for x in U]) - cosd(c, ep)
                        rec[f"{tag}_sim"] = np.mean(U @ unit(ep))
                    recs.append(rec)
                G = pd.DataFrame(recs)
                for m in ["lvr", "ddcond", "ddglob", "lnorm", "dprompt", "sim"]:
                    a, b = G[f"alg_{m}"], G[f"base_{m}"]
                    f = np.exp if m in ("lvr", "lnorm") else (lambda z: z)
                    est, lo, hi, p = stat(a - b)
                    out.append(dict(alg=alg, base=base, set=s, emb=e, measure=m, alg_est=f(a.mean()), base_est=f(b.mean()),
                                    diff=est, lo=lo, hi=hi, p=p))
    O = pd.DataFrame(out)
    O.to_csv(ROOT / "results/E7_geometry_summary.csv", index=False)
    return O


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    if what in ("all", "seed0"):
        S = seed0()
        heatmap(S)
        cols = ["model", "nfe", "variant", "mask", "set", "wer", "d_wer", "d_wer_p", "sim_eres2net", "d_sim_eres2net",
                "d_sim_eres2net_lo", "d_sim_eres2net_hi", "d_sim_eres2net_p", "d_sim_wavlm", "d_sim_ecapa", "utmos",
                "hf_ratio", "vs_teacher", "recovery"]
        print(S[[c for c in cols if c in S]].round(3).to_string(index=False))
    if what in ("all", "geometry"):
        O = geometry()
        print(O[O.emb == "eres2net"].round(4).to_string(index=False))
