"""E13-1 (v10.5): embedding centre-to-centre distance against the guided teacher (analysis only; existing 8-seed embeddings
_metrics/emb_{eres2net,wavlm,ecapa}.npz, no generation). en_100 / zh_100 (seed 0 read from the 300 folders).
Per utterance, X = 8 L2-normalised embeddings of a condition, T = 8 of the teacher:
  mu = mean of the unit vectors (Euclidean, not renormalised); sigma^2 = mean_k ||x_k - mu||^2 (total variance)
  delta      = ||mu_f - mu_t|| / sigma_t
  delta_corr = sqrt(max(0, delta^2 - (sigma_f^2 + sigma_t^2) / (8 sigma_t^2)))    (centre-estimation noise removed)
  delta_null = teacher 8 seeds split 4/4 at random (20 splits): sqrt(max(0, ||mu_A - mu_B||^2/sigma_t^2
               - (sigma_A^2 + sigma_B^2)/(4 sigma_t^2))), mean over splits   (same-distribution expectation; n = 4 halves)
  dir        = unit vector (mu_f - mu_t)/||.||; reported: mean pairwise cosine across utterances (direction consistency)
Conditions: baseline 2-step, R (ours), mid+R (ours), R′ (Sel-CFG input_text), R′+R (CV2) for CosyVoice2/3, IndexTTS-2 and
GLM-TTS (official speaker-null; no R′), vs each model's teacher. Missing 8-seed conditions are listed and skipped.
python scripts/e13_centre.py   (env_eval) -> results/E13_centre_shift.csv (per utterance), results/E13_centre_shift_summary.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e11_jobs as J11  # noqa: E402
import e8p8_analyze as M8  # noqa: E402
import e8p9_jobs as J9  # noqa: E402
import e13_jobs as J13  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMBS = ["eres2net", "wavlm", "ecapa"]
rng = np.random.default_rng(0)
# v10.6 E13-10 additions draw from their own generator so that every existing row (noise floor, CIs) is unchanged
NEW = {("GLM-TTS all-null", "first step only (10)"), ("GLM-TTS all-null", "mid+R (ours)")}
rng_new = np.random.default_rng(1)
W8 = M8.W
COND = {  # model -> (teacher, [(label, config)], path function)
    "CosyVoice2": ("e82_cv2_all_n10_def", [("baseline", "e82_cv2_all_n2_def"), ("R (ours)", "e82_cv2_all_n2_R"),
                                           ("mid+R (ours)", "A_cv2_midR"), ("R′", "e82_cv2_speaker_n2_def"),
                                           ("R′+R", "e82_cv2_speaker_n2_R"), ("baseline N=4", "e89:c_cv2_def_n4"),
                                           ("R (ours) N=4", "e89:c_cv2_R_n4"), ("baseline N=8", "e89:c_cv2_def_n8"),
                                           ("R (ours) N=8", "e89:c_cv2_R_n8"), ("first step only (10)", "e13:c2_10"),
                                           ("w=0 N=10", "e13:c2_n10_w0")], W8),
    "CosyVoice3": ("e81_cv3_n10_def", [("baseline", "e81_cv3_n2_def"), ("R (ours)", "e81_cv3_n2_R"), ("mid+R (ours)", "A_cv3_midR"),
                                       ("R′", "B_cv3_Rp_n2"), ("first step only (10)", "e13:c3_10"), ("no guidance (00)", "e13:c3_00")], W8),
    "IndexTTS-2": ("e81_idx_n25_def", [("baseline", "e81_idx_n2_def"), ("R (ours)", "e81_idx_n2_R"), ("mid+R (ours)", "A_idx_midR"),
                                       ("R′", "B_idx_Rp_n2"), ("baseline N=4", "e89:c_idx_def_n4"), ("R (ours) N=4", "e89:c_idx_R_n4"),
                                       ("baseline N=8", "e89:c_idx_def_n8"), ("R (ours) N=8", "e89:c_idx_R_n8"),
                                       ("first step only (10)", "e13:ix_10"), ("(0.35, 0.7)", "e13:ix_h"), ("no guidance (00)", "e13:ix_00")], W8),
    "GLM-TTS": ("g_base_n10", [("baseline", "g_base_n2"), ("R (ours)", "g_R_n2"), ("mid+R (ours)", "g_midR")], J11.wav_dir),
    # E13-5: all-null branch against its own all-null teacher (ga_T) and the speaker-null teacher as a condition (G3)
    "GLM-TTS all-null": ("e13:ga_T", [("baseline", "e13:ga_11"), ("R (ours)", "e13:ga_01"), ("(0.35, 0.7)", "e13:ga_h"),
                                      ("no guidance (00)", "e13:ga_00"), ("speaker-null teacher", "e11:g_base_n10"),
                                     ("first step only (10)", "e13:ga_10"), ("mid+R (ours)", "e13:ga_midR")], J11.wav_dir),
}


def res(cfg, wd):
    for pre, fn in (("e89:", J9.wav_dir), ("e13:", J13.wav_dir), ("e11:", J11.wav_dir)):
        if cfg.startswith(pre):
            return cfg[len(pre):], fn
    return cfg, wd


def stats8(X):
    mu = X.mean(0)
    return mu, float(np.mean(np.sum((X - mu) ** 2, 1)))


def null(T, sig_t2, n=20, g=None):
    g = rng if g is None else g
    vals = []
    for _ in range(n):
        p = g.permutation(len(T))
        A, B = T[p[:4]], T[p[4:]]
        (ma, sa), (mb, sb) = stats8(A), stats8(B)
        vals.append(np.sqrt(max(0.0, np.sum((ma - mb) ** 2) / sig_t2 - (sa + sb) / (4 * sig_t2))))
    return float(np.mean(vals))


def boot(x, g=None):
    g = rng if g is None else g
    x = np.asarray(x, float)
    b = g.choice(x, (5000, len(x))).mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def main():
    rows, missing, dirs, vecs = [], [], {}, {}
    for s, lst in (("en", "data/lists/en_100"), ("zh", "data/lists/zh_100")):
        utts = [l.split("|")[0] for l in (ROOT / (lst + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()]
        for model, (tname, conds, wd) in COND.items():
            for e in EMBS:
                try:
                    T = EA.emb_stack(*res(tname, wd)[:1], s, e, utts, res(tname, wd)[1])
                except (FileNotFoundError, KeyError) as ex:
                    missing.append((model, "teacher", tname, s, e, str(ex)[:120]))
                    continue
                for label, cfg in conds:
                    try:
                        c_, f_ = res(cfg, wd)
                        X = EA.emb_stack(c_, s, e, utts, f_)
                    except (FileNotFoundError, KeyError) as ex:
                        missing.append((model, label, cfg, s, e, str(ex)[:120]))
                        continue
                    D = []
                    for u in utts:
                        Tu, Xu = EA.A.unit(T[u]), EA.A.unit(X[u])
                        (mt, st2), (mf, sf2) = stats8(Tu), stats8(Xu)
                        d2 = np.sum((mf - mt) ** 2) / st2
                        dv = (mf - mt) / np.linalg.norm(mf - mt)
                        D.append(dv)
                        vecs.setdefault((model, label, s, e), {})[u] = mf - mt
                        dk = Xu - Tu  # same noise seed k in both conditions -> paired differences
                        db = dk.mean(0)
                        dpc2 = (db @ db - np.sum((dk - db) ** 2) / (8 * 7)) / st2
                        rows.append(dict(model=model, cond=label, config=cfg, set=s, emb=e, utt=u, delta=np.sqrt(d2),
                                         delta_paired=np.sqrt(max(0.0, dpc2)), delta_paired_raw2=dpc2,
                                         delta_corr=np.sqrt(max(0.0, d2 - (sf2 + st2) / (8 * st2))), delta_null=null(Tu, st2, g=rng_new if (model, label) in NEW else None),
                                         sigma_f=np.sqrt(sf2), sigma_t=np.sqrt(st2), var_ratio=sf2 / st2))
                    Dm = np.stack(D)
                    C = Dm @ Dm.T
                    dirs[(model, label, s, e)] = float((C.sum() - np.trace(C)) / (len(Dm) * (len(Dm) - 1)))
                print(model, s, e, flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(ROOT / "results/E13_centre_shift.csv", index=False)
    S = []
    for (model, cond, s, e), g in P.groupby(["model", "cond", "set", "emb"], sort=False):
        r = dict(model=model, cond=cond, set=s, emb=e, n=len(g))
        for c in ("delta", "delta_corr", "delta_null", "var_ratio", "delta_paired"):
            m, lo, hi = boot(g[c], rng_new if (model, cond) in NEW else None)
            r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
        r["ratio_to_null"] = r["delta_corr"] / r["delta_null"]
        r["dir_consistency"] = dirs[(model, cond, s, e)]
        S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(ROOT / "results/E13_centre_shift_summary.csv", index=False)
    pd.DataFrame(missing, columns=["model", "cond", "config", "set", "emb", "error"]).to_csv(ROOT / "results/E13_centre_missing.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[S.emb == "eres2net"][["model", "cond", "set", "delta", "delta_corr", "delta_corr_lo", "delta_corr_hi", "delta_null",
                                  "ratio_to_null", "delta_paired", "var_ratio", "dir_consistency"]].round(3).to_string(index=False))
    # E13-1b: direction of the remaining shift, per utterance cos(mu_X - mu_t, mu_base - mu_t), base = the same-N baseline
    Rd = []
    for (model, label, s, e), v in vecs.items():
        if label.startswith("baseline"):
            continue
        bl = "baseline" + (label[label.index(" N="):] if " N=" in label else "")
        b = vecs.get((model, bl, s, e))
        if not b:
            continue
        cs = [float(v[u] @ b[u] / np.linalg.norm(v[u]) / np.linalg.norm(b[u])) for u in v]
        m, lo, hi = boot(cs, rng_new if (model, label) in NEW else None)
        Rd.append(dict(model=model, cond=label, vs=bl, set=s, emb=e, n=len(cs), cos_to_base_shift=m, lo=lo, hi=hi))
    Rd = pd.DataFrame(Rd)
    Rd.to_csv(ROOT / "results/E13_centre_residual_dir.csv", index=False)
    print(Rd[Rd.emb == "eres2net"].round(3).to_string(index=False))
    print("missing:", sorted({(m, c, cfg) for m, c, cfg, *_ in missing}))


if __name__ == "__main__":
    main()
