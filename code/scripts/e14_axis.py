"""E14-3: is the residual centre shift along the teacher's guidance axis? (analysis of 8-seed embeddings, en/zh 100)
Per utterance, mu = mean of the 8 L2-normalised embeddings (not renormalised):
  mu_t guided teacher, mu_0 w = 0 sampler at the teacher's step count, mu_m method m (2-step), a = mu_t - mu_0, r = mu_m - mu_t
  cos_res_unguided = cos(r, -a)    (> 0: towards the unguided side)
  share_along      = (r . a_hat)^2 / ||r||^2
  proj_ratio       = -(r . a_hat) / ||a||   (fraction of the axis travelled towards the unguided sampler)
Noise floor: teacher 8 seeds split 4/4 (20 random splits), r = mu_A - mu_B with a = mu_B - mu_0, same three quantities.
  (a and r share mu_t, so its estimation noise biases cos_res_unguided upward; the floor shows how much.)
Cross-fit (added after the first CV2 run, 2026-11-03): cos_cf / proj_cf with a from seeds A and r from seeds B (crossfit()).
Utterance mean + bootstrap 95% CI (5,000). Models without the w = 0 run are listed as missing (no generation here).
python scripts/e14_axis.py [model ...]   (env_eval) -> results/E14_residual_axis.csv (per utterance), _summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e11_jobs as J11  # noqa: E402
import e13_jobs as J13  # noqa: E402
import e14_jobs as J14  # noqa: E402
import e8p10_jobs as J10b  # noqa: E402
import e8p8_analyze as M8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMBS = ["eres2net", "wavlm", "ecapa"]
W8 = M8.W
MODELS = {  # model -> (teacher, w = 0 sampler, [(label, config)])
    "CosyVoice2": (("e82_cv2_all_n10_def", W8), ("c2_n10_w0", J13.wav_dir),
                   [("baseline", ("e82_cv2_all_n2_def", W8)), ("R (ours)", ("e82_cv2_all_n2_R", W8)),
                    ("mid+R (ours)", ("A_cv2_midR", W8)), ("first step only (10)", ("c2_10", J13.wav_dir)),
                    ("no guidance (00)", ("cv2_tfr_n2", J10b.wav_dir))]),
    "CosyVoice3": (("e81_cv3_n10_def", W8), ("c3_n10_w0", J14.wav_dir),
                   [("baseline", ("e81_cv3_n2_def", W8)), ("R (ours)", ("e81_cv3_n2_R", W8)), ("mid+R (ours)", ("A_cv3_midR", W8)),
                    ("first step only (10)", ("c3_10", J13.wav_dir)), ("no guidance (00)", ("c3_00", J13.wav_dir))]),
    "IndexTTS-2": (("e81_idx_n25_def", W8), ("ix_n25_w0", J14.wav_dir),
                   [("baseline", ("e81_idx_n2_def", W8)), ("R (ours)", ("e81_idx_n2_R", W8)), ("mid+R (ours)", ("A_idx_midR", W8)),
                    ("first step only (10)", ("ix_10", J13.wav_dir)), ("no guidance (00)", ("ix_00", J13.wav_dir))]),
    "GLM-TTS": (("g_base_n10", J11.wav_dir), ("g_n10_w0", J14.wav_dir),
                [("baseline", ("g_base_n2", J11.wav_dir)), ("R (ours)", ("g_R_n2", J11.wav_dir)),
                 ("mid+R (ours)", ("g_midR", J11.wav_dir))]),
    "GLM-TTS all-null": (("ga_T", J13.wav_dir), ("g_n10_w0", J14.wav_dir),
                         [("baseline", ("ga_11", J13.wav_dir)), ("R (ours)", ("ga_01", J13.wav_dir)),
                          ("first step only (10)", ("ga_10", J14.wav_dir)), ("no guidance (00)", ("ga_00", J13.wav_dir))]),
}
rng = np.random.default_rng(0)


def meas(r, a):
    na, nr = np.linalg.norm(a), np.linalg.norm(r)
    p = r @ a / na
    return dict(cos_res_unguided=-p / nr, share_along=p * p / (nr * nr), proj_ratio=-p / na)


def crossfit(Tu, Zu, Xu, n=20):
    """a from seed half A, r from the other half B (paired within each half: same noise seeds) -> teacher-centre noise is
    not shared between a and r, so the null expectation of cos / proj is 0 (magnitudes attenuated by the 4-seed noise)."""
    c, p = [], []
    for _ in range(n):
        q = rng.permutation(8)
        A, B = q[:4], q[4:]
        aA = Tu[A].mean(0) - Zu[A].mean(0)
        rB = Xu[B].mean(0) - Tu[B].mean(0)
        c.append(-(rB @ aA) / np.linalg.norm(rB) / np.linalg.norm(aA))
        p.append(-(rB @ aA) / (aA @ aA))
    return dict(cos_cf=float(np.mean(c)), proj_cf=float(np.mean(p)))


def boot(x):
    x = np.asarray(x, float)
    b = rng.choice(x, (5000, len(x))).mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def main(only):
    rows, missing = [], []
    for s, lst in (("en", "data/lists/en_100"), ("zh", "data/lists/zh_100")):
        utts = [l.split("|")[0] for l in (ROOT / (lst + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()]
        for model, ((tc, tw), (zc, zw), conds) in MODELS.items():
            if only and model not in only:
                continue
            for e in EMBS:
                try:
                    T, Z = EA.emb_stack(tc, s, e, utts, tw), EA.emb_stack(zc, s, e, utts, zw)
                except (FileNotFoundError, KeyError) as ex:
                    missing.append((model, "teacher / w=0", s, e, str(ex)[:150]))
                    continue
                mt = {u: EA.A.unit(T[u]).mean(0) for u in utts}
                a = {u: mt[u] - EA.A.unit(Z[u]).mean(0) for u in utts}
                for u in utts:  # noise floor
                    Tu, vals = EA.A.unit(T[u]), []
                    for _ in range(20):
                        p = rng.permutation(8)
                        mA, mB = Tu[p[:4]].mean(0), Tu[p[4:]].mean(0)
                        vals.append(meas(mA - mB, mB - (mt[u] - a[u])))
                    rows.append(dict(model=model, cond="noise floor (teacher 4/4)", set=s, emb=e, utt=u,
                                     **pd.DataFrame(vals).mean().to_dict(), norm_a=np.linalg.norm(a[u]), norm_r=np.nan))
                for label, (c, w) in conds:
                    try:
                        X = EA.emb_stack(c, s, e, utts, w)
                    except (FileNotFoundError, KeyError) as ex:
                        missing.append((model, label, s, e, str(ex)[:150]))
                        continue
                    for u in utts:
                        r = EA.A.unit(X[u]).mean(0) - mt[u]
                        rows.append(dict(model=model, cond=label, set=s, emb=e, utt=u, **meas(r, a[u]),
                                         **crossfit(EA.A.unit(T[u]), EA.A.unit(Z[u]), EA.A.unit(X[u])),
                                         norm_a=np.linalg.norm(a[u]), norm_r=np.linalg.norm(r)))
                print(model, s, e, flush=True)
    P = pd.DataFrame(rows)
    tag = "" if not only else "_" + "_".join(m.split()[0] for m in only)
    P.to_csv(ROOT / f"results/E14_residual_axis{tag}.csv", index=False)
    S = []
    for (model, cond, s, e), g in P.groupby(["model", "cond", "set", "emb"], sort=False):
        r = dict(model=model, cond=cond, set=s, emb=e, n=len(g))
        for c in ("cos_res_unguided", "share_along", "proj_ratio", "cos_cf", "proj_cf", "norm_a", "norm_r"):
            if c not in g:
                continue
            m, lo, hi = boot(g[c].dropna()) if g[c].notna().any() else (np.nan,) * 3
            r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
        S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(ROOT / f"results/E14_residual_axis{tag}_summary.csv", index=False)
    pd.DataFrame(missing, columns=["model", "cond", "set", "emb", "error"]).to_csv(
        ROOT / f"results/E14_residual_axis{tag}_missing.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[S.emb == "eres2net"][["model", "cond", "set", "cos_res_unguided", "cos_res_unguided_lo", "cos_res_unguided_hi",
                                  "share_along", "proj_ratio", "cos_cf", "cos_cf_lo", "cos_cf_hi", "proj_cf", "norm_a",
                                  "norm_r"]].round(3).to_string(index=False))
    print("missing:", missing)


if __name__ == "__main__":
    main(sys.argv[1:])
