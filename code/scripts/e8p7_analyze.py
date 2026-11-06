"""E8'-7 analysis: external baselines vs R (ours) — table 7. env_eval.
python scripts/e8p7_analyze.py [table|geometry|all]
Paired by utterance (seed 0, en 300 / zh 300): d vs baseline (same NFE) and d vs R (ours) (same NFE); SIM x100.
Calls = network evaluations actually needed (CFG evaluation = 2, w=0 evaluation = 1, a zero-init step = 0).
Extra compute = per-step reductions on top of CFG (MAMBO-G: 2 norms; CFG-Zero* s*: 1 dot + 1 norm).
Outputs: results/E8p7_table7.csv, results/E8p7_geometry.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e8p7_jobs as J7  # noqa: E402
import e8p_analyze as A  # noqa: E402
from e7_jobs import wav_dir as e7_wav_dir  # noqa: E402

ROOT = A.ROOT
EMBS, EPS = A.EMBS, A.EPS
MODEL = {"cv2": A.CV2, "cv3": A.CV3}


def W(name, k, s):
    return e7_wav_dir(name[3:], k, s) if name.startswith("e7:") else J7.wav_dir(name, k, s)


A.wdir = W
TEACHER = {"cv2": "e82_cv2_all_n10_def", "cv3": "e81_cv3_n10_def"}
METRICS = [f"sim_{e}" for e in EMBS] + ["wer", "utmos"]


def ts(n):
    return list(1 - np.cos(np.pi / 2 * np.arange(n) / n))


def calls_extra(label, n):
    t = ts(n)
    if label.startswith("Kynk"):
        lo, hi = [float(x) for x in label[label.index("[") + 1:-1].split(",")]
        return sum(2 if lo <= x <= hi else 1 for x in t), ""
    if label in ("R (ours)", "Sel-CFG def_text+R"):
        return sum(2 if x >= 0.1 else 1 for x in t), ""
    if label.startswith("CFG-Zero* zi"):
        return 2 * (n - 1), ("1 dot + 1 norm / step" if "s*" in label else "")
    if label == "CFG-Zero* s*":
        return 2 * n, "1 dot + 1 norm / step"
    if label.startswith("MAMBO-G"):
        return 2 * n, "2 norms / step"
    return 2 * n, ""


def rows_for():
    out = []
    for name, (model, label, n) in J7.LABEL.items():
        out.append((model, label, n, name))
    for model, n in [("cv2", 2), ("cv2", 3), ("cv2", 10), ("cv3", 2)]:
        out.append((model, "baseline", n, J7.BASE[(model, n)]))
        out.append((model, "R (ours)", n, J7.RCFG[(model, n)]))
    for n in [4]:
        out.append(("cv2", "baseline", n, J7.BASE[("cv2", n)]))
        out.append(("cv2", "R (ours)", n, J7.RCFG[("cv2", n)]))
    return out


def paired(a, b, s):
    D, R = A.load(a, s), A.load(b, s)
    if D is None or R is None:
        return {}
    c = D.index.intersection(R.index)
    r = {}
    for m in METRICS:
        sc = 100 if m.startswith("sim") else 1
        est, lo, hi, p = A.stat((R.loc[c, m] - D.loc[c, m]) * sc)
        r[m] = (est, lo, hi, p)
    return r


def table():
    recs = []
    for model, label, n, cfg in rows_for():
        for s in ["en", "zh"]:
            D = A.load(cfg, s)
            if D is None:
                continue
            calls, extra = calls_extra(label, n)
            r = dict(model=MODEL[model], method=label, nfe=n, set=s, config=cfg, calls=calls, extra=extra, n=len(D))
            for m in METRICS:
                r[m] = D[m].mean() * (100 if m.startswith("sim") else 1)
            base, rr = J7.BASE[(model, n)], J7.RCFG[(model, n)]
            for tag, ref in [("vs_base", base), ("vs_R", rr), ("vs_teacher", TEACHER[model])]:
                if ref == cfg:
                    continue
                pr = paired(ref, cfg, s)
                for m, (est, lo, hi, p) in pr.items():
                    r[f"{tag}_d_{m}"], r[f"{tag}_d_{m}_lo"], r[f"{tag}_d_{m}_hi"], r[f"{tag}_d_{m}_p"] = est, lo, hi, p
            T, B = A.load(TEACHER[model], s), A.load(base, s)
            gap = (T.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100
            r["recovery"] = (r["sim_eres2net"] - B.sim_eres2net.mean() * 100) / gap if gap > EPS["eres2net"] else np.nan
            recs.append(r)
    T7 = pd.DataFrame(recs)
    T7.to_csv(ROOT / "results/E8p7_table7.csv", index=False)
    return T7


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    if what in ("all", "table"):
        T7 = table()
        cols = ["model", "method", "nfe", "set", "calls", "sim_eres2net", "vs_base_d_sim_eres2net", "vs_R_d_sim_eres2net",
                "vs_R_d_sim_eres2net_p", "recovery", "vs_R_d_sim_wavlm", "vs_R_d_sim_ecapa", "wer", "vs_base_d_wer_p", "utmos",
                "vs_R_d_utmos"]
        print(T7.sort_values(["model", "nfe", "set", "method"])[cols].round(2).to_string(index=False))
    if what in ("all", "geometry"):
        G = [tuple(x) for x in getattr(J7, "GEOM7", [])]
        if G:
            O = A.geometry(G, W, "results/E8p7_geometry.csv")
            print(O[O.emb == "eres2net"].round(4).to_string(index=False))
