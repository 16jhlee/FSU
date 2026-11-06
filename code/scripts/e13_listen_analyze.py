"""E13-7 listening-test analysis: samples/E13/listening/responses/*.csv (rater, trial_id, response; downloaded from the task
pages) + _key/key.csv -> results/E13_listen_{smos,abx,cmos}.csv and the L1–L3 verdicts.
Raters failing more than one attention check in a task are excluded from that task. Utterance = unit of analysis
(responses averaged over raters per utterance x condition), paired Wilcoxon / bootstrap 95% CI over utterances.
  L1 SMOS(R) - SMOS(baseline) > 0, p < 0.05
  L2 ABX: rate at which the condition is chosen over the teacher (confusion with the teacher) is higher for R than baseline
  L3 CMOS (R vs teacher, + = R better) 95% CI inside +-0.3 and containing 0
python scripts/e13_listen_analyze.py   (env_eval)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "samples" / "E13" / "listening"
rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, float)
    b = rng.choice(x, (10000, len(x))).mean(1)
    return x.mean(), *np.percentile(b, [2.5, 97.5])


def check_ok(r):
    c, v = r.check, r.response
    if c == "same>=4":
        return float(v) >= 4
    if c == "different<=2":
        return float(v) <= 2
    if c.startswith("answer="):
        return str(v) == c.split("=")[1]
    if c == "abs<=1":
        return abs(float(v)) <= 1
    return True


def main():
    files = sorted((D / "responses").glob("*.csv"))
    if not files:
        print("no responses in", D / "responses")
        return
    R = pd.concat([pd.read_csv(f, dtype={"response": str}) for f in files])
    K = pd.read_csv(D / "_key" / "key.csv", dtype=str)
    X = R.merge(K, on="trial_id")
    X["check"] = X["check"].fillna("")
    chk = X[X.check != ""].copy()
    chk["ok"] = chk.apply(check_ok, axis=1)
    bad = chk.groupby(["task", "model", "set", "rater"]).ok.apply(lambda s: (~s).sum() > 1)
    X = X[X.check == ""].merge(bad.rename("bad").reset_index(), on=["task", "model", "set", "rater"], how="left")
    X = X[~X.bad.fillna(False)]
    out = {}
    for (m, s), g in X.groupby(["model", "set"]):
        sm = g[g.task == "smos"].assign(v=lambda d: d.response.astype(float)).groupby(["utt", "cond"]).v.mean().unstack()
        if {"R", "base"} <= set(sm.columns):
            d = sm["R"] - sm["base"]
            est, lo, hi = boot(d)
            out.setdefault("smos", []).append(dict(model=m, set=s, n_utt=len(d), raters=g[g.task == "smos"].rater.nunique(),
                                                   **{f"smos_{c}": sm[c].mean() for c in sm.columns}, d_R_base=est, lo=lo, hi=hi,
                                                   p=wilcoxon(d).pvalue, L1=bool(est > 0 and wilcoxon(d).pvalue < 0.05)))
        ab = g[g.task == "abx"].assign(v=lambda d: (d.response == d.cond_side).astype(float)).groupby(["utt", "cond"]).v.mean().unstack()
        if {"R", "base"} <= set(ab.columns):
            d = ab["R"] - ab["base"]
            est, lo, hi = boot(d)
            out.setdefault("abx", []).append(dict(model=m, set=s, n_utt=len(d), **{f"confusion_{c}": ab[c].mean() for c in ab.columns},
                                                  d_R_base=est, lo=lo, hi=hi, p=wilcoxon(d).pvalue, L2=bool(est > 0)))
        cm = g[g.task == "cmos"].copy()
        if len(cm):
            cm["v"] = cm.response.astype(float) * np.where(cm.cond_side == "B", 1, -1)
            u = cm.groupby("utt").v.mean()
            est, lo, hi = boot(u)
            out.setdefault("cmos", []).append(dict(model=m, set=s, n_utt=len(u), cmos_R_vs_teacher=est, lo=lo, hi=hi,
                                                   L3=bool(-0.3 < lo <= 0 <= hi < 0.3)))
    for t, rows in out.items():
        df = pd.DataFrame(rows)
        df.to_csv(ROOT / f"results/E13_listen_{t}.csv", index=False)
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
