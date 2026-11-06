"""Aggregate E1: long-format results CSV + paired comparisons vs the family reference.

Paired by utt_id (same utterance, same tokens/duration, same noise seed). For each metric:
mean diff (setting - reference), bootstrap 95% CI (10k), Wilcoxon signed-rank p, Cliff's delta.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ROOT, jobs  # noqa: E402

EMBS = ["wavlm", "ecapa", "eres2net", "campp"]  # campp (zh-cn, = CosyVoice cond. encoder) -> appendix
METRICS = (["wer"] + [f"sim_{e}" for e in EMBS] + [f"simr_{e}" for e in EMBS]
           + [f"sim_{e}_c6" for e in EMBS] + ["utmos"])
REF = {"cv2": "cv2_nfe10", "zipvoice": "zipvoice_nfe16", "zipvoice_distill": "zipvoice_nfe16",
       "dmo_teacher": "dmo_teacher_nfe32", "dmo_student": "dmo_teacher_nfe32", "dmo_official": "dmo_teacher_nfe32",
       "f5v0": "f5v0_nfe32"}
rng = np.random.default_rng(0)


def group(name):
    for k in sorted(REF, key=len, reverse=True):
        if name.startswith(k):
            return k
    return None


def cliffs(d):
    d = np.asarray(d)
    return float((np.sign(d)).mean())  # paired version: P(x>y) - P(x<y) over pairs


def pkey(p):
    import os
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


_RESYNTH = {}


def resynth(voc, e):
    if (voc, e) not in _RESYNTH:
        _RESYNTH[(voc, e)] = dict(np.load(ROOT / "data" / "cache" / f"prompt_resynth_{voc}_{e}.npz"))
    return _RESYNTH[(voc, e)]


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load():
    rows = []
    for j in jobs():
        if not j["out"]:
            continue
        voc = "hift" if j["name"].startswith(("cv2", "cv3")) else "vocos"  # family vocoder for SIM-r
        for s, meta, _ in j["sets"]:
            d = ROOT / j["out"] / s
            f = d / "_metrics" / "metrics.csv"
            if not (d / "_evaldone").exists() or not f.exists():
                continue
            df = pd.read_csv(f)
            pw = {l.split("|")[0]: l.split("|")[2] for l in
                  (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()}
            for e in EMBS:
                ef = d / "_metrics" / f"emb_{e}.npz"
                if not ef.exists() or not (ROOT / "data" / "cache" / f"prompt_resynth_{voc}_{e}.npz").exists():
                    continue
                g, r = np.load(ef), resynth(voc, e)
                # prompts outside the E1 subset (CosyVoice3 full set) were not resynthesised -> NaN
                df[f"simr_{e}"] = [cosv(g[f"gen/{u}"], r[pkey(pw[u])]) if pkey(pw[u]) in r else np.nan
                                   for u in df.utt_id]
            df["job"], df["set"], df["dir"] = j["name"], s, j["out"]
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    d = load()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    d.to_csv(out / "E1_long.csv", index=False, encoding="utf-8")
    recs = []
    for (job, s), g in d.groupby(["job", "set"], sort=False):
        grp = group(job)
        ref = d[(d.job == REF.get(grp)) & (d.set == s)].set_index("utt_id")
        g = g.set_index("utt_id")
        r = dict(job=job, set=s, n=len(g))
        for m in METRICS:
            if m not in g:
                continue
            r[m] = g[m].mean() * (100 if m == "wer" else 1)
            if len(ref) and job != REF[grp]:
                common = g.index.intersection(ref.index)
                diff = (g.loc[common, m] - ref.loc[common, m]).values * (100 if m == "wer" else 1)
                bs = rng.choice(diff, (10000, len(diff))).mean(1)
                try:
                    p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                except ValueError:
                    p = np.nan
                r[f"d_{m}"], r[f"lo_{m}"], r[f"hi_{m}"] = diff.mean(), *np.percentile(bs, [2.5, 97.5])
                r[f"p_{m}"], r[f"delta_{m}"] = p, cliffs(diff)
        recs.append(r)
    S = pd.DataFrame(recs)
    S.to_csv(out / "E1_summary.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    for s in ["en", "zh", "hard"]:
        t = S[S.set == s]
        if not len(t):
            continue
        print(f"\n=== {s} ===")
        cols = [c for c in ["job", "n", "wer", "sim_wavlm", "sim_ecapa", "sim_eres2net", "utmos"] if c in t]
        show = t[cols].copy()
        for m in ["wer", "sim_wavlm", "sim_ecapa", "sim_eres2net"]:
            if f"d_{m}" in t:
                sc = 1 if m == "wer" else 100
                show[f"Δ{m}"] = [("" if pd.isna(a) else f"{a*sc:+.2f}[{lo*sc:+.2f},{hi*sc:+.2f}]{'*' if p < 0.05 else ''}")
                                 for a, lo, hi, p in zip(t[f"d_{m}"], t[f"lo_{m}"], t[f"hi_{m}"], t[f"p_{m}"])]
        print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
