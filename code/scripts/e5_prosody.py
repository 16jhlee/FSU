"""E5: across-seed prosody variance (F0 mean, F0 sd, active speaking rate) for the 8-seed configs,
as log variance ratio vs the teacher reference and vs the Euler baseline (paired over utterances)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from e5_analyze import BASELINE, TEACHER, fam  # noqa: E402
from e5_jobs import CONFIGS, ROOT, SETS, wav_dir  # noqa: E402

F = ["f0_mean_st", "f0_sd_st", "rate_active"]
rng = np.random.default_rng(0)


def var(name, s):
    A = pd.concat([pd.read_csv(wav_dir(name, k, s) / "_metrics" / "acoustic.csv").assign(k=k) for k in range(8)])
    return A.groupby("utt_id")[F].var()


def est(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (10000, len(x))).mean(1)
    return np.exp(x.mean()), np.exp(np.percentile(bs, 2.5)), np.exp(np.percentile(bs, 97.5)), wilcoxon(x).pvalue


rows = []
for n, c in CONFIGS.items():
    if len(c["K"]) != 8 or n not in BASELINE:
        continue
    for s, _, _ in SETS:
        a, b, t = var(n, s), var(BASELINE[n], s), var(TEACHER[fam(n)], s)
        for f in F:
            vt = est(np.log((a[f] + 1e-9) / (t[f] + 1e-9)))
            vb = est(np.log((b[f] + 1e-9) / (t[f] + 1e-9)))
            vd = est(np.log((a[f] + 1e-9) / (b[f] + 1e-9)))
            rows.append(dict(alg=n, base=BASELINE[n], set=s, feat=f, alg_vs_teacher=vt[0], base_vs_teacher=vb[0],
                             alg_vs_base=vd[0], lo=vd[1], hi=vd[2], p=vd[3]))
D = pd.DataFrame(rows)
D.to_csv(ROOT / "results/E5_prosody.csv", index=False)
pd.set_option("display.width", 200)
print(D[D.set == "en"].round(3).to_string(index=False))
