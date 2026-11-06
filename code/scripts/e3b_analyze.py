"""E3b §2.2 analysis (mel space, before the vocoder). env_eval.

Distances: RMS over mel elements, d(A,B) = sqrt(mean((A-B)^2)) in log-mel units (CosyVoice2 log magnitude mel).
(a) support test with eps fixed (approved correction of v3 (a)):
    nn(x)   = min_k d(x, x10_k), k = 0..7 (reference NFE10 samples, 8 seeds)
    nn_ref  = for each sample k, min_{j!=k} d(x10_k, x10_j); off-support if nn(x) > 95th percentile of nn_ref (per utt)
    centre  : d(x, mean_k x10_k) vs leave-one-out d(x10_k, mean_{j!=k} x10_j)
    states  : d(z_e, z_ode) (Euler vs ODE state at t1) and d(z_lin, z_ode) (straight interpolation vs ODE state)
(b,c) smoothness of x0hat (= NFE1 output), nfe2 (= prediction at the Euler state), x10 (seed 0),
    xhat_ode / xhat_lin (predictions at on-path states, same t1):
    hf      : energy share of mel bins with centre >= 4 kHz (energy = exp(2*mel))
    flat    : mean over frames of spectral flatness across mel bins
    flux    : variance over frames of the L2 norm of the time difference of log-mel
Outputs: results/E3b_mechanism_utt.csv, results/E3b_mechanism_summary.csv
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
CENT = librosa.mel_frequencies(n_mels=82, fmin=0, fmax=8000)[1:-1]
HF = CENT >= 4000
rng = np.random.default_rng(0)


def d(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smooth(m):
    e = np.exp(2 * m.astype(np.float64))
    hf = e[HF].sum() / e.sum()
    flat = np.mean(np.exp(np.mean(np.log(e + 1e-12), 0)) / (np.mean(e, 0) + 1e-12))
    flux = np.var(np.linalg.norm(np.diff(m, axis=1), axis=0))
    return hf, flat, flux


def stat(x):
    x = np.asarray(x, float)
    bs = rng.choice(x, (10000, len(x))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean=x.mean(), lo=lo, hi=hi, p=wilcoxon(x).pvalue if np.any(x != 0) else 1.0, n=len(x))


def main():
    import sys
    sdir = sys.argv[1] if len(sys.argv) > 1 else "out/E3b/states"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    rows = []
    for f in sorted((ROOT / sdir).glob("*.npz")):
        z = np.load(f)
        X = z["x10"]
        K = len(X)
        nn_ref = [min(d(X[k], X[j]) for j in range(K) if j != k) for k in range(K)]
        q95 = np.percentile(nn_ref, 95)
        xbar = X.mean(0)
        loo = np.mean([d(X[k], np.delete(X, k, 0).mean(0)) for k in range(K)])
        r = dict(utt=f.stem, frames=X.shape[2], nn_ref_med=np.median(nn_ref), nn_ref_q95=q95,
                 pair_nat=np.mean([d(X[i], X[j]) for i in range(K) for j in range(i + 1, K)]), centre_loo=loo,
                 dz_euler_ode=d(z["z_e"], z["z_ode"]), dz_lin_ode=d(z["z_lin"], z["z_ode"]))
        for name in ["x0hat", "nfe2", "xhat_ode", "xhat_lin"]:
            x = z[name]
            nn = min(d(x, X[k]) for k in range(K))
            r[f"nn_{name}"] = nn
            r[f"off_{name}"] = nn > q95
            r[f"centre_{name}"] = d(x, xbar)
            r[f"dev_{name}"] = d(x, X[0])
        for name in ["x0hat", "nfe2", "x10", "xhat_ode", "xhat_lin"]:
            m = X[0] if name == "x10" else z[name]
            r[f"hf_{name}"], r[f"flat_{name}"], r[f"flux_{name}"] = smooth(m)
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / f"results/E3b_mechanism{tag}_utt.csv", index=False)
    print("utterances:", len(D))
    out = []
    # (a)
    for name in ["x0hat", "nfe2", "xhat_ode", "xhat_lin"]:
        out.append(dict(test="a", what=f"off-support fraction {name}", mean=D[f"off_{name}"].mean()))
        out.append(dict(test="a", what=f"nn({name}) - q95(nn_ref)", **stat(D[f"nn_{name}"] - D.nn_ref_q95)))
        out.append(dict(test="a", what=f"centre dist {name} - LOO sample centre dist",
                        **stat(D[f"centre_{name}"] - D.centre_loo)))
    out.append(dict(test="a", what="d(z_euler, z_ode) - d(z_lin, z_ode)", **stat(D.dz_euler_ode - D.dz_lin_ode)))
    # (b,c) + M-P1 at mel level
    for met in ["hf", "flat", "flux"]:
        for a, b in [("x0hat", "x10"), ("nfe2", "x10"), ("x0hat", "nfe2"), ("nfe2", "xhat_ode"),
                     ("xhat_ode", "x10"), ("xhat_lin", "x10")]:
            out.append(dict(test="b/c", what=f"{met}: {a} - {b}", **stat(D[f"{met}_{a}"] - D[f"{met}_{b}"])))
        for name in ["x0hat", "nfe2", "x10", "xhat_ode", "xhat_lin"]:
            out.append(dict(test="level", what=f"{met} {name}", mean=D[f"{met}_{name}"].mean()))
    for c in ["nn_ref_med", "nn_ref_q95", "pair_nat", "centre_loo", "dz_euler_ode", "dz_lin_ode"] + \
             [f"nn_{n}" for n in ["x0hat", "nfe2", "xhat_ode", "xhat_lin"]]:
        out.append(dict(test="level", what=c, mean=D[c].mean()))
    S = pd.DataFrame(out)
    S.to_csv(ROOT / f"results/E3b_mechanism{tag}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print(S.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
