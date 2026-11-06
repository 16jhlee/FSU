"""E3 pilot analysis (v2 §2.2). env_eval.

Cloud size = unbiased variance of L2-normalised embeddings around their centroid:
    V = n/(n-1) * mean_i ||u_i - mean(u)||^2           (u = e/||e||;  ||u-v||^2 = 2(1-cos))
  V_seed  : 8 noise seeds of one utterance at the reference NFE
  V_noise : 4 copies of the k=0 wav (orig, +1/-1 sample shift, -0.5 dB, 16-bit re-encode)
  V_real  : 3 real recordings of one speaker (Seed-TTS en prompt + its 2 ground-truth targets)
Also SIM-scale: sd over seeds of SIM-o vs eps_SIM (E1b).
Pass rule (v2): V_seed > 3 * V_noise for the main embedders, and the cloud is not a point.
"""
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import EMBEDDERS, load16, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMBS = ["wavlm", "ecapa", "eres2net", "campp"]
MODELS = {"A_cosyvoice2_nfe10": ("out/E3/pilot/A/cosyvoice2/nfe10/cfg0.7/utt{k}/en",
                                 "out/E1/A/cosyvoice2/nfe10/cfg0.7/utt0/en"),
          "B_zipvoice_nfe16": ("out/E3/pilot/B/zipvoice/nfe16/cfg1.0/utt{k}/en",
                               "out/E1/B/zipvoice/nfe16/cfg1.0/utt0/en"),
          "C_dmo_teacher_nfe32": ("out/E3/pilot/C/dmo_teacher/nfe32/cfg2.0/utt{k}/en",
                                  "out/E1/C/dmo_teacher/nfe32/cfg2.0/utt0/en")}
K = 8
SEEDTTS_EN = ROOT / "data/raw/seedtts/seedtts_testset/en"


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cloud_var(E):
    U = unit(E)
    n = len(U)
    return n / (n - 1) * np.mean(np.sum((U - U.mean(0)) ** 2, axis=1))


def perturbed_copies(wav_path, tmp):
    w, sr = sf.read(wav_path, dtype="float32")
    vs = [w, np.concatenate([[0.0], w[:-1]]), np.concatenate([w[1:], [0.0]]), w * 10 ** (-0.5 / 20)]
    out = []
    for i, v in enumerate(vs):
        f = tmp / f"{abs(hash(str(wav_path))) % 10**9}_{i}.wav"
        sf.write(f, np.asarray(v, dtype="float32"), sr, subtype="PCM_16")
        out.append(str(f))
    return out


def main():
    rows = read_meta(ROOT / "data/lists/en_20.meta.lst")
    utts = [r["utt"] for r in rows]
    prompt = {r["utt"]: r["prompt_wav"] for r in rows}

    # determinism check: regenerated k=0 vs E1 wav
    det = {}
    for m, (pat, e1) in MODELS.items():
        diffs = []
        for u in utts:
            a, _ = sf.read(ROOT / pat.format(k=0) / f"{u}.wav", dtype="float32")
            b, _ = sf.read(ROOT / e1 / f"{u}.wav", dtype="float32")
            diffs.append(np.abs(a - b).max() if a.shape == b.shape else np.inf)
        det[m] = float(np.max(diffs))
    print("determinism (max |k0 - E1|):", det, flush=True)

    # real same-speaker recordings: 100 speakers with prompt + 2 GT targets
    g = defaultdict(list)
    for l in (SEEDTTS_EN / "meta.lst").read_text(encoding="utf-8").splitlines():
        f = l.split("|")
        if len(f) >= 4:
            g[f[2]].append(f[0])
    spk = sorted(p for p, v in g.items() if len(v) == 2)
    rng = np.random.default_rng(0)
    spk = list(rng.choice(spk, 100, replace=False))
    real_sets = [[str(SEEDTTS_EN / p)] + [str(SEEDTTS_EN / "wavs" / f"{u}.wav") for u in g[p]] for p in spk]

    tmp = Path(tempfile.mkdtemp())
    pert = {(m, u): perturbed_copies(ROOT / pat.format(k=0) / f"{u}.wav", tmp)
            for m, (pat, _) in MODELS.items() for u in utts}
    eps = (pd.read_csv(ROOT / "results/E1b_noise.csv").groupby("emb").dsim.quantile(0.95)).to_dict()

    recs, store = [], {}
    for e in EMBS:
        emb = EMBEDDERS[e]()
        V_real = [cloud_var([emb(load16(p)) for p in s]) for s in real_sets]
        for m, (pat, _) in MODELS.items():
            for u in utts:
                E = np.stack([emb(load16(str(ROOT / pat.format(k=k) / f"{u}.wav"))) for k in range(K)])
                store[f"{e}/{m}/{u}"] = E.astype(np.float32)
                ep = emb(load16(prompt[u]))
                sims = unit(E) @ unit(ep)
                Vn = cloud_var([emb(load16(p)) for p in pert[(m, u)]])
                recs.append(dict(emb=e, model=m, utt=u, V_seed=cloud_var(E), V_noise=Vn, sim_sd=sims.std(ddof=1),
                                 sim_mean=sims.mean(), eps_sim=eps[e]))
        recs.append(dict(emb=e, model="REAL", utt="", V_real_med=float(np.median(V_real)),
                         V_real_mean=float(np.mean(V_real))))
        del emb
        print(e, "done", flush=True)
    np.savez(ROOT / "results/E3_pilot_emb.npz", **store)
    D = pd.DataFrame(recs)
    D.to_csv(ROOT / "results/E3_pilot.csv", index=False)
    real = D[D.model == "REAL"].set_index("emb")
    X = D[D.model != "REAL"].copy()
    X["ratio_noise"] = X.V_seed / X.V_noise
    X["simvar_ratio"] = X.sim_sd ** 2 / X.eps_sim ** 2
    S = X.groupby(["model", "emb"]).agg(V_seed=("V_seed", "median"), V_noise=("V_noise", "median"),
                                         ratio_med=("ratio_noise", "median"),
                                         frac_ratio_gt3=("ratio_noise", lambda r: (r > 3).mean()),
                                         sim_sd=("sim_sd", "median"), simvar_over_eps2=("simvar_ratio", "median"))
    S["V_real"] = [real.loc[e, "V_real_med"] for _, e in S.index]
    S["V_seed/V_real"] = S.V_seed / S.V_real
    pd.set_option("display.width", 220)
    print(S.round(5).to_string())
    S.to_csv(ROOT / "results/E3_pilot_summary.csv")


if __name__ == "__main__":
    main()
