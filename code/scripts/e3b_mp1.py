"""v3 §2.1 M-P1: is the NFE1 output smoother than NFE2 (and the reference NFE)?
Spectral metrics per wav (24 kHz, |STFT| 1024/256, active frames by 40 dB energy VAD):
  hf_ratio  : energy 4-8 kHz / 0-8 kHz
  flatness  : mean spectral flatness (geometric/arithmetic mean of power, 0-8 kHz)
  flux_var  : variance of frame-to-frame L2 difference of 80-bin log-mel
Paired by utterance (seed 0, E1 outputs), en 300 / zh 300, families A (CosyVoice2) and B teacher (ZipVoice).
Note: measured on the vocoded waveform (re-extracted spectrum), not on the model's mel.
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"A": {1: "out/E1/A/cosyvoice2/nfe1/cfg0.7/utt0", 2: "out/E1/A/cosyvoice2/nfe2/cfg0.7/utt0",
              10: "out/E1/A/cosyvoice2/nfe10/cfg0.7/utt0"},
        "B": {1: "out/E1/B/zipvoice/nfe1/cfg1.0/utt0", 2: "out/E1/B/zipvoice/nfe2/cfg1.0/utt0",
              16: "out/E1/B/zipvoice/nfe16/cfg1.0/utt0"}}
rng = np.random.default_rng(0)


def spec(path):
    w, sr = sf.read(path, dtype="float32")
    if sr != 24000:
        w = librosa.resample(w, orig_sr=sr, target_sr=24000)
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=256)) ** 2
    f = librosa.fft_frequencies(sr=24000, n_fft=1024)
    e = S[f < 8000].sum(0)
    act = e > e.max() * 10 ** (-40 / 10)
    Sa = S[:, act]
    hf = Sa[(f >= 4000) & (f < 8000)].sum() / (Sa[f < 8000].sum() + 1e-12)
    P = Sa[f < 8000] + 1e-12
    flat = float(np.mean(np.exp(np.mean(np.log(P), 0)) / np.mean(P, 0)))
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=Sa, sr=24000, n_mels=80, fmax=8000) + 1e-10)
    flux = np.linalg.norm(np.diff(mel, axis=1), axis=0)
    return dict(hf_ratio=float(hf), flatness=flat, flux_var=float(np.var(flux)))


def ci(x):
    bs = rng.choice(x, (10000, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5]), wilcoxon(x).pvalue


def main():
    rows = []
    for fam, d in DIRS.items():
        for s in ["en", "zh"]:
            utts = sorted(p.stem for p in (ROOT / d[1] / s).glob("*.wav"))
            for nfe, dd in d.items():
                for u in utts:
                    rows.append(dict(fam=fam, set=s, nfe=nfe, utt=u, **spec(ROOT / dd / s / f"{u}.wav")))
            print(fam, s, "done", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results" / "E3b_mp1_spectral.csv", index=False)
    out = []
    for (fam, s), g in D.groupby(["fam", "set"]):
        ref = max(DIRS[fam])
        P = g.pivot(index="utt", columns="nfe")
        for m in ["hf_ratio", "flatness", "flux_var"]:
            for a, b in [(1, ref), (2, ref), (1, 2)]:
                est, lo, hi, p = ci((P[(m, a)] - P[(m, b)]).values)
                out.append(dict(fam=fam, set=s, metric=m, comp=f"NFE{a}-NFE{b}", mean_a=P[(m, a)].mean(),
                                mean_b=P[(m, b)].mean(), diff=est, lo=lo, hi=hi, p=p))
    O = pd.DataFrame(out)
    O.to_csv(ROOT / "results" / "E3b_mp1_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(O.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
