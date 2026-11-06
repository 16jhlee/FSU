"""v8 §2.2 chunk-boundary metrics, streaming vs offline at the SAME positions (CosyVoice2, long_en / long_zh).
Boundary positions = chunk ends of the streaming run (<out>/_chunks.tsv, samples @ 24 kHz). For each wav:
  flux_ratio = var of log-mel frame differences within ±40 ms of the boundaries / the same quantity at all other frames
  f0_jump    = fraction of consecutive voiced-frame pairs within ±40 ms of a boundary whose F0 differs by > 3 semitones
               (pyworld harvest, 10 ms hop), and the same fraction elsewhere
The offline wav of the same configuration is measured at the streaming run's boundary positions (control).
env_eval.  python scripts/v8_chunk_metrics.py  -> results/E8v8_chunks.csv
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyworld
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import e8v8s_jobs as S  # noqa: E402

ROOT = S.ROOT
SR, HOP = 24000, 240  # 10 ms frames
WIN = 4  # ±40 ms


def boundaries(d):
    out = {}
    f = d / "_chunks.tsv"
    if f.exists():
        for l in f.read_text(encoding="utf-8").splitlines():
            u, _, ends = l.split("\t")
            out.setdefault(u, []).extend(int(x) for x in ends.split(",") if x)
    return out


def measure(wav, bnd):
    y, sr = sf.read(wav)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    M = np.log(librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=HOP, n_mels=80) + 1e-6)
    dif = np.sum(np.diff(M, axis=1) ** 2, axis=0)
    n = len(dif)
    near = np.zeros(n, bool)
    for b in bnd:
        c = b // HOP
        near[max(0, c - WIN):min(n, c + WIN + 1)] = True
    if near.sum() == 0 or (~near).sum() == 0:
        return np.nan, np.nan, np.nan
    flux_ratio = np.var(dif[near]) / np.var(dif[~near])
    f0, t = pyworld.harvest(y.astype(np.float64), SR, frame_period=10.0)
    f0 = f0[: n + 1]
    v = (f0[:-1] > 0) & (f0[1:] > 0)
    jump = np.zeros_like(v)
    jump[v] = np.abs(12 * np.log2(f0[1:][v] / f0[:-1][v])) > 3
    nb = near[: len(v)]
    jn = jump[nb & v].mean() if (nb & v).any() else np.nan
    jo = jump[~nb & v].mean() if (~nb & v).any() else np.nan
    return flux_ratio, jn, jo


def main():
    rows = []
    for tag in ["n2_def", "n2_R", "n3_def", "n3_R", "n10_def"]:
        for s, _, _ in S.SETS:
            dS, dO = S.wav_dir(f"s_str_{tag}", 0, s), S.wav_dir(f"s_off_{tag}", 0, s)
            B = boundaries(dS)
            for u, bnd in B.items():
                if len(bnd) < 2:
                    continue
                for mode, d in [("stream", dS), ("offline", dO)]:
                    fr, jn, jo = measure(d / f"{u}.wav", bnd[:-1])  # last chunk end = end of audio
                    rows.append(dict(cond=tag, set=s, utt=u, mode=mode, n_bound=len(bnd) - 1, flux_ratio=fr,
                                     f0_jump_near=jn, f0_jump_other=jo))
        print(tag, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E8v8_chunks.csv", index=False)
    print(D.groupby(["cond", "set", "mode"])[["n_bound", "flux_ratio", "f0_jump_near", "f0_jump_other"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
