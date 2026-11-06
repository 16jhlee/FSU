"""E3 extra diagnostics per generated wav (v2 §2.3 (i), (ii)). CPU only.

Prosody (i):
  f0_mean_st, f0_sd_st : WORLD dio+stonemask F0 (16 kHz, 5 ms), voiced frames only, in semitones re 100 Hz
  rate_active          : syllables / active-speech seconds. Total duration is fixed across seeds by design,
                         so the rate is computed over frames above an energy VAD threshold
                         (frame RMS > max_rms * 10^(-40/20), 20 ms frames). Syllables: en = vowel groups
                         per word (min 1), zh = CJK characters.
Spectral smearing (ii), from the 24 kHz wav, |STFT| 1024/256:
  hf_ratio    : energy 4-8 kHz / energy 0-8 kHz over active frames
  flux_var    : variance over active frames of the frame-to-frame L2 difference of log-mel (80 bins)
Output: <wav_dir>/_metrics/acoustic.csv
"""
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pyworld
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def syllables(text, lang):
    if lang == "zh":
        return max(1, len(re.findall(r"[一-鿿]", text)))
    if lang == "ko":  # Hangul syllable blocks
        return max(1, len(re.findall(r"[\uac00-\ud7a3]", text)))
    if lang == "ja":  # kana + kanji characters (rough mora proxy)
        return max(1, len(re.findall(r"[ぁ-ゟ゠-ヿ一-鿿]", text)))
    if lang == "de":
        return max(1, sum(max(1, len(re.findall(r"[aeiouyäöü]+", w.lower())))
                          for w in re.findall(r"[A-Za-zÄÖÜäöüß']+", text)))
    n = 0
    for w in re.findall(r"[A-Za-z']+", text):
        n += max(1, len(re.findall(r"[aeiouy]+", w.lower())))
    return max(1, n)


def feats(path, text, lang):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    w24 = librosa.resample(w, orig_sr=sr, target_sr=24000) if sr != 24000 else w
    # VAD (20 ms frames)
    hop = 480
    fr = librosa.util.frame(np.pad(w24, (0, hop)), frame_length=hop, hop_length=hop)
    rms = np.sqrt((fr ** 2).mean(0) + 1e-12)
    active = rms > rms.max() * 10 ** (-40 / 20)
    act_sec = active.sum() * hop / 24000
    # F0
    w16 = librosa.resample(w24, orig_sr=24000, target_sr=16000).astype(np.float64)
    f0, t = pyworld.dio(w16, 16000, frame_period=5.0)
    f0 = pyworld.stonemask(w16, f0, t, 16000)
    v = f0[f0 > 0]
    st = 12 * np.log2(v / 100.0) if len(v) else np.array([np.nan])
    # spectral
    S = np.abs(librosa.stft(w24, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=24000, n_fft=1024)
    act_s = np.repeat(active, 480 // 256 + 1)[: S.shape[1]] if S.shape[1] else active
    act_s = np.resize(act_s, S.shape[1]).astype(bool)
    Sa = S[:, act_s] if act_s.any() else S
    hf = Sa[(freqs >= 4000) & (freqs < 8000)].sum() / (Sa[freqs < 8000].sum() + 1e-12)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=Sa, sr=24000, n_mels=80, fmax=8000) + 1e-10)
    flux = np.linalg.norm(np.diff(mel, axis=1), axis=0)
    return dict(f0_mean_st=float(np.mean(st)), f0_sd_st=float(np.std(st)), voiced_frac=float(len(v) / max(1, len(f0))),
                active_sec=act_sec, rate_active=syllables(text, lang) / max(act_sec, 1e-3),
                hf_ratio=float(hf), flux_var=float(np.var(flux)) if len(flux) else np.nan)


def main():
    meta, wav_dir, lang = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    out = wav_dir / "_metrics" / "acoustic.csv"
    if out.exists():
        return
    rows = []
    for l in Path(meta).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        u, _, _, text = l.split("|")[:4]
        p = wav_dir / f"{u}.wav"
        if p.exists():
            rows.append(dict(utt_id=u, **feats(p, text, lang)))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(wav_dir, len(rows))


if __name__ == "__main__":
    main()
