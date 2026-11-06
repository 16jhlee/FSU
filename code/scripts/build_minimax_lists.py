"""E8'-5: MiniMax multilingual test lists (ko / ja / de), seed-tts-eval meta format.

utt|prompt_text|prompt_wav|text  with utt = mm_<lang>_<line:03d>_<female|male>.
Prompt mp3 (Common Voice) -> mono wav at native rate in data/raw/minimax_wav/<lang>/ (no other processing),
because the CosyVoice frontend and the evaluators read wav. 100 lines per language (both speakers, as listed).
Run in env_eval:  python scripts/build_minimax_lists.py
"""
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MM = ROOT / "data" / "raw" / "minimax"
OUT_WAV = ROOT / "data" / "raw" / "minimax_wav"
LANGS = {"ko": "korean", "ja": "japanese", "de": "german"}

ptext = {}
for l in (MM / "speaker" / "prompt_text.txt").read_text(encoding="utf-8").splitlines():
    if "|" in l:
        f, t = l.split("|", 1)
        ptext[f.strip()] = t.strip()

for code, lang in LANGS.items():
    spk = {}
    for d in (MM / "speaker" / lang).iterdir():
        mp3 = next(d.glob("*.mp3"))
        wav = OUT_WAV / code / (mp3.stem + ".wav")
        if not wav.exists():
            wav.parent.mkdir(parents=True, exist_ok=True)
            y, sr = librosa.load(str(mp3), sr=None, mono=True)
            sf.write(str(wav), y, sr, subtype="PCM_16")
        spk[d.name] = (ptext[mp3.name], wav)
    rows = []
    for i, l in enumerate((MM / "text" / f"{lang}.txt").read_text(encoding="utf-8").splitlines()):
        if not l.strip():
            continue
        s, text = l.split("|", 1)
        pt, wav = spk[s.strip()]
        g = s.strip().split("_")[-1]
        rows.append(f"mm_{code}_{i:03d}_{g}|{pt}|{wav}|{text.strip()}")
    p = ROOT / "data" / "lists" / f"mm_{code}.meta.lst"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(code, len(rows), {k: (round(sf.info(str(v[1])).duration, 2), v[0][:40]) for k, v in spk.items()})
