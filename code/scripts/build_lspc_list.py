"""E8'-10 JRR reproduction set: LibriSpeech-PC test-clean cross-sentence list (F5-TTS repo, 1,127 pairs, 4-10 s) ->
data/lists/lspc_all.meta.lst (utt|prompt_text|prompt_wav|text, utt = target id) with prompt wavs converted to 16 kHz wav.
LibriSpeech test-clean from openslr (resource 12). env_dom.  python scripts/build_lspc_list.py
"""
import subprocess
import tarfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DL = Path(r"C:\speechFM_store\dl\librispeech")
OUT = ROOT / "data/raw/librispeech_pc"
DL.mkdir(parents=True, exist_ok=True)
(OUT / "prompt").mkdir(parents=True, exist_ok=True)
tgz = DL / "test-clean.tar.gz"
if not tgz.exists() or tgz.stat().st_size < 300e6:
    for k in range(5):  # urllib stalled at 0 bytes on this machine (2026-10-27); curl resumes and retries
        subprocess.call(["curl.exe", "-L", "-s", "-C", "-", "--retry", "5", "-o", str(tgz),
                         "https://www.openslr.org/resources/12/test-clean.tar.gz"])
        if tgz.exists() and tgz.stat().st_size > 300e6:
            break
        print("retry", k, flush=True)
if not (DL / "LibriSpeech").exists():
    with tarfile.open(tgz) as t:
        t.extractall(DL)
rows = [l.split("\t") for l in (ROOT / "repos/F5-TTS/data/librispeech_pc_test_clean_cross_sentence.lst").read_text(
    encoding="utf-8").splitlines() if l.strip()]
lines = []
for pid, _, ptext, tid, _, ttext in rows:
    spk, ch, _ = pid.split("-")
    src = DL / "LibriSpeech/test-clean" / spk / ch / f"{pid}.flac"
    dst = OUT / "prompt" / f"{pid}.wav"
    if not dst.exists():
        y, sr = sf.read(str(src))
        sf.write(str(dst), y, sr)
    lines.append(f"{tid}|{ptext}|{dst}|{ttext}")
(ROOT / "data/lists/lspc_all.meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(lines), "pairs", flush=True)
