"""E16 (v10.9 §4) listening samples: 3 LibriSpeech-PC utterances (first, middle, last of the list) per condition
-> samples/E16/<decoder>/<utt>/<key>.wav (+ prompt.wav). python scripts/e16_samples.py (env_eval)"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e15_analyze as E15  # noqa: E402
import e16_analyze as AN  # noqa: E402

ROOT = AN.ROOT
lines = [l.split("|") for l in (ROOT / "data/lists/lspc_all.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()]
pick = [lines[0], lines[len(lines) // 2], lines[-1]]
n = 0
for dec, items in AN.BLOCKS.items():
    for utt, _, prompt, _ in pick:
        d = ROOT / "samples" / "E16" / dec.replace(" ", "_") / utt
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prompt, d / "prompt.wav")
        for key, _, cfg, _ in items:
            f = Path(E15.W(cfg, 0, "lspc")) / f"{utt}.wav"
            if f.exists():
                shutil.copy2(f, d / f"{key}.wav")
                n += 1
print(n, "files")
