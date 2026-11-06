"""E13-10 (v10.6): ga_midR seed 0 (en/zh 300, regenerated in out/E13) vs the same utterances of E14-1 allnull_midR
(en_full / zh_full, same command / cache / noise): audio data chunk byte-identical (float-WAV PEAK headers carry a write
timestamp, so whole-file bytes differ).   python scripts/e13_10_check.py -> results/E13_10_bitcheck.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e13_jobs as J13  # noqa: E402
import e14_jobs as J14  # noqa: E402


def data(p):
    x = p.read_bytes()
    return x[x.index(b"data"):]


rows = []
for s, full in (("en", "en_full"), ("zh", "zh_full")):
    a, b = Path(J13.wav_dir("ga_midR", 0, s)), Path(J14.wav_dir("allnull_midR", 0, full))
    names = sorted(f.name for f in a.glob("*.wav"))
    rows.append(dict(new=str(a), ref=str(b), n=len(names), n_identical=sum(data(a / n) == data(b / n) for n in names)))
with open(J13.ROOT / "results/E13_10_bitcheck.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(rows)
