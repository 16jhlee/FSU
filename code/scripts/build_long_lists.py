"""v8 §2.2 long-sentence lists for the streaming test. Seed-TTS-eval sentences are short (en <= 27 words), so each long
text joins three test sentences of the same set (rows i, i+100, i+200 of en_all / zh_all) with the prompt of row i.
-> data/lists/long_en.meta.lst, long_zh.meta.lst (100 each). Target >= 10 s; actual durations are reported."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for s, sep in [("en", " "), ("zh", "")]:
    R = [l.split("|") for l in (ROOT / f"data/lists/{s}_all.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()]
    out = [f"long_{s}_{i:03d}|{R[i][1]}|{R[i][2]}|{sep.join(R[j][3] for j in (i, i + 100, i + 200))}" for i in range(100)]
    (ROOT / f"data/lists/long_{s}.meta.lst").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(s, len(out))
