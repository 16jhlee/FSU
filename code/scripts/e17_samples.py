"""E17 (v10.10 §4) listening samples: 3 utterances (first, middle, last of the list) per filled condition, the E17 rows and
their comparators -> samples/E17/<group>_<set>/<utt>/<key>.wav (+ prompt.wav). python scripts/e17_samples.py (env_eval)"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e15_analyze as E15  # noqa: E402
import e17_analyze as AN  # noqa: E402

ROOT = AN.ROOT
T = pd.read_csv(ROOT / "results/E17_robust_fill.csv")
n = 0
for (g, s), x in T.groupby(["group", "set"]):
    if not (x.block == "E17").any():
        continue
    meta = next(m for tag, _, sets in AN.groups() if tag == g for st, m, _ in sets if st == s)
    lines = [l.split("|") for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()]
    for utt, _, prompt, _ in [lines[0], lines[len(lines) // 2], lines[-1]]:
        d = ROOT / "samples" / "E17" / f"{g}_{s}" / utt
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prompt, d / "prompt.wav")
        for _, r in x.iterrows():
            f = Path(E15.W(r.config, 0, s)) / f"{utt}.wav" if isinstance(r.config, str) else None
            if f is not None and f.exists():
                shutil.copy2(f, d / f"{r.key}.wav")
                n += 1
print(n, "files")
