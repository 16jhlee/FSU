"""E13: mark reused (ALIAS) folders so the queues skip them: _done when the folder holds every wav of its list, _qdone when
its _metrics/metrics.csv already has the metrics e13_jobs.metrics_for asks for (rows = wavs). Prints what is new work.
python scripts/e13_premark.py [--dry]   (env_eval)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e13_jobs as J  # noqa: E402

NEED = {"wer": "wer", "utmos": "utmos", "wavlm": "sim_wavlm", "ecapa": "sim_ecapa", "eres2net": "sim_eres2net"}
dry = "--dry" in sys.argv
new_gen, new_eval, marked = 0, 0, 0
for name, c in J.CONFIGS.items():
    for k in c["K"]:
        for s, m, lang in J.sets_for(name, k):
            d = J.wav_dir(name, k, s)
            n = J.n_lines(m)
            if "E13" in d.parts:
                new_gen += n
                continue
            if not dry:
                (d / "_done").touch()
            f = d / "_metrics" / "metrics.csv"
            cols = [NEED[x] for x in J.metrics_for(name, k, s, lang).split(",")]
            ok = f.exists() and all(col in pd.read_csv(f, nrows=1).columns for col in cols) and sum(1 for _ in open(f, encoding="utf-8")) - 1 >= n
            if ok:
                marked += 1
                if not dry:
                    (d / "_qdone").write_text("ok")
            else:
                new_eval += n
                print("reused folder needs evaluation:", name, k, s, d)
print(f"new wavs to generate: {new_gen}; reused folders marked evaluated: {marked}; reused wavs to (re)evaluate: {new_eval}")
