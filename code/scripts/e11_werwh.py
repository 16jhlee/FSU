"""E11 block A/B: Whisper-large-v3 CER (wer_wh) next to the official Paraformer CER for every zh seed-0 folder.
python scripts/e11_werwh.py   (tools env; eval GPU lock)"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402
import e11_jobs as J  # noqa: E402
from e1_jobs import ENVS  # noqa: E402

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
for name, c in J.CONFIGS.items():
    if c.get("prep"):
        continue
    for s, m, lang in J.sets_for(name, 0):
        if s != "zh":
            continue
        d = J.wav_dir(name, 0, s)
        with gpu_lock.Lock("eval_job"):
            rc = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", m + ".meta.lst", "--wav_dir", str(d),
                                  "--lang", "zh", "--metrics", "werwh"], cwd=J.ROOT, env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        print(name, rc, flush=True)
