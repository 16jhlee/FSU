"""E10-1′ ε: UTMOS (v1, UTMOS22 strong, as in every table) for teacher seeds 1-7 (en/zh) so that the resolution ε of
UTMOS is computed the same way as for UTMOSv2 / DNSMOS / NISQA (8-seed re-synthesis SD). eval_metrics.py adds the
utmos column only. Inside the evaluation GPU lock.   python scripts/e10_eps_utmos.py   (tools env)"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402
import e8p9_jobs as J9  # noqa: E402
from e1_jobs import ENVS  # noqa: E402
LISTS = {"en": "data/lists/en_all", "zh": "data/lists/zh_all"}  # as e10_objq_jobs (that module needs pandas)
TEACHERS = ["e82_cv2_all_n10_def", "e81_cv3_n10_def", "e81_idx_n25_def"]

ROOT = J9.ROOT
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
for c in TEACHERS:
    for k in range(1, 8):
        for s in ("en", "zh"):
            d = J9.wav_dir(c, k, s)
            f = d / "_metrics" / "metrics.csv"
            if not d.exists() or (f.exists() and "utmos" in f.read_text(encoding="utf-8").splitlines()[0].split(",")):
                continue
            gpu_lock.wait_free("heavy_gen")
            with gpu_lock.Lock("eval_job"):
                rc = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", LISTS[s] + ".meta.lst",
                                      "--wav_dir", str(d), "--lang", s, "--metrics", "utmos"], cwd=ROOT, env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(c, k, s, rc, flush=True)
print("done", flush=True)
