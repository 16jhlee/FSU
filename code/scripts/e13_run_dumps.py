"""E13 (v10.5) GPU driver for the dump measurements (E13-4, E13-2 dumps): one process at a time, after queue_gen e13_jobs.
IndexTTS-2 jobs take the heavy_gen lock and wait for a running evaluation job (as queue_gen). Skip-existing on the
output CSV; logs/E13_<tag>.log.   python scripts/e13_run_dumps.py   (tools env, detached)"""
import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E = Path(sys.executable).resolve().parents[1]
JOBS = [  # tag, env, args, output, extra env
    ("rms_cv2", "env_cosy", ["scripts/e13_rms.py", "cv2"], "results/E13_rms_cv2.csv", {}),
    ("rms_cv3", "env_cosy", ["scripts/e13_rms.py", "cv3"], "results/E13_rms_cv3.csv", {}),
    ("rms_glm", "env_glm", ["scripts/e13_rms.py", "glm"], "results/E13_rms_glm.csv", {}),
    ("rms_idx", "env_index", ["scripts/e13_rms.py", "idx"], "results/E13_rms_idx.csv", {}),
]
only = sys.argv[1:]
while subprocess.call(["powershell", "-NoProfile", "-Command",
                       "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*queue_gen.py e13_jobs*' }) { exit 1 } else { exit 0 }"]):
    time.sleep(60)
for tag, env_name, args, out, extra in JOBS:
    if (only and tag not in only) or (ROOT / out).exists():
        continue
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", **extra)
    heavy = env_name == "env_index"
    t0 = time.time()
    with gpu_lock.Lock("heavy_gen") if heavy else contextlib.nullcontext():
        if heavy:
            gpu_lock.wait_free("eval_job")
        with open(ROOT / "logs" / f"E13_{tag}.log", "w", encoding="utf-8") as f:
            rc = subprocess.call([str(E / env_name / "python.exe"), *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {tag} rc={rc} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
