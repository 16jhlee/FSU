"""E12 (v10.4) GPU driver: one measurement process at a time, after the E12 generation queue (queue_gen e12_jobs).
IndexTTS-2 jobs take the heavy_gen lock and wait for a running evaluation job (as queue_gen). Skip-existing on the
output CSV; logs/E12_<tag>.log.   python scripts/e12_run_all.py   (tools env, detached)"""
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
RK = {"E10_RHO_OUT": "E12_rk"}
JOBS = [  # tag, env, args, output, extra env
    ("direction", "env_cosy", ["scripts/e12_direction.py", "100"], "results/E12_direction.csv", {}),
    ("norm_cv2", "env_cosy", ["scripts/e12_norm.py", "cv2", "100"], "results/E12_norm_cv2.csv", {}),
    ("norm_cv3", "env_cosy", ["scripts/e12_norm.py", "cv3", "100"], "results/E12_norm_cv3.csv", {}),
    ("lip", "env_cosy", ["scripts/e12_lip.py", "50"], "results/E12_lip.csv", {}),
    ("rk_cv2", "env_cosy", ["scripts/e10_rho.py", "cv2", "50"], "results/E12_rk_cv2.csv", dict(RK, E10_RHO_KT="32")),
    ("rk_cv3", "env_cosy", ["scripts/e10_rho.py", "cv3", "50"], "results/E12_rk_cv3.csv", RK),
    ("rk_f5", "env_f5dmo", ["scripts/e10_rho.py", "f5", "50"], "results/E12_rk_f5.csv", RK),
    ("rk_zip", "env_zip", ["scripts/e10_rho.py", "zip", "50"], "results/E12_rk_zip.csv", dict(RK, PYTHONPATH=str(ROOT / "repos" / "ZipVoice"))),
    ("norm_idx", "env_index", ["scripts/e12_norm.py", "idx", "100"], "results/E12_norm_idx.csv", {}),
    ("rk_idx", "env_index", ["scripts/e10_rho.py", "idx", "50"], "results/E12_rk_idx.csv", RK),
]
only = sys.argv[1:]
while subprocess.call(["powershell", "-NoProfile", "-Command",
                       "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*queue_gen.py e12_jobs*' }) { exit 1 } else { exit 0 }"]):
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
        with open(ROOT / "logs" / f"E12_{tag}.log", "w", encoding="utf-8") as f:
            rc = subprocess.call([str(E / env_name / "python.exe"), *args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {tag} rc={rc} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
