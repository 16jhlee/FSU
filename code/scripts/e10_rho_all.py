"""E10-7 driver: run scripts/e10_rho.py for the five models one after another (one GPU process at a time).
python scripts/e10_rho_all.py   (tools env)"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = Path(sys.executable).resolve().parents[1]
JOBS = [("cv2", "env_cosy"), ("cv3", "env_cosy"), ("idx", "env_index"), ("f5", "env_f5dmo"), ("zip", "env_zip")]
for key, env_name in JOBS:
    if (ROOT / f"results/E10_rho_{key}.csv").exists() and sum(1 for _ in open(ROOT / f"results/E10_rho_{key}.csv")) > 150:
        continue
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if key == "zip":
        env["PYTHONPATH"] = str(ROOT / "repos" / "ZipVoice")
    t0 = time.time()
    with open(ROOT / "logs" / f"E10_rho_{key}.log", "w", encoding="utf-8") as f:
        rc = subprocess.call([str(E / env_name / "python.exe"), "scripts/e10_rho.py", key, "100"], cwd=ROOT, env=env,
                             stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {key} rc={rc} {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
