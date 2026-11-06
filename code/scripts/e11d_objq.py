"""E11-D / E11-L objective quality (E10-1′ pipeline, as e11_objq.py): DNSMOS P.835 on CPU at once; UTMOSv2 + NISQA on GPU
inside the eval lock, started after the E12 evaluation queue (queue_eval e12_jobs) has exited so that it does not hold
the lock for hours ahead of it. Every seed-0 d_* / l_* folder (resolution eps reused from E11, results/E11_objq_eps.csv).
python scripts/e11d_objq.py   (tools env, detached)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402
import e11d_jobs as J  # noqa: E402

ROOT = J.ROOT
PY = str(Path(sys.executable).resolve().parents[1] / "env_objq" / "Scripts" / "python.exe")
jobs = []
for name, c in J.CONFIGS.items():
    if c.get("prep"):
        continue
    for s, m, _ in J.sets_for(name, 0):
        jobs.append((m + ".meta.lst", str(J.wav_dir(name, 0, s))))
jf = ROOT / "logs" / "e11d_objq_jobs.tsv"
jf.write_text("".join(f"{m}\t{d}\n" for m, d in jobs), encoding="utf-8")
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
cenv = dict(env, CUDA_VISIBLE_DEVICES="")
log = open(ROOT / "logs" / "e11d_objq.log", "a", encoding="utf-8")
cpu = subprocess.Popen([PY, "scripts/objq_metrics.py", "--jobs", str(jf), "--metrics", "dnsmos"], cwd=ROOT, env=cenv,
                       stdout=log, stderr=subprocess.STDOUT)
chk = "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*queue_eval.py e12_jobs*' }) { exit 1 } else { exit 0 }"
while subprocess.call(["powershell", "-NoProfile", "-Command", chk]):
    time.sleep(60)
gpu_lock.wait_free("heavy_gen")
with gpu_lock.Lock("eval_job"):
    rc = subprocess.call([PY, "scripts/objq_metrics.py", "--jobs", str(jf), "--metrics", "utmosv2,nisqa"], cwd=ROOT, env=env,
                         stdout=log, stderr=subprocess.STDOUT)
print("gpu rc", rc, "cpu rc", cpu.wait(), len(jobs), "folders", flush=True)
