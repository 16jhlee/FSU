"""E11 objective quality (E10-1′ pipeline: UTMOSv2 + NISQA on GPU inside the eval lock, DNSMOS P.835 on CPU) for every
seed-0 E11 folder and the teacher seeds 1-7 (en/zh 100; resolution eps_utt with seed 0 restricted to the same 100).
python scripts/e11_objq.py   (tools env)
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402
import e11_jobs as J  # noqa: E402

ROOT = J.ROOT
PY = str(Path(sys.executable).resolve().parents[1] / "env_objq" / "Scripts" / "python.exe")
jobs = []
for name, c in J.CONFIGS.items():
    if c.get("prep"):
        continue
    for s, m, _ in J.sets_for(name, 0):
        jobs.append((m + ".meta.lst", str(J.wav_dir(name, 0, s))))
for k in range(1, 8):
    for s, m, _ in J.S100:
        jobs.append((m + ".meta.lst", str(J.wav_dir("g_base_n10", k, s))))
jf = ROOT / "logs" / "e11_objq_jobs.tsv"
jf.write_text("".join(f"{m}\t{d}\n" for m, d in jobs), encoding="utf-8")
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
cenv = dict(env, CUDA_VISIBLE_DEVICES="")
log = open(ROOT / "logs" / "e11_objq.log", "a", encoding="utf-8")
cpu = subprocess.Popen([PY, "scripts/objq_metrics.py", "--jobs", str(jf), "--metrics", "dnsmos"], cwd=ROOT, env=cenv,
                       stdout=log, stderr=subprocess.STDOUT)
gpu_lock.wait_free("heavy_gen")
with gpu_lock.Lock("eval_job"):
    rc = subprocess.call([PY, "scripts/objq_metrics.py", "--jobs", str(jf), "--metrics", "utmosv2,nisqa"], cwd=ROOT, env=env,
                         stdout=log, stderr=subprocess.STDOUT)
print("gpu rc", rc, "cpu rc", cpu.wait(), len(jobs), "folders", flush=True)
