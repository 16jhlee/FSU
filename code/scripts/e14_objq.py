"""E14-1 objective quality (as e11d_objq.py): DNSMOS P.835 on CPU at once; UTMOSv2 (+ NISQA except Seed-TTS hard, where
NISQA fails above 13 s) on GPU inside the eval lock, interleaved with the E14 evaluation queue.
Folders: allnull_n2 / allnullR_n2 hard, allnull_midR en_full / zh_full / hard (seed 0).
python scripts/e14_objq.py   (tools env, detached)
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402
import e14_jobs as J  # noqa: E402

ROOT = J.ROOT
PY = str(Path(sys.executable).resolve().parents[1] / "env_objq" / "Scripts" / "python.exe")
jobs = [(m + ".meta.lst", str(J.wav_dir(n, 0, s)), s) for n in ("allnull_n2", "allnullR_n2", "allnull_midR")
        for s, m, _ in J.sets_for(n, 0)]
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
cenv = dict(env, CUDA_VISIBLE_DEVICES="")
log = open(ROOT / "logs" / "e14_objq.log", "a", encoding="utf-8")


def jobfile(tag, sel):
    jf = ROOT / "logs" / f"e14_objq_jobs_{tag}.tsv"
    jf.write_text("".join(f"{m}\t{d}\n" for m, d, s in sel), encoding="utf-8")
    return str(jf)


cpu = subprocess.Popen([PY, "scripts/objq_metrics.py", "--jobs", jobfile("all", jobs), "--metrics", "dnsmos"], cwd=ROOT,
                       env=cenv, stdout=log, stderr=subprocess.STDOUT)
rcs = []
for tag, sel, metrics in [("full", [j for j in jobs if j[2] != "hard"], "utmosv2,nisqa"),
                          ("hard", [j for j in jobs if j[2] == "hard"], "utmosv2")]:
    gpu_lock.wait_free("heavy_gen")
    with gpu_lock.Lock("eval_job"):
        rcs.append(subprocess.call([PY, "scripts/objq_metrics.py", "--jobs", jobfile(tag, sel), "--metrics", metrics],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
print("gpu rc", rcs, "cpu rc", cpu.wait(), len(jobs), "folders", flush=True)
