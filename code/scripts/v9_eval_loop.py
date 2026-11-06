"""v9 second GPU line: evaluate newly completed folders while the generation line runs.
python scripts/v9_eval_loop.py <gen_pid> [<gen_pid> ...]
Every 20 min (and once more after all generation PIDs exit): image eval (env_dom) -> CLAP (env_dom) -> toolkit (env_aeval).
Each stage skips folders it already scored, so a pass over nothing new is cheap. Output appended to logs/v9_eval_loop.log.
"""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gpu_lock  # noqa: E402
PY_DOM = r"C:\Users\Rose\miniconda3\envs\env_dom\python.exe"
PY_AEV = r"C:\Users\Rose\miniconda3\envs\env_aeval\python.exe"
LOG = ROOT / "logs" / "v9_eval_loop.log"
PY_TOOLS = r"C:\Users\Rose\miniconda3\envs\tools\python.exe"
STAGES = [(PY_DOM, ["scripts/v9_image_eval.py"]), (PY_DOM, ["scripts/v9_audio_eval.py", "clap"]),
          (PY_AEV, ["scripts/v9_audio_eval.py", "toolkit"]),
          (PY_TOOLS, ["scripts/queue_eval.py", "e8p10_jobs"])]  # E8'-10 speech (one pass: QUEUE_EVAL_ONEPASS)


def alive(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259


def run_pass():
    env = dict(os.environ, PYTHONUTF8="1", QUEUE_EVAL_ONEPASS="1")
    env["PATH"] = r"C:\Users\Rose\miniconda3\envs\env_aeval\Library\bin;" + env["PATH"]
    with open(LOG, "a", encoding="utf-8") as f:
        for py, args in STAGES:
            f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] start {' '.join(args)}\n")
            f.flush()
            if args[0].endswith("queue_eval.py"):  # takes the locks itself, per item
                rc = subprocess.call([py, "-u", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            else:  # never overlap IndexTTS-2 generation (GPU memory); one evaluation job at a time
                gpu_lock.wait_free("heavy_gen")
                with gpu_lock.Lock("eval_job"):
                    rc = subprocess.call([py, "-u", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            f.write(f"[{time.strftime('%m-%d %H:%M:%S')}] rc={rc} {' '.join(args)}\n")
            f.flush()


pids = [int(p) for p in sys.argv[1:]]
while any(alive(p) for p in pids):
    run_pass()
    time.sleep(1200)
run_pass()
