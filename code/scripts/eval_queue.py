"""Evaluation queue: polls generation outputs and evaluates each finished (job, set) once.
Runs alongside gen_queue.py (one eval process at a time). Exits when every job is evaluated."""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS, ROOT, jobs  # noqa: E402

LOG = ROOT / "logs" / "E1"
LOG.mkdir(parents=True, exist_ok=True)


def pending():
    for j in jobs():
        if not j["out"]:
            continue
        for s, meta, lang in j["sets"]:
            o = ROOT / j["out"] / s
            if not (o / "_evaldone").exists():
                yield j, s, meta, lang, o


def main():
    fails = {}
    while True:
        todo = [t for t in pending() if fails.get(str(t[4]), 0) < 2]
        if not todo:
            print("all evaluated" + (f" (failed: {list(fails)})" if fails else ""), flush=True)
            return
        ran = False
        for j, s, meta, lang, o in todo:
            if not (o / "_done").exists():
                continue
            env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
            with open(LOG / f"eval_{j['name']}_{s}.log", "a", encoding="utf-8") as f:
                rc = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", meta + ".meta.lst",
                                      "--wav_dir", str(o), "--lang", lang], cwd=ROOT, env=env,
                                     stdout=f, stderr=subprocess.STDOUT)
            print(f"[{time.strftime('%H:%M:%S')}] eval {j['name']} {s} rc={rc}", flush=True)
            if rc == 0:
                (o / "_evaldone").write_text("ok")
            else:
                fails[str(o)] = fails.get(str(o), 0) + 1
            ran = True
            break  # re-scan so earlier jobs get priority
        if not ran:
            time.sleep(60)


if __name__ == "__main__":
    main()
