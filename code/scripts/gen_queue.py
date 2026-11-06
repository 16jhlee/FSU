"""Sequential generation queue for E1. Resumable: a (job, set) is skipped when its _done marker
exists; every runner also skips existing wavs. Run with the tools env python."""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS, ROOT, jobs, n_lines  # noqa: E402

LOG = ROOT / "logs" / "E1"
LOG.mkdir(parents=True, exist_ok=True)
only = sys.argv[1:]  # optional job-name filter


def main():
    for j in jobs():
        if only and not any(j["name"].startswith(p) for p in only):
            continue
        for s, meta, _ in j["sets"]:
            if j["out"]:
                o = f"{j['out']}/{s}"
                done = ROOT / o / "_done"
            else:
                o = None
                done = ROOT / "cache" / "_done" / f"{j['name']}_{s}"
            if done.exists():
                continue
            cmd = [ENVS[j["env"]]] + j["cmd_fn"](s, meta, o)
            # PYTHONUTF8: upstream scripts call open() without encoding (Windows default cp949)
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1",
                       **j["extra_env"])
            t0 = time.time()
            with open(LOG / f"gen_{j['name']}_{s}.log", "a", encoding="utf-8") as f:
                rc = subprocess.call(cmd, cwd=j["cwd"], env=env, stdout=f, stderr=subprocess.STDOUT)
            ok = rc == 0
            if ok and o:
                n = len(list((ROOT / o).glob("*.wav")))
                ok = n == n_lines(meta)
                msg = f"{n}/{n_lines(meta)} wavs"
            else:
                msg = ""
            print(f"[{time.strftime('%H:%M:%S')}] {j['name']} {s} rc={rc} {msg} {time.time() - t0:.0f}s", flush=True)
            if ok:
                done.parent.mkdir(parents=True, exist_ok=True)
                done.write_text("ok")


if __name__ == "__main__":
    main()
