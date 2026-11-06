"""E5 generation queue: seed-major (k=0 for every config first, then k=1..7 for the 8-seed configs). Resumable."""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS  # noqa: E402
from e5_jobs import CONFIGS, ROOT, SETS, ZIPDIR, n_lines, wav_dir  # noqa: E402

LOG = ROOT / "logs" / "E5"
LOG.mkdir(parents=True, exist_ok=True)

for k in range(8):
    for name, c in CONFIGS.items():
        if k not in c["K"]:
            continue
        for s, meta, _ in SETS:
            o = wav_dir(name, k, s)
            if (o / "_done").exists():
                continue
            o.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
            cwd = ROOT
            if c["env"] == "env_zip":
                env["PYTHONPATH"], cwd = str(ZIPDIR), ZIPDIR
            out_arg = str(o.relative_to(ROOT)) if c["env"] == "env_cosy" else str(o)
            t0 = time.time()
            with open(LOG / f"gen_{name}_k{k}_{s}.log", "a", encoding="utf-8") as f:
                rc = subprocess.call([ENVS[c["env"]]] + c["cmd"](s, meta, k, out_arg), cwd=cwd, env=env,
                                     stdout=f, stderr=subprocess.STDOUT)
            n = len(list(o.glob("*.wav")))
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] {name} k{k} {s} rc={rc} {n} wavs {time.time() - t0:.0f}s",
                  flush=True)
            if rc == 0 and n == n_lines(meta):
                (o / "_done").write_text("ok")
