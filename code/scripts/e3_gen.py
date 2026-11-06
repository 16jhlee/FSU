"""E3 generation queue: seed-major order (k=1 for every config, then k=2, ...), resumable."""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS  # noqa: E402
from e3_jobs import CONFIGS, K, ROOT, SETS, ZIPDIR, n_lines, wav_dir  # noqa: E402

LOG = ROOT / "logs" / "E3"
LOG.mkdir(parents=True, exist_ok=True)

for k in K[1:]:
    for name, c in CONFIGS.items():
        for s, meta, _ in SETS:
            o = wav_dir(name, k, s)
            if (o / "_done").exists():
                continue
            cwd = ZIPDIR if c["env"] == "env_zip" else ROOT
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
            if c["env"] == "env_zip":
                env["PYTHONPATH"] = str(ZIPDIR)
            out_arg = str(o) if c["env"] != "env_cosy" else str(o.relative_to(ROOT))
            t0 = time.time()
            with open(LOG / f"gen_{name}_k{k}_{s}.log", "a", encoding="utf-8") as f:
                rc = subprocess.call([ENVS[c["env"]]] + c["cmd"](s, meta, k, out_arg), cwd=cwd, env=env,
                                     stdout=f, stderr=subprocess.STDOUT)
            n = len(list(o.glob("*.wav")))
            ok = rc == 0 and n == n_lines(meta)
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] {name} k{k} {s} rc={rc} {n} wavs {time.time() - t0:.0f}s",
                  flush=True)
            if ok:
                (o / "_done").write_text("ok")
