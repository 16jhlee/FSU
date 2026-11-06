"""Generic generation queue: python queue_gen.py <jobs_module> [name_prefix ...]
Seed-major, resumable (_done markers), one GPU job at a time. The jobs module must define
CONFIGS{name: dict(env, K, cmd(s, meta, k, out), sets)}, wav_dir(name,k,s), sets_for(name,k), n_lines(meta), ROOT."""
import contextlib
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402
from e1_jobs import ENVS  # noqa: E402

J = importlib.import_module(sys.argv[1])
only = sys.argv[2:]
ROOT = J.ROOT
LOG = ROOT / "logs" / sys.argv[1]
LOG.mkdir(parents=True, exist_ok=True)
ZIPDIR = ROOT / "repos" / "ZipVoice"
for k in range(8):
    for name, c in J.CONFIGS.items():
        if k not in c["K"] or (only and not any(name.startswith(p) for p in only)):
            continue
        for s, meta, _ in J.sets_for(name, k):
            o = J.wav_dir(name, k, s)
            if (o / "_done").exists():
                continue
            o.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
            cwd = ROOT
            if c["env"] == "env_zip":
                env["PYTHONPATH"], cwd = str(ZIPDIR), ZIPDIR
            out_arg = str(o.relative_to(ROOT)) if c["env"] == "env_cosy" else str(o)
            py = ENVS.get(c["env"]) or ENVS["env_cosy"].replace("env_cosy", c["env"])
            heavy = c["env"] == "env_index" or c.get("heavy", False)  # IndexTTS-2 / GLM-TTS LLM never overlap an evaluation job (GPU memory)
            with open(LOG / f"gen_{name}_k{k}_{s}.log", "a", encoding="utf-8") as f, \
                    (gpu_lock.Lock("heavy_gen") if heavy else contextlib.nullcontext()):
                if heavy:  # announce first, then wait for a running evaluation to finish
                    gpu_lock.wait_free("eval_job")
                t0 = time.time()
                rc = subprocess.call([py] + c["cmd"](s, meta, k, out_arg), cwd=cwd, env=env,
                                     stdout=f, stderr=subprocess.STDOUT)
            n = len(list(o.glob("*.wav")))
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] {name} k{k} {s} rc={rc} {n} wavs {time.time() - t0:.0f}s",
                  flush=True)
            if rc == 0 and (c.get("prep") or n == J.n_lines(meta)):
                (o / "_done").write_text("ok")
