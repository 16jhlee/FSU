"""E3 evaluation queue: embeddings (WavLM, ECAPA, ERes2Net-en; raw vectors saved by eval_metrics) and
acoustic diagnostics for every finished (config, seed, set). Runs alongside e3_gen.py; exits when complete."""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS  # noqa: E402
from e3_jobs import CONFIGS, K, ROOT, SETS, wav_dir  # noqa: E402

LOG = ROOT / "logs" / "E3"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
fails = {}


def todo():
    for k in K:
        for name in CONFIGS:
            for s, meta, lang in SETS:
                d = wav_dir(name, k, s)
                if not (d / "_e3done").exists() and fails.get(str(d), 0) < 2:
                    yield name, k, s, meta, lang, d


while True:
    items = list(todo())
    if not items:
        print("all E3 evaluated", fails or "", flush=True)
        break
    ran = False
    for name, k, s, meta, lang, d in items:
        ready = (d / "_done").exists() or (k == 0 and (d / "_evaldone").exists())
        if not ready:
            continue
        with open(LOG / f"eval_{name}_k{k}_{s}.log", "a", encoding="utf-8") as f:
            rc1 = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", meta + ".meta.lst",
                                   "--wav_dir", str(d), "--lang", lang, "--metrics", "wavlm,ecapa,eres2net"],
                                  cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            rc2 = subprocess.call([ENVS["env_eval"], "scripts/e3_acoustic.py", meta + ".meta.lst", str(d), lang],
                                  cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] eval {name} k{k} {s} rc={rc1},{rc2}", flush=True)
        if rc1 == 0 and rc2 == 0:
            (d / "_e3done").write_text("ok")
        else:
            fails[str(d)] = fails.get(str(d), 0) + 1
        ran = True
        break
    if not ran:
        time.sleep(60)
