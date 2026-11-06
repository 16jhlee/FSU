"""E5 evaluation queue (runs alongside e5_gen.py, one job at a time; exits when complete).
k=0: full E1 metrics (WER, WavLM/ECAPA/ERes2Net SIM + raw embeddings, UTMOS) + acoustic diagnostics.
k>=1: embeddings (3 main embedders) + acoustic diagnostics (E3 geometry)."""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS  # noqa: E402
from e5_jobs import CONFIGS, ROOT, SETS, wav_dir  # noqa: E402

LOG = ROOT / "logs" / "E5"
LOG.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
fails = {}


def todo():
    for k in range(8):
        for name, c in CONFIGS.items():
            if k not in c["K"]:
                continue
            for s, meta, lang in SETS:
                d = wav_dir(name, k, s)
                if not (d / "_e5done").exists() and fails.get(str(d), 0) < 2:
                    yield name, k, s, meta, lang, d


while True:
    items = list(todo())
    if not items:
        print("all E5 evaluated", fails or "", flush=True)
        break
    ran = False
    for name, k, s, meta, lang, d in items:
        if not (d / "_done").exists():
            continue
        metrics = "wer,wavlm,ecapa,eres2net,utmos" if k == 0 else "wavlm,ecapa,eres2net"
        with open(LOG / f"eval_{name}_k{k}_{s}.log", "a", encoding="utf-8") as f:
            rc1 = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", meta + ".meta.lst",
                                   "--wav_dir", str(d), "--lang", lang, "--metrics", metrics],
                                  cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            rc2 = subprocess.call([ENVS["env_eval"], "scripts/e3_acoustic.py", meta + ".meta.lst", str(d), lang],
                                  cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] eval {name} k{k} {s} rc={rc1},{rc2}", flush=True)
        if rc1 == 0 and rc2 == 0:
            (d / "_e5done").write_text("ok")
        else:
            fails[str(d)] = fails.get(str(d), 0) + 1
        ran = True
        break
    if not ran:
        time.sleep(60)
