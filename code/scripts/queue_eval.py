"""Generic evaluation queue: python queue_eval.py <jobs_module>
k=0: full metrics (WER/CER, SIM x3 + raw embeddings, UTMOS) + acoustic; k>=1: embeddings + acoustic.
Runs alongside queue_gen.py (one eval process at a time), exits when everything is evaluated.
Language per set comes from the jobs module's set tuples (set, meta, lang)."""
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
ONLY = sys.argv[2:]  # optional config-name prefixes (E13: run beside the bulk embedding pass without sharing folders)
ROOT = J.ROOT
LOG = ROOT / "logs" / sys.argv[1]
LOG.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
MARK = getattr(J, "QMARK", "_qdone")  # E10: own marker, so re-evaluated alias folders get the extra metrics
# E15: a second evaluation queue on disjoint config prefixes may use its own lock name (GPU limit 2 processes: no generation runs)
EVAL_LOCK = os.environ.get("QUEUE_EVAL_LOCK", "eval_job")
fails = {}
# QUEUE_EVAL_ASYNC_ACOUSTIC=1 (E10): e3_acoustic.py is CPU-only -> run it after the GPU lock is released, in the
# background (at most ACOUSTIC_MAX at once); the folder is marked done when both steps succeeded.
ASYNC = bool(os.environ.get("QUEUE_EVAL_ASYNC_ACOUSTIC"))
ACOUSTIC_MAX = 3
pending = {}  # str(dir) -> (Popen, log file, rc1, name, k, s)


def reap(block=False):
    for key, (p, f, rc1, name, k, s) in list(pending.items()):
        if block:
            p.wait()
        if p.poll() is None:
            continue
        f.close()
        rc2 = p.returncode
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] eval {name} k{k} {s} rc={rc1},{rc2}", flush=True)
        if rc1 == 0 and rc2 == 0:
            (Path(key) / MARK).write_text("ok")
        else:
            fails[key] = fails.get(key, 0) + 1
        del pending[key]


def todo():
    for k in range(8):
        for name, c in J.CONFIGS.items():
            if k not in c["K"] or c.get("prep") or (ONLY and not name.startswith(tuple(ONLY))):
                continue
            for s, meta, lang in J.sets_for(name, k):
                d = J.wav_dir(name, k, s)
                if not (d / MARK).exists() and fails.get(str(d), 0) < 2 and str(d) not in pending:
                    yield name, k, s, meta, lang, d


while True:
    reap()
    items = list(todo())
    if not items:
        if pending:
            reap(block=True)
            continue
        print("all evaluated", fails or "", flush=True)
        break
    ran = False
    for name, k, s, meta, lang, d in items:
        if not (d / "_done").exists():
            continue
        if ASYNC:
            while len(pending) >= ACOUSTIC_MAX:
                time.sleep(10)
                reap()
            m_ = J.metrics_for(name, k, s, lang) if hasattr(J, "metrics_for") else \
                ("wer,wavlm,ecapa,eres2net,utmos" if k == 0 else "wavlm,ecapa,eres2net")
            gpu_lock.wait_free("heavy_gen")
            f = open(LOG / f"eval_{name}_k{k}_{s}.log", "a", encoding="utf-8")
            with gpu_lock.Lock(EVAL_LOCK):
                ev = getattr(J, "eval_meta", lambda x: x)(meta)
                rc1 = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", ev + ".meta.lst",
                                       "--wav_dir", str(d), "--lang", lang, "--metrics", m_],
                                      cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            p = subprocess.Popen([ENVS["env_eval"], "scripts/e3_acoustic.py", meta + ".meta.lst", str(d), lang],
                                 cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            pending[str(d)] = (p, f, rc1, name, k, s)
            ran = True
            time.sleep(int(os.environ.get("QUEUE_EVAL_YIELD", "0")))
            break
        metrics = "wer,wavlm,ecapa,eres2net,utmos" if k == 0 else "wavlm,ecapa,eres2net"
        if k == 0 and lang in ("ko", "ja"):
            metrics += ",cer2"  # E8'-5: second ASR (SenseVoice)
        if hasattr(J, "metrics_for"):  # E10: per-config metric set
            metrics = J.metrics_for(name, k, s, lang)
        gpu_lock.wait_free("heavy_gen")  # never overlap IndexTTS-2 generation (GPU memory)
        with open(LOG / f"eval_{name}_k{k}_{s}.log", "a", encoding="utf-8") as f, gpu_lock.Lock(EVAL_LOCK):
            ev = getattr(J, "eval_meta", lambda m_: m_)(meta)  # v8 §2.3: SIM reference list may differ from the gen list
            rc1 = subprocess.call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", ev + ".meta.lst",
                                   "--wav_dir", str(d), "--lang", lang, "--metrics", metrics],
                                  cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            rc2 = subprocess.call([ENVS["env_eval"], "scripts/e3_acoustic.py", meta + ".meta.lst", str(d), lang],
                                  cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] eval {name} k{k} {s} rc={rc1},{rc2}", flush=True)
        if rc1 == 0 and rc2 == 0:
            (d / MARK).write_text("ok")
        else:
            fails[str(d)] = fails.get(str(d), 0) + 1
        ran = True
        time.sleep(int(os.environ.get("QUEUE_EVAL_YIELD", "0")))  # E10: let another eval-lock user take a turn
        break
    if not ran:
        if os.environ.get("QUEUE_EVAL_ONEPASS"):  # v9 eval loop: evaluate what is ready, then return
            print("one pass done", flush=True)
            break
        time.sleep(60)
