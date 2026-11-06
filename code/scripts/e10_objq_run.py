"""E10-1′ driver: objective-quality re-scoring of every folder in e10_objq_jobs.jobs() (priority: Table 1, ε teacher seeds,
Table 9, Table 7′, v8 tables, then E10 rows as they finish). Resumable; exits when everything incl. E10 is scored.
  python scripts/e10_objq_run.py gpu   UTMOSv2 + NISQA, chunks of folders inside the evaluation GPU lock (gpu_lock
                                       'eval_job', never overlaps IndexTTS-2 generation) -> at most 2 GPU processes
  python scripts/e10_objq_run.py cpu   DNSMOS P.835 on CPU (CUDA hidden), no lock
(tools env; runs scripts/objq_metrics.py in env_objq)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu_lock  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = str(Path(sys.executable).resolve().parents[1] / "env_objq" / "Scripts" / "python.exe")
MODE = sys.argv[1]
METRICS = {"gpu": "utmosv2,nisqa", "cpu": "dnsmos"}[MODE]
FILES = {"gpu": ["objq.csv"], "cpu": ["objq_dnsmos.csv"]}[MODE]
NEED = {"gpu": ["utmosv2", "nisqa_tts"], "cpu": ["dnsmos_ovrl"]}[MODE]
PRIO = {"T1": 0, "eps": 1, "T9": 2, "T7": 3, "v8seed": 4, "v8stream": 4, "v8robust": 4, "E10full": 5, "E10fill": 5}
CHUNK_WAVS = 1500  # smaller chunks: the process exits (frees all GPU memory) more often
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
if MODE == "cpu":
    env["CUDA_VISIBLE_DEVICES"] = ""
LOG = ROOT / "logs" / "e10_objq"
LOG.mkdir(parents=True, exist_ok=True)


def listing():
    r = subprocess.run([str(Path(sys.executable).resolve().parents[1] / "env_eval" / "python.exe"), "-c",
                        "import sys; sys.path.insert(0,'scripts'); import e10_objq_jobs as O\n"
                        "for m,d,t in O.jobs(): print(m+'\\t'+d+'\\t'+t)"], cwd=ROOT, capture_output=True, text=True, env=env)
    J = [tuple(l.split("\t")) for l in r.stdout.splitlines() if l.count("\t") == 2]
    return sorted(J, key=lambda x: PRIO.get(x[2], 9))


def done(meta, d):
    f = Path(d) / "_metrics" / FILES[0]
    if not f.exists():
        return False
    head = f.read_text(encoding="utf-8").splitlines()[0].split(",")
    return all(c in head for c in NEED)


def ready(d, tag):
    d = Path(d)
    if not (d / "_done").exists():
        return False
    return not tag.startswith("E10") or (d / "_q10").exists()  # E10 rows: after their own evaluation


def nlines(meta):
    return sum(1 for l in (ROOT / meta).read_text(encoding="utf-8").splitlines() if l.strip())


while True:
    J = listing()
    todo = [(m, d, t) for m, d, t in J if not done(m, d)]
    import e10_jobs as JE  # noqa: E402  (E10 completion: every full / fill folder generated and evaluated)
    e10_left = any(not (JE.wav_dir(n, k, s) / "_q10").exists() for n, c in JE.CONFIGS.items()
                   if JE.BLOCK.get(n) in ("full", "fill") and not c.get("prep") and not c.get("evalonly")
                   for k in c["K"] if k <= 2 for s, _, _ in JE.sets_for(n, k))
    if not todo and not e10_left:
        print("all scored", flush=True)
        break
    run = [(m, d, t) for m, d, t in todo if ready(d, t)]
    if not run:
        time.sleep(600)
        continue
    chunk, n = [], 0
    for m, d, t in run:
        chunk.append((m, d, t))
        n += nlines(m)
        if n >= CHUNK_WAVS:
            break
    jf = LOG / f"chunk_{MODE}.tsv"
    jf.write_text("".join(f"{m}\t{d}\n" for m, d, _ in chunk), encoding="utf-8")
    t0 = time.time()
    with open(LOG / f"{MODE}.log", "a", encoding="utf-8") as f:
        if MODE == "gpu":
            gpu_lock.wait_free("heavy_gen")
            # after generation has finished the GPU holds only queue_eval -> OBJQ_LOCK=objq_job lets the re-scoring
            # run beside it (still 2 GPU processes; UTMOSv2/NISQA need ~3 GB)
            with gpu_lock.Lock(os.environ.get("OBJQ_LOCK", "eval_job")):
                rc = subprocess.call([PY, "scripts/objq_metrics.py", "--jobs", str(jf), "--metrics", METRICS], cwd=ROOT,
                                     env=env, stdout=f, stderr=subprocess.STDOUT)
        else:
            rc = subprocess.call([PY, "scripts/objq_metrics.py", "--jobs", str(jf), "--metrics", METRICS], cwd=ROOT,
                                 env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {MODE} {len(chunk)} folders {n} wavs rc={rc} {time.time() - t0:.0f}s "
          f"({chunk[0][2]}..{chunk[-1][2]}); left {len(todo) - len(chunk)}", flush=True)
    if rc != 0:
        time.sleep(60)
    elif MODE == "gpu":
        time.sleep(45)  # yield the eval lock so queue_eval (polls every 20 s) gets a turn between chunks
