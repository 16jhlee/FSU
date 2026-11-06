"""E10: embeddings-only evaluation (seeds >= 1 of E10-3 / E10-4, metrics_for() = 'wavlm,ecapa,eres2net') for many folders
with each embedder loaded once (eval_metrics.py reloads the three embedders per 100-utterance folder). The per-folder
computation is eval_metrics.py's embedding block verbatim (same classes, same full-wav + 6 s centre-crop SIMs, same
emb_<e>.npz / metrics.csv / prompt cache); e3_acoustic.py runs in the background on CPU; '_q10' is written when both
are done, exactly like queue_eval.py. Batches of BATCH folders inside the evaluation GPU lock.
python scripts/e10_bulk_emb.py   (tools env; runs itself in env_eval for each batch)
E13: BULK_JOBS=e13_jobs selects another jobs module; the done marker is that module's QMARK (default _qdone).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = 25


def todo():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib
    J = importlib.import_module(os.environ.get("BULK_JOBS", "e10_jobs"))
    out = []
    for name, c in J.CONFIGS.items():
        if c.get("prep") or c.get("evalonly"):
            continue
        for k in c["K"]:
            for s, meta, lang in J.sets_for(name, k):
                if J.metrics_for(name, k, s, lang) != "wavlm,ecapa,eres2net":
                    continue
                d = J.wav_dir(name, k, s)
                if (d / "_done").exists() and not (d / mark(J)).exists():
                    out.append((meta + ".meta.lst", str(d), lang))
    return out


def mark(J):
    return getattr(J, "QMARK", "_qdone")


def worker(jobfile):
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_metrics as EM
    jobs = [l.split("\t") for l in Path(jobfile).read_text(encoding="utf-8").splitlines() if l.strip()]
    acoustic = []
    states = {}
    for meta, wd, lang in jobs:
        wd = Path(wd)
        rows = [r for r in EM.read_meta(ROOT / meta) if (wd / f"{r['utt']}.wav").exists()]
        out = wd / "_metrics"
        out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"utt_id": [r["utt"] for r in rows]})
        f = out / "metrics.csv"
        if f.exists():
            old = pd.read_csv(f, encoding="utf-8")
            if list(old["utt_id"]) == list(df["utt_id"]):
                df = old
        df["dur_gen_s"] = [sf.info(str(wd / f"{r['utt']}.wav")).duration for r in rows]
        df["dur_prompt_s"] = [sf.info(r["prompt_wav"]).duration for r in rows]
        states[str(wd)] = (rows, df, out)
    cache = ROOT / "data" / "cache"
    for e in ["wavlm", "ecapa", "eres2net"]:
        emb = EM.EMBEDDERS[e]()
        pc_path = cache / f"prompt_{e}.npz"
        pc = dict(np.load(pc_path)) if pc_path.exists() else {}
        new_prompt = False
        for meta, wd, lang in jobs:
            rows, df, out = states[str(Path(wd))]
            if f"sim_{e}" in df:
                continue
            store, s_full, s_c6 = {}, [], []
            for r in rows:
                pk = os.path.normcase(os.path.abspath(r["prompt_wav"])).replace("\\", "/")
                if pk not in pc:
                    pc[pk] = emb(EM.load16(r["prompt_wav"]))
                    new_prompt = True
                w = EM.load16(str(Path(wd) / f"{r['utt']}.wav"))
                g, g6 = emb(w), emb(EM.centre_crop(w))
                store[f"gen/{r['utt']}"], store[f"gen_c6/{r['utt']}"] = g, g6
                s_full.append(EM.cos(g, pc[pk]))
                s_c6.append(EM.cos(g6, pc[pk]))
            df[f"sim_{e}"], df[f"sim_{e}_c6"] = s_full, s_c6
            np.savez(out / f"emb_{e}.npz", **store)
            df.to_csv(out / "metrics.csv", index=False, encoding="utf-8")
        if new_prompt:
            np.savez(pc_path, **pc)
        del emb
        torch.cuda.empty_cache()
    print("embeddings done", len(jobs), flush=True)


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import gpu_lock
    py_eval = str(Path(sys.executable).resolve().parents[1] / "env_eval" / "python.exe")
    log = ROOT / "logs" / "e10_bulk_emb"
    log.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    import importlib
    global MARK
    MARK = mark(importlib.import_module(os.environ.get("BULK_JOBS", "e10_jobs")))
    while True:
        J = todo()
        if not J:
            print("all embedding folders done", flush=True)
            break
        chunk = J[:BATCH]
        jf = log / "chunk.tsv"
        jf.write_text("".join("\t".join(x) + "\n" for x in chunk), encoding="utf-8")
        t0 = time.time()
        gpu_lock.wait_free("heavy_gen")
        with open(log / "worker.log", "a", encoding="utf-8") as f, gpu_lock.Lock("eval_job"):
            rc = subprocess.call([py_eval, __file__, "--worker", str(jf)], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] worker rc={rc}", flush=True)
            time.sleep(60)
            continue
        procs = []
        with open(log / "acoustic.log", "a", encoding="utf-8") as f:
            for meta, wd, lang in chunk:  # CPU only, outside the lock
                procs.append((wd, subprocess.Popen([py_eval, "scripts/e3_acoustic.py", meta, wd, lang], cwd=ROOT, env=env,
                                                   stdout=f, stderr=subprocess.STDOUT)))
                if len(procs) >= 4:
                    w_, p_ = procs.pop(0)
                    if p_.wait() == 0:
                        (Path(w_) / MARK).write_text("ok")
            for w_, p_ in procs:
                if p_.wait() == 0:
                    (Path(w_) / MARK).write_text("ok")
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {len(chunk)} folders {time.time() - t0:.0f}s; left {len(J) - len(chunk)}", flush=True)
        time.sleep(45)  # yield the lock


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        main()
