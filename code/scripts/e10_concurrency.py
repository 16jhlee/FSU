"""E10-6: throughput under concurrency in the official CosyVoice2 streaming pipeline (v8 §1.4 settings: fp32, no TensorRT,
no vLLM, stream=True, one model instance). Concurrent requests = Python threads calling cv.inference_zero_shot on the same
model, as the official FastAPI/gRPC runtimes do (per-uuid state; the official LLM lock serialises LLM forward passes).
Conditions: teacher NFE 10, baseline NFE 2, R (ours) NFE 2. Concurrency c in {1, 4, 8, 16}: a queue of 50 requests
(en_100 utterances 0..49) served by c worker threads, after 5 sequential warm-up requests per condition.
token_hop_len is reset to 25 at the start of every request (as v8 §1.4; the official code grows this model-wide
attribute in place — under concurrency the reset can race with other requests, which is the official behaviour).
Per run: throughput = sum(audio seconds) / wall seconds; first-packet latency P50 / P99; mean request latency.
Timing pass (sequential, 20 requests): cuda-synchronised time of every LLM token step, flow.inference and hift.inference
(v8 §1.4 method; torch.profiler does not record ops issued from worker threads). All kernels of all requests share
the default stream, so decoder share of GPU busy time = (flow + hift) / (flow + hift + llm) at every concurrency.
env_cosy, idle GPU:  python scripts/e10_concurrency.py   -> results/E10_concurrency.csv, results/E10_concurrency_requests.csv
E14-9: E10C_CONDS="R (ours) NFE2|mid+R (ours)" E10C_OUT=E14_midR_throughput (conditions by name, '|'-separated).
"""
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: F401,E402
from run_cosy import AutoModel, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QUICK = "--quick" in sys.argv  # functionality test only (2 conditions x c in {1, 4}, 4 requests)
CONC = [1, 4] if QUICK else [1, 4, 8, 16]
N_REQ = 4 if QUICK else 50


def cos_sched(n):
    return (1 - torch.cos(torch.linspace(0, 1, n + 1) * 0.5 * torch.pi)).tolist()


def main():
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / "CosyVoice2-0.5B"))
    m = cv.model
    dec = m.flow.decoder

    # GPU-time attribution (v8 §1.4 method): cuda-synchronised wall time of flow.inference, hift.inference and of every
    # LLM token step, only during the sequential timing pass (torch.profiler does not record ops of worker threads).
    TIM = {"on": False, "flow": 0.0, "hift": 0.0, "llm": 0.0}

    def timed(obj, attr, key):
        f = getattr(obj, attr)

        def g(*a, **k):
            if not TIM["on"]:
                return f(*a, **k)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = f(*a, **k)
            torch.cuda.synchronize()
            TIM[key] += time.perf_counter() - t0
            return out
        setattr(obj, attr, g)
    timed(m.flow, "inference", "flow")
    timed(m.hift, "inference", "hift")
    orig_llm_inf = m.llm.inference

    def llm_inf(*a, **k):
        it = orig_llm_inf(*a, **k)
        while True:
            if TIM["on"]:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                tok = next(it)
            except StopIteration:
                return
            if TIM["on"]:
                torch.cuda.synchronize()
                TIM["llm"] += time.perf_counter() - t0
            yield tok
    m.llm.inference = llm_inf

    rows_meta = read_meta(ROOT / "data/lists/en_100.meta.lst")
    conds = {"teacher NFE10": dict(nfe=10), "baseline NFE2": dict(nfe=2), "R (ours) NFE2": dict(nfe=2, iv=[0.1, 1.0]),
             # E14-9: mid+R (ours) = one midpoint step (3 evaluations), unguided at t = 0; installed last (permanent solver)
             "mid+R (ours)": dict(nfe=1, iv=[0.1, 1.0], solver="midpoint")}
    sel = [c for c in os.environ.get("E10C_CONDS", "teacher NFE10|baseline NFE2|R (ours) NFE2").split("|")]
    conds = {k: conds[k] for k in sel}

    def one(i, rec):
        utt, ptext, pwav, text = rows_meta[i % 50]
        m.token_hop_len = 25
        t0 = time.perf_counter()
        first, audio = None, 0
        for chunk in cv.inference_zero_shot(text, ptext, pwav, stream=True):
            if first is None:
                first = time.perf_counter() - t0
            audio += chunk["tts_speech"].shape[1]
        rec.append(dict(utt=utt, first_packet_ms=first * 1000, latency_s=time.perf_counter() - t0,
                        audio_s=audio / cv.sample_rate))

    def run(c, n):
        q = queue.Queue()
        for i in range(n):
            q.put(i)
        rec, lock = [], threading.Lock()

        def worker():
            while True:
                try:
                    i = q.get_nowait()
                except queue.Empty:
                    return
                r = []
                one(i, r)
                with lock:
                    rec.extend(r)
        ths = [threading.Thread(target=worker) for _ in range(c)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        torch.cuda.synchronize()
        return rec, time.perf_counter() - t0

    summary, reqs = [], []
    for name, cd in conds.items():
        dec.inference_cfg_rate = 0.7
        dec.cfg_interval = cd.get("iv")
        dec.t_span_override = cos_sched(cd["nfe"])
        if cd.get("solver"):
            run_cosy.install_solver(dec, cd["solver"])
        run(1, 5)  # warm-up
        for c in CONC:
            rec, wall = run(c, N_REQ)
            fp = np.array([r["first_packet_ms"] for r in rec])
            aud = sum(r["audio_s"] for r in rec)
            summary.append(dict(cond=name, concurrency=c, n=len(rec), wall_s=wall, audio_s=aud, throughput=aud / wall,
                                first_p50_ms=np.percentile(fp, 50), first_p99_ms=np.percentile(fp, 99),
                                latency_mean_s=np.mean([r["latency_s"] for r in rec])))
            reqs += [dict(cond=name, concurrency=c, **r) for r in rec]
            print(summary[-1], flush=True)
        # sequential timing pass: per-request GPU work of each module. All requests' kernels run on the one default
        # stream, so the decoder share of GPU busy time is this work share at every concurrency (no overlap of work).
        TIM.update(on=True, flow=0.0, hift=0.0, llm=0.0)
        trec, _ = run(1, 4 if QUICK else 20)
        TIM["on"] = False
        tot = TIM["flow"] + TIM["hift"] + TIM["llm"]
        aud_t = sum(r["audio_s"] for r in trec)
        for s in summary:
            if s["cond"] == name:
                s.update(gpu_flow_s_per_audio_s=TIM["flow"] / aud_t, gpu_hift_s_per_audio_s=TIM["hift"] / aud_t,
                         gpu_llm_s_per_audio_s=TIM["llm"] / aud_t, decoder_share=(TIM["flow"] + TIM["hift"]) / tot)
        print(name, {k: round(v, 2) for k, v in TIM.items() if k != "on"}, flush=True)
    S = pd.DataFrame(summary)
    if QUICK:
        print(pd.DataFrame(summary).round(3).to_string())
        return
    out = os.environ.get("E10C_OUT", "E10_concurrency")
    S.to_csv(ROOT / f"results/{out}.csv", index=False)
    pd.DataFrame(reqs).to_csv(ROOT / f"results/{out}_requests.csv", index=False)
    pd.set_option("display.width", 250)
    print(torch.cuda.get_device_name(0), torch.__version__)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
