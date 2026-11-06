"""Latency and throughput of the official CosyVoice 2 streaming pipeline (environment env_cosy; run on an otherwise idle
GPU). Official settings: fp32, no TensorRT, no vLLM, stream=True, one model instance; token_hop_len is reset to 25 at
the start of every request (the official code grows this model-wide attribute in place and never resets it).
Conditions: teacher (10 steps), official 2 steps, FSU (2 steps, guidance only for steps starting in [0.1, 1]) and FSU-Mid
(one midpoint step, 3 evaluations, unguided at t = 0).

    python -m fsu.probes.timing latency
    python -m fsu.probes.timing concurrency [--quick]
    python -m fsu.probes.timing concurrency_fsu_mid
"""
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from fsu.common import CKPT
from fsu.probes import main_for, utterance_rows

CONDITIONS = {"teacher NFE10": dict(nfe=10), "baseline NFE2": dict(nfe=2), "R (ours) NFE2": dict(nfe=2, iv=[0.1, 1.0]),
              "mid+R (ours)": dict(nfe=1, iv=[0.1, 1.0], solver="midpoint")}
EVALUATIONS = {"teacher NFE10": 20, "baseline NFE2": 4, "R (ours) NFE2": 3, "mid+R (ours)": 3}


def cos_sched(n):
    return (1 - torch.cos(torch.linspace(0, 1, n + 1) * 0.5 * torch.pi)).tolist()


class StreamingTiming:
    """The official CosyVoice 2 model with timing hooks on flow.inference, hift.inference and the LLM token stream."""

    def __init__(self, decoder="cosyvoice2"):
        from fsu.decoders.cosyvoice import CosyVoice
        self.runner_cls = CosyVoice
        self.cv = CosyVoice(SimpleNamespace(model_dir=str(CKPT / "CosyVoice2-0.5B"))).cv
        self.m = self.cv.model
        self.dec = self.m.flow.decoder

    def latency(self, n_utt=None, quick=False):
        """50 requests per condition (list utterances 0..49) after 5 warm-up requests. Per request: first-packet latency
        (wall clock from request to the first audio chunk), flow and HiFT time up to the first chunk, LLM time to the
        first-chunk tokens, total compute, audio seconds; decoder RTF, GPU-busy seconds (LLM + flow + HiFT, cuda
        synchronised) and real-time sessions per GPU = audio seconds / GPU-busy seconds. (quick has no effect here.)"""
        import pandas as pd
        cv, m, dec = self.cv, self.m, self.dec
        orig_solve = dec.solve_euler
        timers = {"flow": 0.0, "hift": 0.0}
        tok_times = []

        def timed(obj, attr, key):
            f = getattr(obj, attr)

            def g(*a, **k):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = f(*a, **k)
                torch.cuda.synchronize()
                timers[key] += time.perf_counter() - t0
                return out
            setattr(obj, attr, g)
        timed(m.flow, "inference", "flow")
        timed(m.hift, "inference", "hift")
        orig_llm_inf = m.llm.inference

        def llm_inf(*a, **k):
            for tok in orig_llm_inf(*a, **k):
                tok_times.append(time.perf_counter())
                yield tok
        m.llm.inference = llm_inf
        rows_meta = utterance_rows()
        out = []
        for name, c in CONDITIONS.items():
            dec.solve_euler = orig_solve
            if c.get("solver"):
                self.runner_cls.install_solver(dec, c["solver"])
            dec.inference_cfg_rate = 0.7
            dec.cfg_interval = c.get("iv")
            dec.t_span_override = cos_sched(c["nfe"])
            for i in range(55):
                utt, ptext, pwav, text = rows_meta[i % 50]
                m.token_hop_len = 25
                timers.update(flow=0.0, hift=0.0)
                tok_times.clear()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                first, first_flow, first_hift, audio = None, None, None, 0
                for j, chunk in enumerate(cv.inference_zero_shot(text, ptext, pwav, stream=True)):
                    if first is None:
                        first = time.perf_counter() - t0
                        first_flow, first_hift = timers["flow"], timers["hift"]
                        n_first_tok = len(tok_times)
                        llm_first = (tok_times[-1] - t0) if tok_times else np.nan
                    audio += chunk["tts_speech"].shape[1]
                total = time.perf_counter() - t0
                if i < 5:
                    continue  # warm-up
                llm_total = (tok_times[-1] - t0) if tok_times else np.nan
                out.append(dict(cond=name, calls=EVALUATIONS[name], utt=utt, first_packet_ms=first * 1000,
                                first_flow_ms=first_flow * 1000, first_hift_ms=first_hift * 1000,
                                llm_first_chunk_ms=llm_first * 1000, n_first_tokens=n_first_tok, flow_s=timers["flow"],
                                hift_s=timers["hift"], llm_s=llm_total, total_s=total, audio_s=audio / cv.sample_rate))
            print(name, "done", flush=True)
        D = pd.DataFrame(out)
        D["decoder_rtf"] = D.flow_s / D.audio_s
        D["gpu_busy_s"] = D.flow_s + D.hift_s + D.llm_s
        D["sessions"] = D.audio_s / D.gpu_busy_s
        D["decoder_share_first"] = D.first_flow_ms / D.first_packet_ms
        print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
        print(D.groupby("cond", sort=False).mean(numeric_only=True).round(3).to_string())
        return {"": D}

    def concurrency(self, n_utt=None, quick=False, names=("teacher NFE10", "baseline NFE2", "R (ours) NFE2")):
        """Concurrent requests = Python threads calling inference_zero_shot on the one model (per-uuid state; the
        official LLM lock serialises LLM forward passes), as the official FastAPI / gRPC runtimes. Per condition: 5
        sequential warm-up requests, then for c in {1, 4, 8, 16} a queue of 50 requests (utterances 0..49) served by c
        worker threads: throughput = audio seconds / wall seconds, first-packet P50 / P99, mean request latency.
        Timing pass (sequential, 20 requests): cuda-synchronised time of every LLM token step, flow.inference and
        hift.inference; every request runs on the default stream, so the decoder share of GPU busy time
        (flow + hift) / (flow + hift + llm) holds at every concurrency. quick: c in {1, 4}, 4 requests, nothing
        written. The midpoint solver of FSU-Mid stays installed once set, so it must come last."""
        import pandas as pd
        cv, m, dec = self.cv, self.m, self.dec
        conc, n_req = ([1, 4], 4) if quick else ([1, 4, 8, 16], 50)
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
        rows_meta = utterance_rows()
        conds = {k: CONDITIONS[k] for k in names}

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
                self.runner_cls.install_solver(dec, cd["solver"])
            run(1, 5)  # warm-up
            for c in conc:
                rec, wall = run(c, n_req)
                fp = np.array([r["first_packet_ms"] for r in rec])
                aud = sum(r["audio_s"] for r in rec)
                summary.append(dict(cond=name, concurrency=c, n=len(rec), wall_s=wall, audio_s=aud, throughput=aud / wall,
                                    first_p50_ms=np.percentile(fp, 50), first_p99_ms=np.percentile(fp, 99),
                                    latency_mean_s=np.mean([r["latency_s"] for r in rec])))
                reqs += [dict(cond=name, concurrency=c, **r) for r in rec]
                print(summary[-1], flush=True)
            TIM.update(on=True, flow=0.0, hift=0.0, llm=0.0)
            trec, _ = run(1, 4 if quick else 20)
            TIM["on"] = False
            tot = TIM["flow"] + TIM["hift"] + TIM["llm"]
            aud_t = sum(r["audio_s"] for r in trec)
            for s in summary:
                if s["cond"] == name:
                    s.update(gpu_flow_s_per_audio_s=TIM["flow"] / aud_t, gpu_hift_s_per_audio_s=TIM["hift"] / aud_t,
                             gpu_llm_s_per_audio_s=TIM["llm"] / aud_t, decoder_share=(TIM["flow"] + TIM["hift"]) / tot)
            print(name, {k: round(v, 2) for k, v in TIM.items() if k != "on"}, flush=True)
        S = pd.DataFrame(summary)
        print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
        print(S.round(3).to_string())
        if quick:
            return None
        return {"": S, "_requests": pd.DataFrame(reqs)}

    def concurrency_fsu_mid(self, n_utt=None, quick=False):
        return self.concurrency(n_utt, quick, names=("R (ours) NFE2", "mid+R (ours)"))


def main(argv=None):
    main_for("timing", StreamingTiming, argv)


if __name__ == "__main__":
    sys.exit(main())
