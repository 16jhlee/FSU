"""v8 §1.4: measured latency / throughput in the official CosyVoice2 streaming pipeline (LLM token stream + chunk-aware
causal flow matching + HiFT), for teacher NFE 10 / baseline NFE 2 / R (ours) NFE 2 / mid+R (ours).
Official settings: fp32, no TensorRT, no vLLM, batch 1, stream=True, token_hop_len reset to 25 before every request
(the official code grows it in place and never resets it). 50 requests per condition (en_100 utterances 0..49), after
5 warm-up requests. Run on an otherwise idle GPU.  env_cosy:  python scripts/v8_latency.py
Per request: first-packet latency (wall clock from request to first audio chunk), decoder (flow) time, HiFT time,
LLM time-to-first-chunk-tokens, total compute, audio seconds. Real-time sessions per GPU = audio seconds / GPU-busy
seconds (LLM + flow + HiFT, measured with cuda synchronisation) — an upper bound assuming perfect time sharing.
Output results/E8v8_latency.csv (+ summary printed).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: E402
from run_cosy import AutoModel, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def cos_sched(n):
    return (1 - torch.cos(torch.linspace(0, 1, n + 1) * 0.5 * torch.pi)).tolist()


def main():
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / "CosyVoice2-0.5B"))
    m = cv.model
    dec = m.flow.decoder
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

    conds = {"teacher NFE10": dict(nfe=10), "baseline NFE2": dict(nfe=2), "R (ours) NFE2": dict(nfe=2, iv=[0.1, 1.0]),
             "mid+R (ours)": dict(nfe=1, iv=[0.1, 1.0], solver="midpoint")}
    calls = {"teacher NFE10": 20, "baseline NFE2": 4, "R (ours) NFE2": 3, "mid+R (ours)": 3}
    rows_meta = read_meta(ROOT / "data/lists/en_100.meta.lst")
    out = []
    for name, c in conds.items():
        dec.solve_euler = orig_solve
        if c.get("solver"):
            run_cosy.install_solver(dec, c["solver"])
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
            out.append(dict(cond=name, calls=calls[name], utt=utt, first_packet_ms=first * 1000, first_flow_ms=first_flow * 1000,
                            first_hift_ms=first_hift * 1000, llm_first_chunk_ms=llm_first * 1000, n_first_tokens=n_first_tok,
                            flow_s=timers["flow"], hift_s=timers["hift"], llm_s=llm_total, total_s=total,
                            audio_s=audio / cv.sample_rate))
        print(name, "done", flush=True)
    D = pd.DataFrame(out)
    D["decoder_rtf"] = D.flow_s / D.audio_s
    D["gpu_busy_s"] = D.flow_s + D.hift_s + D.llm_s
    D["sessions"] = D.audio_s / D.gpu_busy_s
    D["decoder_share_first"] = D.first_flow_ms / D.first_packet_ms
    D.to_csv(ROOT / "results/E8v8_latency.csv", index=False)
    S = D.groupby("cond", sort=False)[["first_packet_ms", "first_flow_ms", "first_hift_ms", "llm_first_chunk_ms",
                                        "decoder_rtf", "sessions", "decoder_share_first"]].agg(['mean', 'std'])
    pd.set_option("display.width", 250)
    print(torch.cuda.get_device_name(0), torch.__version__)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
