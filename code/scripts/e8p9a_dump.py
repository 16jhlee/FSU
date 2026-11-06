"""E8'-9a: guidance size at EVERY evaluation, RMS(v_c - v_u)(t_k) over the generated region, CosyVoice2/3, en 100,
seed-0 noise, w=0.7, for every condition of the Fig-10 point set (official NFE, branches, t1 sweep).
python scripts/e8p9a_dump.py cv2|cv3   (env_cosy; CPU is fine: CUDA_VISIBLE_DEVICES='')
Output results/E8p9a_dump_<model>.csv: cond, utt, k, t, dt, rms.  E_tot is assembled in e8p9a_law.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: F401,E402
from run_cosy import AutoModel, h, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"cv2": ("CosyVoice2-0.5B", "cache/cosy2_tokens/en"), "cv3": ("Fun-CosyVoice3-0.5B", "cache/cosy3_tokens/en_full")}


def cos_sched(n):
    return list(1 - np.cos(np.pi / 2 * np.arange(n + 1) / n))


def conds(model):
    c = [(f"all_n{n}", "all", cos_sched(n)) for n in [1, 2, 3, 4, 10]]
    c += [(f"speaker_n{n}", "speaker", cos_sched(n)) for n in [2, 3, 10]]
    if model == "cv2":
        c += [(f"token_n{n}", "token", cos_sched(n)) for n in [2, 3, 10]]
    c += [(f"t1_{t1}", "all", [0.0, t1, 1.0]) for t1 in [0.1, 0.2, 0.3, 0.5, 0.7]]
    return c


def main():
    key = sys.argv[1]
    grid = len(sys.argv) > 2 and sys.argv[2] == "grid"  # v8 §1.3: gamma(t) on t = 0, 0.05, ..., 0.95 (20-step Euler)
    cond_list = [("grid20", "all", list(np.linspace(0, 1, 21)))] if grid else conds(key)
    ckpt, cache = MODELS[key]
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / ckpt))
    m, dec = cv.model, cv.model.flow.decoder
    dec.inference_cfg_rate = 0.7
    rows = []
    for utt, *_ in read_meta(ROOT / "data/lists/en_100.meta.lst"):
        seg = torch.load(ROOT / cache / f"{utt}.pt")[0]
        P = seg["prompt_feat"].shape[1]
        for name, br, ts in cond_list:
            dec.t_span_override = ts
            dec.cfg_branch = br
            g = torch.Generator().manual_seed(h(f"{utt}|noise|0"))
            dec.rand_noise = torch.randn([1, 80, 50 * 300], generator=g)
            dec.dump_states = []
            with torch.inference_mode():
                m.flow.inference(token=seg["token"].to(m.device, dtype=torch.int32),
                                 token_len=torch.tensor([seg["token"].shape[1]], dtype=torch.int32).to(m.device),
                                 prompt_token=seg["prompt_token"].to(m.device),
                                 prompt_token_len=torch.tensor([seg["prompt_token"].shape[1]], dtype=torch.int32).to(m.device),
                                 prompt_feat=seg["prompt_feat"].to(m.device),
                                 prompt_feat_len=torch.tensor([seg["prompt_feat"].shape[1]], dtype=torch.int32).to(m.device),
                                 embedding=seg["embedding"].to(m.device), streaming=False, finalize=True)
            for k, st in enumerate(dec.dump_states):
                d = (st["vc"] - st["vu"])[0, :, P:].double()
                rows.append(dict(cond=name, utt=utt, k=k, t=ts[k], dt=ts[k + 1] - ts[k], rms=float(d.pow(2).mean().sqrt())))
            dec.dump_states = None
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / (f"results/E8v8_gamma_{key}.csv" if grid else f"results/E8p9a_dump_{key}.csv"), index=False)
    print(D.groupby(["cond", "k"]).rms.mean().round(3).to_string())


if __name__ == "__main__":
    main()
