"""E8'-7: MAMBO-G guidance-update ratio r_k = ||v_c - v_u|| / ||v_u|| at every evaluation of the official CosyVoice2
sampler (all-null CFG, w=0.7), NFE 2/3/10, en 20 utterances, seed 0 -> results/E8p7_mambo_r.csv. env_cosy.
Shows where MAMBO-G's damping w_k = w exp(-alpha r_k) bites (expected: largest r at t=0)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: F401,E402
from run_cosy import AutoModel, h, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / "CosyVoice2-0.5B"))
    m, dec = cv.model, cv.model.flow.decoder
    dec.inference_cfg_rate = 0.7
    rows = []
    for utt, *_ in read_meta(ROOT / "data/lists/en_20.meta.lst"):
        seg = torch.load(ROOT / "cache/cosy2_tokens/en" / f"{utt}.pt")[0]
        for n in [2, 3, 10]:
            dec.t_span_override = list(1 - np.cos(np.pi / 2 * np.arange(n + 1) / n))
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
                r = float(torch.norm(st["vc"] - st["vu"]) / torch.norm(st["vu"]))
                rows.append(dict(utt=utt, nfe=n, k=k, t=st["t"], r=r))
            dec.dump_states = None
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E8p7_mambo_r.csv", index=False)
    S = D.groupby(["nfe", "k"]).agg(t=("t", "mean"), r=("r", "mean")).reset_index()
    for a in [0.5, 1, 2, 4]:
        S[f"w_eff_a{a}"] = 0.7 * np.exp(-a * S.r)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
