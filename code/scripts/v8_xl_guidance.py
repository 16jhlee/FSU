"""v8 §2.1 (c): first-step guidance size RMS(v_c - v_u) at t=0 (all-null, w=0.7 dump) on the cross-lingual and
same-language lists (CosyVoice2 / CosyVoice3, 100 utterances each, seed-0 noise). env_cosy (CPU is fine).
python scripts/v8_xl_guidance.py cv2|cv3   -> results/E8v8_xl_guidance_<model>.csv
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: F401,E402
from run_cosy import AutoModel, h, read_meta  # noqa: E402
from e8v8x_jobs import LISTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    key = sys.argv[1]
    ckpt = {"cv2": "CosyVoice2-0.5B", "cv3": "Fun-CosyVoice3-0.5B"}[key]
    tok = {"cv2": "cosy2_tokens", "cv3": "cosy3_tokens"}[key]
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / ckpt))
    m, dec = cv.model, cv.model.flow.decoder
    dec.inference_cfg_rate, dec.t_span_override = 0.7, [0.0, 1.0]
    rows = []
    for s, lang in LISTS:
        for utt, *_ in read_meta(ROOT / f"data/lists/{s}.meta.lst"):
            seg = torch.load(ROOT / "cache" / tok / s / f"{utt}.pt")[0]
            P = seg["prompt_feat"].shape[1]
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
            st = dec.dump_states[0]
            dec.dump_states = None
            d = (st["vc"] - st["vu"])[0, :, P:].double()
            rows.append(dict(list=s, utt=utt, rms=float(d.pow(2).mean().sqrt())))
        print(s, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / f"results/E8v8_xl_guidance_{key}.csv", index=False)
    print(D.groupby("list").rms.mean().round(4).to_string())


if __name__ == "__main__":
    main()
