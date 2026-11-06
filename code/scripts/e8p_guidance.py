"""First-step guidance vector g1 = w*(v_cond - v_uncond) at t=0 (state = pure noise), CosyVoice2 / CosyVoice3,
en 100, seed 0, for each unconditional-branch variant (all / speaker / token). env_cosy. Uses the branch patch dump.
python scripts/e8p_guidance.py [cv2|cv3]   (summary statistics: scripts/guidance_stats.py)
Output: results/E8p_guidance.csv (cv2, E8'-2) / results/E8p8_guidance_cv3.csv (E8'-8 block D)
"""
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: F401,E402
from guidance_stats import summarise  # noqa: E402
from run_cosy import AutoModel, h, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"cv2": ("CosyVoice2-0.5B", "cache/cosy2_tokens/en", "results/E8p_guidance.csv"),
          "cv3": ("Fun-CosyVoice3-0.5B", "cache/cosy3_tokens/en_full", "results/E8p8_guidance_cv3.csv")}


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "cv2"
    ckpt, cache, out = MODELS[key]
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / ckpt))
    m, dec = cv.model, cv.model.flow.decoder
    dec.inference_cfg_rate = 0.7
    dec.t_span_override = [0.0, 1.0]  # one step from t=0: only g1 is needed
    rows = []
    for utt, *_ in read_meta(ROOT / "data/lists/en_100.meta.lst"):
        seg = torch.load(ROOT / cache / f"{utt}.pt")[0]
        P = seg["prompt_feat"].shape[1]
        G = {}
        for br in ["all", "speaker", "token"]:
            g = torch.Generator().manual_seed(h(f"{utt}|noise|0"))
            dec.rand_noise = torch.randn([1, 80, 50 * 300], generator=g)
            dec.cfg_branch = br
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
            G[br] = (0.7 * (st["vc"] - st["vu"]))[0, :, P:].numpy().astype("float64")
        rows += summarise(utt, G)
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / out, index=False)
    print(D.groupby("branch").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
