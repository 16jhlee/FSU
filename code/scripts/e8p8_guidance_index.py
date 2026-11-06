"""E8'-8 block D: IndexTTS-2 s2mel first-step guidance g1 = w (v_c - v_u) at t=0, en 100, seed 0 noise,
branches all (official) / speaker (prompt mel + style null) / token (content mu null). env_index, patches
IndexTTS_E8p.diff + IndexTTS_branch.diff. First text segment of each utterance (as for CosyVoice).
Output: results/E8p8_guidance_idx.csv  (statistics: scripts/guidance_stats.py)
"""
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from guidance_stats import summarise  # noqa: E402
from run_index import REPO, h, load_model, read_meta, seed_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows_meta = read_meta(ROOT / "data/lists/en_100.meta.lst")
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    rows = []
    for utt, *_ in rows_meta:
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]
        G = {}
        for br in ["all", "speaker", "token"]:
            cfm.cfg_branch = br
            cfm.dump_states = []
            seed_all(h(f"{utt}|noise|0|0"))
            dev = tts.device
            with torch.inference_mode():
                cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev),
                              None, 1, inference_cfg_rate=0.7)
            st = cfm.dump_states[0]
            cfm.dump_states = None
            G[br] = (0.7 * (st["vc"] - st["vu"]))[0, :, st["prompt_len"]:].numpy().astype("float64")
        rows += summarise(utt, G)
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E8p8_guidance_idx.csv", index=False)
    print(D.groupby("branch").mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
