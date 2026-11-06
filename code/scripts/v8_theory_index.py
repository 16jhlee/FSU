"""v8 §4 (IndexTTS-2): Lemma 1 (seed-invariance of v_c - v_u at t=0, 8 seeds) and Prop 4 (nested decomposition
g_all = g_spk + g_tok with g_tok = v_u(speaker-null) - v_u(all-null)). en first 50 utterances, first text segment.
env_index (patches IndexTTS_E8p + IndexTTS_branch).  Output results/E8v8_theory_idx.csv
"""
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_index import REPO, h, load_model, read_meta, seed_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def cosv(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    n_utt = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    meta = read_meta(ROOT / "data/lists/en_100.meta.lst")[:n_utt]
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    dev = tts.device
    rows = []

    def step(s, utt, k, branch):
        cfm.cfg_branch, cfm.dump_states = branch, []
        seed_all(h(f"{utt}|noise|{k}|0"))
        with torch.inference_mode():
            cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev), None, 1,
                          inference_cfg_rate=0.7)
        st, cfm.dump_states = cfm.dump_states[0], None
        P = st["prompt_len"]
        return st["vc"][0, :, P:].double().numpy(), st["vu"][0, :, P:].double().numpy()
    for utt, *_ in meta:
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]
        D = []
        for k in range(8):
            vc, vu = step(s, utt, k, "all")
            D.append(vc - vu)
        pc = [cosv(a, b) for a, b in itertools.combinations(D, 2)]
        rm = [float(np.sqrt(np.mean(d ** 2))) for d in D]
        rows.append(dict(utt=utt, check="lemma1", cos_mean=np.mean(pc), cos_min=np.min(pc), rms_cv=np.std(rm) / np.mean(rm)))
        for k in range(2):
            vc_a, vu_a = step(s, utt, k, "all")
            vc_s, vu_s = step(s, utt, k, "speaker")
            g_all, g_spk, g_tok = vc_a - vu_a, vc_s - vu_s, vu_s - vu_a
            rows.append(dict(utt=utt, check="prop4", k=k, n_all=np.sum(g_all ** 2), n_spk=np.sum(g_spk ** 2),
                             n_tok=np.sum(g_tok ** 2), cos_spk_tok=cosv(g_spk, g_tok)))
        print(utt, flush=True)
    Df = pd.DataFrame(rows)
    Df.to_csv(ROOT / "results/E8v8_theory_idx.csv", index=False)
    for c, g in Df.groupby("check"):
        print(c, g.drop(columns=["utt", "check"]).mean(numeric_only=True).round(4).to_dict())


if __name__ == "__main__":
    main()
