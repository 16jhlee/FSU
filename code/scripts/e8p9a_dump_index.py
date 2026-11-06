"""E8'-9a: IndexTTS-2 s2mel guidance size at every evaluation, RMS(v_c - v_u)(t_k), en 100, seed-0 noise, w=0.7,
conditions: official linspace NFE {1,2,3,4,10,25}, speaker-null NFE {2,3,10}, schedule [0,0.3,1]. env_index
(patches IndexTTS_E8p + IndexTTS_branch; CPU is fine). Output results/E8p9a_dump_idx.csv (cond, utt, k, t, dt, rms)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_index import REPO, h, install_schedule_solver, load_model, read_meta, seed_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDS = [(f"all_n{n}", "all", list(np.linspace(0, 1, n + 1))) for n in [1, 2, 3, 4, 10, 25]]
CONDS += [(f"speaker_n{n}", "speaker", list(np.linspace(0, 1, n + 1))) for n in [2, 3, 10]]
CONDS += [("t1_0.3", "all", [0.0, 0.3, 1.0])]


def main():
    grid = len(sys.argv) > 1 and sys.argv[1] == "grid"  # v8 §1.3: gamma(t) on a 20-step linspace grid
    cond_list = [("grid20", "all", list(np.linspace(0, 1, 21)))] if grid else CONDS
    rows_meta = read_meta(ROOT / "data/lists/en_100.meta.lst")
    os.chdir(REPO)
    tts = load_model()
    cfm = tts.s2mel.models["cfm"]
    orig_solve = cfm.solve_euler
    rows = []
    for utt, *_ in rows_meta:
        s = torch.load(ROOT / "cache/index2/en" / f"{utt}.pt")[0]
        dev = tts.device
        for name, br, ts in cond_list:
            cfm.solve_euler = orig_solve
            install_schedule_solver(cfm, ",".join(str(float(x)) for x in ts), "euler")
            cfm.cfg_branch = br
            cfm.dump_states = []
            seed_all(h(f"{utt}|noise|0|0"))
            with torch.inference_mode():
                cfm.inference(s["mu"].to(dev), s["x_lens"].to(dev), s["prompt"].to(dev), s["style"].to(dev),
                              None, len(ts) - 1, inference_cfg_rate=0.7)
            for k, st in enumerate(cfm.dump_states):
                d = (st["vc"] - st["vu"])[0, :, st["prompt_len"]:].double()
                rows.append(dict(cond=name, utt=utt, k=k, t=ts[k], dt=ts[k + 1] - ts[k], rms=float(d.pow(2).mean().sqrt())))
            cfm.dump_states = None
        print(utt, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / ("results/E8v8_gamma_idx.csv" if grid else "results/E8p9a_dump_idx.csv"), index=False)
    print(D.groupby(["cond", "k"]).rms.mean().round(3).to_string())


if __name__ == "__main__":
    main()
