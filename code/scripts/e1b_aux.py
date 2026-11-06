"""E1b auxiliary measurements (env_eval).

prompts : embed vocoder-resynthesised prompts (hift, vocos) with every embedder -> SIM-r reference
          data/cache/prompt_resynth_<voc>_<emb>.npz keyed like the SIM-o prompt cache.
noise   : embedder noise floor. For a fixed sample of generated wavs, re-encode with
          +1 / -1 sample shift and -0.5 dB gain (written as 16-bit PCM and reloaded); record
          |SIM(pert) - SIM(orig)| and 1 - cos(orig, pert). eps_emb = 95th percentile of |dSIM|.
          -> results/E1b_noise.csv
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import EMBEDDERS, cos, load16, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMBS = ["wavlm", "ecapa", "campp", "eres2net"]
METAS = ["data/lists/en_all.meta.lst", "data/lists/zh_all.meta.lst", "data/lists/hard_all.meta.lst"]


def pkey(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


def resynth_index(voc):
    d = ROOT / "data" / "cache" / "prompt_resynth" / voc
    idx = {}
    for l in (d / "index.tsv").read_text(encoding="utf-8").splitlines():
        p, f = l.split("\t")
        idx[pkey(p)] = d / f
    return idx


def do_prompts():
    for e in EMBS:
        emb = EMBEDDERS[e]()
        for voc in ["hift", "vocos"]:
            out = ROOT / "data" / "cache" / f"prompt_resynth_{voc}_{e}.npz"
            if out.exists():
                continue
            store = {k: emb(load16(str(f))) for k, f in resynth_index(voc).items()}
            np.savez(out, **store)
            print(e, voc, len(store), flush=True)
        del emb


SAMPLE_DIRS = ["out/E1/A/cosyvoice2/nfe10/cfg0.7/utt0/en", "out/E1/A/cosyvoice2/nfe2/cfg0.7/utt0/en",
               "out/E1/B/zipvoice/nfe16/cfg1.0/utt0/en", "out/E1/B/zipvoice/nfe8/cfg1.0/utt0/en",
               "out/E1/B/zipvoice_distill/nfe4/cfg3.0/utt0/en", "out/E1/C/dmo_student/nfe4/cfgfused/utt0/en"]


def do_noise():
    rows = {r["utt"]: r for r in read_meta(ROOT / METAS[0])}
    rng = np.random.default_rng(0)
    utts = list(rng.choice(sorted(rows), 20, replace=False))
    tmp = Path(tempfile.mkdtemp())
    pert = {}
    for d in SAMPLE_DIRS:
        for u in utts:
            w, sr = sf.read(ROOT / d / f"{u}.wav", dtype="float32")
            vs = {"orig": w, "shift+1": np.concatenate([[0.0], w[:-1]]), "shift-1": np.concatenate([w[1:], [0.0]]),
                  "gain-0.5dB": w * 10 ** (-0.5 / 20)}
            for k, v in vs.items():
                f = tmp / f"{abs(hash(d)) % 10**8}_{u}_{k}.wav"
                sf.write(f, np.asarray(v, dtype="float32"), sr, subtype="PCM_16")
                pert[(d, u, k)] = f
    recs = []
    for e in EMBS:
        emb = EMBEDDERS[e]()
        pc = dict(np.load(ROOT / "data" / "cache" / f"prompt_{e}.npz"))
        for d in SAMPLE_DIRS:
            for u in utts:
                ep = pc[pkey(rows[u]["prompt_wav"])]
                eo = emb(load16(str(pert[(d, u, "orig")])))
                so = cos(eo, ep)
                for k in ["shift+1", "shift-1", "gain-0.5dB"]:
                    ek = emb(load16(str(pert[(d, u, k)])))
                    recs.append(dict(emb=e, dir=d, utt=u, pert=k, dsim=abs(cos(ek, ep) - so), dcos=1 - cos(eo, ek)))
        del emb
        print(e, "noise done", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(ROOT / "results" / "E1b_noise.csv", index=False)
    print(df.groupby("emb").agg(eps95=("dsim", lambda x: np.percentile(x, 95)), dsim_med=("dsim", "median"),
                                dcos95=("dcos", lambda x: np.percentile(x, 95))).round(5))


if __name__ == "__main__":
    {"prompts": do_prompts, "noise": do_noise}[sys.argv[1]]()
