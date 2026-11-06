"""E12 (v10.4, review response) generation jobs for queue_gen / queue_eval (CosyVoice2, official checkpoints, tokens from
cache/cosy2_tokens/<set>, noise hash(utt|noise|k) as everywhere).
E12-1  c1_{base,R}_n2: en_100, seeds 0 and 1, NFE 2 baseline / R (ours) — the waveforms whose final-mel displacement
       (baseline - R) e12_direction.py decomposes; per-utterance ΔSIM / ΔWER come from these.
E12-5  st_h{25,50,100}_{def,R}_n2 + st_h{H}_n10 (teacher at the same hop): long_en (v8 §2.2 sentences), seed 0,
       official chunk-aware streaming with a FIXED token hop H (run_cosy.py --stream_hop H; official = 25 -> 50 -> 100).
Outputs out/E12/<config>/utt<k>/<set>/ (out/E12 = junction to C:\\speechFM_store\\out\\E12).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
EN100 = [("en", "data/lists/en_100", "en")]
LONG = [("long_en", "data/lists/long_en", "en")]
CONFIGS, LABEL = {}, {}


def add(name, label, nfe, ex, sets, K):
    LABEL[name] = (label, nfe)

    def cmd(s, m, k, o):
        return ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst", "--stage", "decode",
                "--cache_dir", f"cache/cosy2_tokens/{s}", "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7", *ex]
    CONFIGS[name] = dict(env="env_cosy", K=K, cmd=cmd, sets=sets, extra=ex)


add("c1_base_n2", "baseline", 2, ["--nfe", "2"], EN100, [0, 1])
add("c1_R_n2", "R (ours)", 2, ["--nfe", "2", "--cfg_interval", "0.1,1"], EN100, [0, 1])
for H in (25, 50, 100):
    st = ["--stream", "1", "--stream_hop", str(H)]
    add(f"st_h{H}_def_n2", "baseline", 2, ["--nfe", "2", *st], LONG, [0])
    add(f"st_h{H}_R_n2", "R (ours)", 2, ["--nfe", "2", "--cfg_interval", "0.1,1", *st], LONG, [0])
    add(f"st_h{H}_n10", "teacher", 10, ["--nfe", "10", *st], LONG, [0])


def wav_dir(name, k, s):
    return ROOT / "out" / "E12" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


n_lines = P.n_lines
