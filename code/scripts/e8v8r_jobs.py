"""v8 §2.3 prompt robustness (CosyVoice2, en 100): prompt length {1, 3, 10 s} (clean) and prompt SNR {20, 10 dB} (3 s),
lists from scripts/build_robust_lists.py. NFE 2 x {baseline, R (ours)} + teacher NFE 10, 8 seeds (geometry).
SIM reference = clean 10 s crop of the same utterance for every condition (eval_meta -> *_ref lists).
Outputs out/E8v8/<config>/utt<k>/<set>/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
NAMES = ["rb_len1", "rb_len3", "rb_len10", "rb_snr20", "rb_snr10"]
SETS = [(s, f"data/lists/{s}", "en") for s in NAMES if (ROOT / f"data/lists/{s}.meta.lst").exists()]
CONFIGS = {}
CONFIGS["rprep_cv2"] = dict(env="env_cosy", K=[0], sets=SETS, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", f"cache/cosy2_tokens/{s}"])
for name, ex in [("r_n2_def", ["--nfe", "2"]), ("r_n2_R", ["--nfe", "2", "--cfg_interval", "0.1,1"]), ("r_n10_def", ["--nfe", "10"])]:
    def cmd(s, m, k, o, ex=ex):
        return ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst", "--stage", "decode",
                "--cache_dir", f"cache/cosy2_tokens/{s}", "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7", *ex]
    CONFIGS[name] = dict(env="env_cosy", K=list(range(8)), cmd=cmd, sets=SETS, extra=ex)


def eval_meta(meta):
    return meta + "_ref"


def wav_dir(name, k, s):
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / name / s
    return ROOT / "out" / "E8v8" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


n_lines = P.n_lines
