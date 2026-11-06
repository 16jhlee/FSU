"""v8 §2.2 streaming chunk mode (CosyVoice2): long sentences (data/lists/long_{en,zh}, three joined test sentences),
offline vs official chunk-aware streaming (run_cosy.py --stream 1), NFE 2·3 x {baseline, R (ours)} + teacher NFE 10
(offline and streaming), seed 0. R in streaming = w=0 for the t<0.1 step of every chunk's flow call.
Chunk boundaries -> <out>/_chunks.tsv (sample index of each chunk end). Outputs out/E8v8/<config>/utt0/<set>/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
SETS = [("long_en", "data/lists/long_en", "en"), ("long_zh", "data/lists/long_zh", "zh")]
CONFIGS = {}
CONFIGS["sprep_cv2"] = dict(env="env_cosy", K=[0], sets=SETS, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", f"cache/cosy2_tokens/{s}"])
for mode in ["off", "str"]:
    for n, r in [(2, False), (2, True), (3, False), (3, True), (10, False)]:
        ex = ["--cfg", "0.7", "--nfe", str(n)] + (["--cfg_interval", "0.1,1"] if r else []) + (["--stream", "1"] if mode == "str" else [])

        def cmd(s, m, k, o, ex=ex):
            return ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst", "--stage", "decode",
                    "--cache_dir", f"cache/cosy2_tokens/{s}", "--noise", f"utt:{k}", "--out_dir", o, *ex]
        CONFIGS[f"s_{mode}_n{n}_{'R' if r else 'def'}"] = dict(env="env_cosy", K=[0], sets=SETS, cmd=cmd, extra=ex)


def wav_dir(name, k, s):
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / name / s
    return ROOT / "out" / "E8v8" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


n_lines = P.n_lines
