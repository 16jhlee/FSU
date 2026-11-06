"""v8 §2.1 cross-lingual prompts: CosyVoice2 / CosyVoice3, NFE 2·3 x {baseline, R (ours)} + teacher NFE 10, seed 0.
Lists (scripts/build_crossling_lists.py): xl_zh2en, xl_en2zh, xl_en2ko, xl_ko2en + same-language references xl_en2en,
xl_zh2zh, xl_ko2ko (identical prompts / texts). Texts or prompts in Korean use raw text (--text_frontend 0, as E8'-5).
Evaluation language = text language. Outputs out/E8v8/<config>/utt0/<list>/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
LISTS = [("xl_zh2en", "en"), ("xl_en2zh", "zh"), ("xl_en2ko", "ko"), ("xl_ko2en", "en"),
         ("xl_en2en", "en"), ("xl_zh2zh", "zh"), ("xl_ko2ko", "ko")]
SETS = [(s, f"data/lists/{s}", lang) for s, lang in LISTS]
CK = {"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
CONFIGS = {}


def cache(model, s):
    return f"cache/{'cosy2' if model == 'cv2' else 'cosy3'}_tokens/{s}"


def tf(s):
    return ["--text_frontend", "0"] if "ko" in s else []


for model in ["cv2", "cv3"]:
    def prep_cmd(s, m, k, o, model=model):
        return ["scripts/run_cosy.py", "--model_dir", CK[model], "--meta", m + ".meta.lst", "--stage", "tokens",
                "--cache_dir", cache(model, s), *tf(s)] + (["--prompt_prefix", CV3_PREFIX] if model == "cv3" else [])
    CONFIGS[f"xprep_{model}"] = dict(env="env_cosy", K=[0], sets=SETS, prep=True, cmd=prep_cmd)
    for n, r in [(2, False), (2, True), (3, False), (3, True), (10, False)]:
        ex = ["--cfg", "0.7", "--nfe", str(n)] + (["--cfg_interval", "0.1,1"] if r else [])

        def cmd(s, m, k, o, model=model, ex=ex):
            return ["scripts/run_cosy.py", "--model_dir", CK[model], "--meta", m + ".meta.lst", "--stage", "decode",
                    "--cache_dir", cache(model, s), "--noise", f"utt:{k}", "--out_dir", o, *ex]
        CONFIGS[f"x_{model}_n{n}_{'R' if r else 'def'}"] = dict(env="env_cosy", K=[0], sets=SETS, cmd=cmd, extra=ex)


def wav_dir(name, k, s):
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / name / s
    return ROOT / "out" / "E8v8" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


n_lines = P.n_lines
