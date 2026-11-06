"""E8' (v5) job configuration — generic over models. Used by queue_gen.py / queue_eval.py / e8p_analyze.py.

R = CFG off for steps starting at t_k < 0.1 (repo time, t=0 noise) == --cfg_interval 0.1,1 (all repos patched).
Outputs: out/E8p/<config>/utt<k>/<set>/ (out/E8p is a junction to C:\\speechFM_store\\out\\E8p).
ALIAS maps a config name to existing result directories from E1/E3/E7 (no regeneration).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e3_jobs import wav_dir as e3_wav_dir  # noqa: E402
from e7_jobs import wav_dir as e7_wav_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDTTS = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh"), ("hard", "data/lists/hard_all", "zh")]
ZIPDIR = ROOT / "repos" / "ZipVoice"
R_IV = "0.1,1"
K8 = list(range(8))
CONFIGS = {}


def cosy(name, model_dir, cache, K, extra, sets=SEEDTTS, prefix=None):
    def cmd(s, m, k, o):
        c = ["scripts/run_cosy.py", "--model_dir", model_dir, "--meta", m + ".meta.lst", "--stage", "decode",
             "--cache_dir", f"{cache}/{s}", "--noise", f"utt:{k}", "--out_dir", o, *extra]
        return c
    CONFIGS[name] = dict(env="env_cosy", K=K, cmd=cmd, sets=sets, extra=extra)


# ---------------- E8'-2: unconditional-branch experiment (CosyVoice2) ----------------
for br in ["speaker", "token"]:
    for n in [2, 3, 10]:
        for r in [False, True]:
            name = f"e82_cv2_{br}_n{n}_{'R' if r else 'def'}"
            extra = ["--cfg", "0.7", "--nfe", str(n), "--cfg_branch", br] + (["--cfg_interval", R_IV] if r else [])
            cosy(name, "ckpt/CosyVoice2-0.5B", "cache/cosy2_tokens", K8 if n == 2 else [0], extra)

# ---------------- prep (no wavs): CosyVoice3 LLM tokens, IndexTTS-2 GPT/s2mel input cache ----------------
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"
CV3_CACHE = {"en": "cache/cosy3_tokens/en_full", "zh": "cache/cosy3_tokens/zh", "hard": "cache/cosy3_tokens/hard"}
CONFIGS["prep_cv3_tokens"] = dict(env="env_cosy", K=[0], sets=SEEDTTS, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", "ckpt/Fun-CosyVoice3-0.5B", "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", CV3_CACHE[s], "--prompt_prefix", CV3_PREFIX])
CONFIGS["prep_idx_cond"] = dict(env="env_index", K=[0], sets=SEEDTTS, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "cond", "--cache_dir", f"cache/index2/{s}",
    "--out_dir", f"out/E8p/idx_official_gptnoise/utt0/{s}"])


def cosy3(name, K, extra, sets=SEEDTTS):
    def cmd(s, m, k, o):
        return ["scripts/run_cosy.py", "--model_dir", "ckpt/Fun-CosyVoice3-0.5B", "--meta", m + ".meta.lst",
                "--stage", "decode", "--cache_dir", CV3_CACHE[s], "--noise", f"utt:{k}", "--out_dir", o, *extra]
    CONFIGS[name] = dict(env="env_cosy", K=K, cmd=cmd, sets=sets, extra=extra)


def idx(name, K, extra):
    def cmd(s, m, k, o):
        return ["scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                f"cache/index2/{s}", "--out_dir", o, "--noise", f"utt:{k}", *extra]
    CONFIGS[name] = dict(env="env_index", K=K, cmd=cmd, sets=SEEDTTS, extra=extra)


def f5(name, K, extra):
    def cmd(s, m, k, o):
        return ["scripts/run_f5.py", "--exp", "F5TTS_Base", "--ckpt", "ckpt/F5-TTS/F5TTS_Base/model_1200000.safetensors",
                "--meta", m + ".meta.lst", "--out_dir", o, "--cfg", "2.0", "--sway", "-1", "--use_epss", "0",
                "--noise", f"utt:{k}", *extra]
    CONFIGS[name] = dict(env="env_f5dmo", K=K, cmd=cmd, sets=SEEDTTS, extra=extra)


# ---------------- E8'-1: model generalisation (+ E8'-6 geometry via 8 seeds at NFE 2) ----------------
for n in [1, 2, 3, 4, 10]:
    for r in [False, True]:
        # NFE 2 (and the teacher default, as the E8'-6 geometry reference) get 8 seeds
        cosy3(f"e81_cv3_n{n}_{'R' if r else 'def'}", K8 if n == 2 or (n == 10 and not r) else [0],
              ["--cfg", "0.7", "--nfe", str(n)] + (["--cfg_interval", R_IV] if r else []))
for n in [1, 2, 3, 4, 25]:
    for r in [False, True]:
        idx(f"e81_idx_n{n}_{'R' if r else 'def'}", K8 if n == 2 or (n == 25 and not r) else [0],
            ["--cfg", "0.7", "--nfe", str(n)] + (["--cfg_interval", R_IV] if r else []))
for n in [4, 8, 16, 32]:
    f5(f"e81_f5_n{n}_R", [0], ["--nfe", str(n), "--cfg_interval", R_IV])
cosy("e81_cv2_n1_R", "ckpt/CosyVoice2-0.5B", "cache/cosy2_tokens", [0], ["--cfg", "0.7", "--nfe", "1", "--cfg_interval", R_IV])

# ---------------- E8'-3: first-step size (NFE 2 with schedule [0, t1, 1]) ----------------
for t1 in [0.1, 0.2, 0.3, 0.5, 0.7]:
    for r in [False, True]:
        ex = ["--cfg", "0.7", "--t_span", f"0,{t1},1"] + (["--cfg_interval", R_IV] if r else [])
        cosy(f"e83_cv2_t{t1}_{'R' if r else 'def'}", "ckpt/CosyVoice2-0.5B", "cache/cosy2_tokens", [0], ex)
        cosy3(f"e83_cv3_t{t1}_{'R' if r else 'def'}", [0], ex)

# ---------------- E8'-4: solvers and guidance strength (CosyVoice2) ----------------
for sol in ["midpoint", "heun"]:
    for steps in [1, 2]:
        for r in [False, True]:
            cosy(f"e84_cv2_{sol}_s{steps}_{'R' if r else 'def'}", "ckpt/CosyVoice2-0.5B", "cache/cosy2_tokens", [0],
                 ["--cfg", "0.7", "--nfe", str(steps), "--solver", sol] + (["--cfg_interval", R_IV] if r else []))
for n in [2, 3]:
    for w in ["0.3", "0.5", "1.0", "1.5"]:
        for r in [False, True]:
            name = f"e84_cv2_w{w}_n{n}_{'R' if r else 'def'}"
            cosy(name, "ckpt/CosyVoice2-0.5B", "cache/cosy2_tokens", [0],
                 ["--cfg", w, "--nfe", str(n)] + (["--cfg_interval", R_IV] if r else []))

# ---------------- E8'-5: languages (MiniMax ko/ja/de, scripts/build_minimax_lists.py); Seed hard is in E8'-1/2 ----------------
MM = [("mm_ko", "data/lists/mm_ko", "ko"), ("mm_ja", "data/lists/mm_ja", "ja"), ("mm_de", "data/lists/mm_de", "de")]
for s, _, _ in MM:
    CV3_CACHE[s] = f"cache/cosy3_tokens/{s}"
CONFIGS["prep_cv2_tokens_mm"] = dict(env="env_cosy", K=[0], sets=MM, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", f"cache/cosy2_tokens/{s}", "--text_frontend", "0"])
CONFIGS["prep_cv3_tokens_mm"] = dict(env="env_cosy", K=[0], sets=MM, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", "ckpt/Fun-CosyVoice3-0.5B", "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", CV3_CACHE[s], "--prompt_prefix", CV3_PREFIX, "--text_frontend", "0"])
for n in [2, 3]:
    for r in [False, True]:
        ex = ["--cfg", "0.7", "--nfe", str(n)] + (["--cfg_interval", R_IV] if r else [])
        tag = f"n{n}_{'R' if r else 'def'}"
        cosy(f"e85_cv2_{tag}", "ckpt/CosyVoice2-0.5B", "cache/cosy2_tokens", [0], ex, sets=MM)
        cosy3(f"e85_cv3_{tag}", [0], ex, sets=MM)
cosy("e85_cv2_n10_def", "ckpt/CosyVoice2-0.5B", "cache/cosy2_tokens", [0], ["--cfg", "0.7", "--nfe", "10"], sets=MM)
cosy3("e85_cv3_n10_def", [0], ["--cfg", "0.7", "--nfe", "10"], sets=MM)

# official all-null branch from earlier experiments
ALIAS = {
    "e82_cv2_all_n2_def": lambda k, s: e3_wav_dir("cv2_nfe2", k, s) if s != "hard" else ROOT / "out/E1/A/cosyvoice2/nfe2/cfg0.7/utt0/hard",
    "e82_cv2_all_n3_def": lambda k, s: e3_wav_dir("cv2_nfe3", k, s) if s != "hard" else ROOT / "out/E1/A/cosyvoice2/nfe3/cfg0.7/utt0/hard",
    "e82_cv2_all_n10_def": lambda k, s: e3_wav_dir("cv2_nfe10", k, s) if s != "hard" else ROOT / "out/E1/A/cosyvoice2/nfe10/cfg0.7/utt0/hard",
    "e82_cv2_all_n2_R": lambda k, s: e7_wav_dir("cv2_n2_m01", k, s),
    "e82_cv2_all_n3_R": lambda k, s: e7_wav_dir("cv2_n3_m011", k, s),
    "e82_cv2_all_n10_R": lambda k, s: e7_wav_dir("cv2_n10_m0001111111", k, s),
}


ALIAS.update({
    "e81_cv2_n1_def": lambda k, s: ROOT / "out/E1/A/cosyvoice2/nfe1/cfg0.7/utt0" / s,
    "e81_f5_n4_def": lambda k, s: ROOT / "out/E1/C/f5tts_v0/nfe4/cfg2.0/utt0" / s,
    "e81_f5_n8_def": lambda k, s: ROOT / "out/E1/C/f5tts_v0/nfe8/cfg2.0/utt0" / s,
    "e81_f5_n16_def": lambda k, s: ROOT / "out/E1/C/f5tts_v0/nfe16/cfg2.0/utt0" / s,
    "e81_f5_n32_def": lambda k, s: ROOT / "out/E1/C/f5tts_v0/nfe32/cfg2.0/utt0" / s,
})


def wav_dir(name, k, s):
    if name in ALIAS:
        return ALIAS[name](k, s)
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / name / s
    return ROOT / "out" / "E8p" / name / f"utt{k}" / s


def sets_for(name, k):
    c = CONFIGS[name]
    return c["sets"] if k == 0 else [x for x in c["sets"] if x[0] in ("en", "zh")]


def n_lines(meta):
    return sum(1 for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip())
