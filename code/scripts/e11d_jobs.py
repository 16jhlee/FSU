"""E11-D (full sets) and E11-L (languages) for GLM-TTS — v10.3. Same runner / patch / pins as E11 (scripts/run_glm.py,
patches/GLMTTS_E11.diff, commit 4b944f4, HF b3feb69). Seed 0: noise hash(utt|noise|0), LLM seed hash(utt); token caches
shared with E11 (subset utterances reuse their tokens).
E11-D  Seed-TTS-eval en 1,088 / zh 2,020 / hard 400 (data/lists_full) + LibriSpeech-PC 1,127: {baseline, R (ours), mid+R (ours)}
       x NFE {2, 3} (mid+R: one midpoint step) + teacher (NFE 10); auxiliary all-null CFG / + R at NFE 2 on en/zh only.
E11-L  MiniMax ko / ja / de 100 (E8'-5 lists, raw text: --text_frontend 0): baseline / R / mid+R NFE 2 + teacher;
       auxiliary all-null pair NFE 2.
Outputs: en/zh under out/E11D (junction E:\\speechFM_store\\out\\E11D); lspc / hard / mm_* under out/E11D_b (junction
D:\\speechFM_store\\out\\E11D_b). Order of the queue: en/zh -> LSPC -> hard -> languages.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
FULL = [("en", "data/lists_full/en_all", "en"), ("zh", "data/lists_full/zh_all", "zh"), ("lspc", "data/lists/lspc_all", "en"),
        ("hard", "data/lists_full/hard_all", "zh")]
ENZH = FULL[:2]
# ko / ja: the GLM-TTS LLM does not terminate on these languages (2026-10-31 check: ko 60-66 characters -> 390-1,540 speech
# tokens = 16-62 s, ja 38 / 94 characters -> 382 / 1,041 tokens, ~3x the expected length; ~6 min per utterance) ->
# "unsupported — not evaluated" in Table 9 (v10.3 E11-L rule). de: 200 / 241 tokens for 52 / 48 characters (normal).
MM_ALL = [("mm_ko", "data/lists/mm_ko", "ko"), ("mm_ja", "data/lists/mm_ja", "ja"), ("mm_de", "data/lists/mm_de", "de")]
MM = [x for x in MM_ALL if x[0] == "mm_de"]
R_IV = "0.1,1"
CONFIGS, LABEL, BLOCK = {}, {}, {}


def cache(s):
    return f"cache/glm_tokens/{s}"


def add(block, name, label, nfe, extra, sets):
    LABEL[name], BLOCK[name] = (label, nfe), block

    def cmd(s, m, k, o):
        return ["scripts/run_glm.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir", cache(s), "--out_dir", o,
                "--noise", f"utt:{k}", "--cfg", "0.7", *extra]
    CONFIGS[name] = dict(env="env_glm", K=[0], cmd=cmd, sets=sets, extra=extra)


CONFIGS["prep_glm_full"] = dict(env="env_glm", K=[0], sets=FULL, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_glm.py", "--meta", m + ".meta.lst", "--stage", "tokens", "--cache_dir", cache(s)])
CONDS = [("base_n2", "baseline", 2, ["--nfe", "2"]), ("R_n2", "R (ours)", 2, ["--nfe", "2", "--cfg_interval", R_IV]),
         ("midR", "mid+R (ours)", 1, ["--nfe", "1", "--solver", "midpoint", "--cfg_interval", R_IV]),
         ("base_n3", "baseline", 3, ["--nfe", "3"]), ("R_n3", "R (ours)", 3, ["--nfe", "3", "--cfg_interval", R_IV]),
         ("teacher", "teacher", 10, ["--nfe", "10"])]
CALLS = {"base_n2": 4, "R_n2": 3, "midR": 3, "base_n3": 6, "R_n3": 5, "teacher": 20, "allnull_n2": 4, "allnullR_n2": 3}
AUX = [("allnull_n2", "all-null CFG", 2, ["--nfe", "2", "--cfg_branch", "allnull"]),
       ("allnullR_n2", "all-null CFG + R (ours)", 2, ["--nfe", "2", "--cfg_branch", "allnull", "--cfg_interval", R_IV])]
for key, lab, n, ex in CONDS:
    add("full", f"d_{key}", lab, n, ex, FULL)
for key, lab, n, ex in AUX:
    add("full", f"d_{key}", lab, n, ex, ENZH)
# heavy: the GLM-TTS LLM (~10 GB) + a Whisper evaluation exceed 16 GB (2026-10-31: 15.7 GB + 1.1 GB shared) -> gpu_lock
CONFIGS["prep_glm_mm"] = dict(env="env_glm", K=[0], sets=MM, prep=True, heavy=True, cmd=lambda s, m, k, o: [
    "scripts/run_glm.py", "--meta", m + ".meta.lst", "--stage", "tokens", "--cache_dir", cache(s), "--text_frontend", "0"])
for key, lab, n, ex in [c for c in CONDS if c[0] in ("base_n2", "R_n2", "midR", "teacher")] + AUX:
    add("lang", f"l_{key}", lab, n, ex, MM)


def wav_dir(name, k, s):
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / name / s
    base = "E11D" if s in ("en", "zh") else "E11D_b"
    return ROOT / "out" / base / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


def metrics_for(name, k, s, lang):
    m = "wer,wavlm,ecapa,eres2net,utmos"
    if s == "zh":
        m += ",werwh"
    if lang in ("ko", "ja"):
        m += ",cer2"
    return m


n_lines = P.n_lines
