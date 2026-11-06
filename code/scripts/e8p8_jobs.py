"""E8'-8 (v6 §2) job configuration, used by queue_gen.py / queue_eval.py / e8p8_analyze.py.
Labels (v6 §0): mid+R (ours) = midpoint 1 step (evaluations t=0, 0.5) + w=0 at t=0 (3 calls);
R′ (Sel-CFG input_text) = speaker-null unconditional branch at every step (= Selective-CFG [2509.19668]
input_text rule; not ours, v6.1 §A); R′+R = both; R (ours) = --cfg_interval 0.1,1.
Outputs: out/E8p8/<config>/utt<k>/<set>/ (out/E8p8 is a junction to C:\\speechFM_store\\out\\E8p8).
Comparisons (baseline NFE 2, R NFE 2, teacher) reuse E8' directories via e8p_jobs.wav_dir.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
SEEDTTS, MM = P.SEEDTTS, [x for x in P.MM if x[0] in ("mm_ko", "mm_ja")]
R_IV, K8 = P.R_IV, P.K8
CONFIGS = {}


def cosy(name, model, K, extra, sets=SEEDTTS):
    ckpt = {"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}[model]

    def cache(s):
        return f"cache/cosy2_tokens/{s}" if model == "cv2" else P.CV3_CACHE[s]

    def cmd(s, m, k, o):
        return ["scripts/run_cosy.py", "--model_dir", ckpt, "--meta", m + ".meta.lst", "--stage", "decode",
                "--cache_dir", cache(s), "--noise", f"utt:{k}", "--out_dir", o, *extra]
    CONFIGS[name] = dict(env="env_cosy", K=K, cmd=cmd, sets=sets, extra=extra)


def idx(name, K, extra):
    def cmd(s, m, k, o):
        return ["scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                f"cache/index2/{s}", "--out_dir", o, "--noise", f"utt:{k}", *extra]
    CONFIGS[name] = dict(env="env_index", K=K, cmd=cmd, sets=SEEDTTS, extra=extra)


MID_R = ["--cfg", "0.7", "--nfe", "1", "--solver", "midpoint", "--cfg_interval", R_IV]
# ---- block A: mid+R (ours), 8 seeds (geometry), + ko/ja for CosyVoice2/3 ----
cosy("A_cv2_midR", "cv2", K8, MID_R, sets=SEEDTTS + MM)
cosy("A_cv3_midR", "cv3", K8, MID_R, sets=SEEDTTS + MM)
idx("A_idx_midR", K8, MID_R)
# ---- block B: R′ (Sel-CFG input_text) = speaker-null branch, NFE 2 (8 seeds) / 3 / 10, and R′+R at NFE 2 ----
for n in [2, 3, 10]:
    ex = ["--cfg", "0.7", "--nfe", str(n), "--cfg_branch", "speaker"]
    cosy(f"B_cv3_Rp_n{n}", "cv3", K8 if n == 2 else [0], ex)
    idx(f"B_idx_Rp_n{n}", K8 if n == 2 else [0], ex)
idx("B_idx_base_n10", [0], ["--cfg", "0.7", "--nfe", "10"])  # H8-2 needs the NFE 10 baseline (E8' had 1-4, 25)
cosy("B_cv3_RpR_n2", "cv3", [0], ["--cfg", "0.7", "--nfe", "2", "--cfg_branch", "speaker", "--cfg_interval", R_IV])
idx("B_idx_RpR_n2", [0], ["--cfg", "0.7", "--nfe", "2", "--cfg_branch", "speaker", "--cfg_interval", R_IV])
# ---- block C: IndexTTS-2 NFE 2 with schedule [0, 0.3, 1], baseline vs R (ours), 8 seeds ----
idx("C_idx_t03_def", K8, ["--cfg", "0.7", "--t_span", "0,0.3,1"])
idx("C_idx_t03_R", K8, ["--cfg", "0.7", "--t_span", "0,0.3,1", "--cfg_interval", R_IV])

# CosyVoice2 mid+R seed 0 on Seed sets already exists (E8'-4)
ALIAS = {"A_cv2_midR": lambda k, s: P.wav_dir("e84_cv2_midpoint_s1_R", 0, s) if k == 0 and s in ("en", "zh", "hard") else None}


def wav_dir(name, k, s):
    if name in ALIAS and ALIAS[name](k, s) is not None:
        return ALIAS[name](k, s)
    if name in CONFIGS:
        return ROOT / "out" / "E8p8" / name / f"utt{k}" / s
    return P.wav_dir(name, k, s)


def sets_for(name, k):
    c = CONFIGS[name]
    return c["sets"] if k == 0 else [x for x in c["sets"] if x[0] in ("en", "zh")]


n_lines = P.n_lines
