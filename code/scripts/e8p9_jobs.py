"""E8'-9 (v7 §2) optional tasks b/c/d — job configuration for queue_gen.py / queue_eval.py.
9b: mid+R + k final Euler steps (CosyVoice2/3, en/zh): --solver mid1 with schedule [0,0.5,1] (5 calls) / [0,0.5,0.85,1]
    (7 calls); the first interval is a midpoint step whose t=0 evaluation has w=0 (R), later evaluations use CFG.
9c: NFE grid {2,3,4,6,8,12,25} x {baseline, R (ours), w=0} for IndexTTS-2 and CosyVoice2 (en/zh, seed 0) + 8-seed
    geometry at NFE 4 and 8 (baseline, R).
9d: CosyVoice2 NFE 10 teacher at w=0.5 (w 0/0.3/1.0/1.5 exist from E5).
Existing runs are reused through ALIAS (per seed). Outputs: out/E8p9/<config>/utt<k>/<set>/ (junction to C:).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p8_jobs as P8  # noqa: E402
from e5_jobs import wav_dir as e5_wav_dir  # noqa: E402

P = P8.P
ROOT = P.ROOT
SETS = [x for x in P.SEEDTTS if x[0] in ("en", "zh")]
R_IV, K8 = P.R_IV, P.K8
CONFIGS, ALIAS = {}, {}
GRID = [2, 3, 4, 6, 8, 12, 25]


def cosy(name, model, K, extra):
    ck = {"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}[model]

    def cmd(s, m, k, o):
        cache = f"cache/cosy2_tokens/{s}" if model == "cv2" else P.CV3_CACHE[s]
        return ["scripts/run_cosy.py", "--model_dir", ck, "--meta", m + ".meta.lst", "--stage", "decode",
                "--cache_dir", cache, "--noise", f"utt:{k}", "--out_dir", o, *extra]
    CONFIGS[name] = dict(env="env_cosy", K=K, cmd=cmd, sets=SETS, extra=extra)


def idx(name, K, extra):
    def cmd(s, m, k, o):
        return ["scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                f"cache/index2/{s}", "--out_dir", o, "--noise", f"utt:{k}", *extra]
    CONFIGS[name] = dict(env="env_index", K=K, cmd=cmd, sets=SETS, extra=extra)


# ---- 9b ----
for model in ["cv2", "cv3"]:
    cosy(f"b_{model}_midR1", model, [0], ["--cfg", "0.7", "--t_span", "0,0.5,1", "--solver", "mid1", "--cfg_interval", R_IV])
    cosy(f"b_{model}_midR2", model, [0], ["--cfg", "0.7", "--t_span", "0,0.5,0.85,1", "--solver", "mid1",
                                          "--cfg_interval", R_IV])

# ---- 9c ----
EXIST = {  # (model, cond, nfe) -> existing config (seed 0 at least)
    ("cv2", "def", 2): "e82_cv2_all_n2_def", ("cv2", "def", 3): "e82_cv2_all_n3_def", ("cv2", "def", 4): "e7:cv2_default_n4",
    ("cv2", "R", 2): "e82_cv2_all_n2_R", ("cv2", "R", 3): "e82_cv2_all_n3_R", ("cv2", "R", 4): "e7:cv2_n4_m0011",
    ("cv2", "w0", 2): "e5:cv2_cfg0_n2",
    ("idx", "def", 2): "e81_idx_n2_def", ("idx", "def", 3): "e81_idx_n3_def", ("idx", "def", 4): "e81_idx_n4_def",
    ("idx", "def", 25): "e81_idx_n25_def", ("idx", "R", 2): "e81_idx_n2_R", ("idx", "R", 3): "e81_idx_n3_R",
    ("idx", "R", 4): "e81_idx_n4_R", ("idx", "R", 25): "e81_idx_n25_R"}
GEO_SEEDS = {("cv2", "def", 4): "all", ("cv2", "R", 4): "k0", ("idx", "def", 4): "k0", ("idx", "R", 4): "k0"}
COND_ARGS = {"def": ["--cfg", "0.7"], "R": ["--cfg", "0.7", "--cfg_interval", R_IV], "w0": ["--cfg", "0"]}
for model in ["cv2", "idx"]:
    for cond in ["def", "R", "w0"]:
        for n in GRID:
            name = f"c_{model}_{cond}_n{n}"
            ex = COND_ARGS[cond] + ["--nfe", str(n)]
            geo = cond in ("def", "R") and n in (4, 8)
            have = EXIST.get((model, cond, n))
            if have and not geo:
                ALIAS[name] = (have, "all")
                continue
            if have and GEO_SEEDS.get((model, cond, n)) == "all":
                ALIAS[name] = (have, "all")
                continue
            K = K8 if geo else [0]
            if have:  # seed 0 exists, generate seeds 1..7 only
                ALIAS[name] = (have, "k0")
                K = list(range(1, 8))
            (cosy if model == "cv2" else idx)(name, *(("cv2", K, ex) if model == "cv2" else (K, ex)))

# ---- 9d ----
cosy("d_cv2_w0.5_n10", "cv2", [0], ["--cfg", "0.5", "--nfe", "10"])


def _resolve(cfg, k, s):
    if cfg.startswith("e5:"):
        return e5_wav_dir(cfg[3:], k, s)
    if cfg.startswith("e7:"):
        return P.e7_wav_dir(cfg[3:], k, s)
    return P8.wav_dir(cfg, k, s)


def wav_dir(name, k, s):
    if name in ALIAS:
        cfg, which = ALIAS[name]
        if which == "all" or k == 0:
            return _resolve(cfg, k, s)
    if name in CONFIGS:
        return ROOT / "out" / "E8p9" / name / f"utt{k}" / s
    if name.startswith(("e5:", "e7:")):
        return _resolve(name, k, s)
    return P8.wav_dir(name, k, s)


def sets_for(name, k):
    return CONFIGS[name]["sets"]


n_lines = P.n_lines
