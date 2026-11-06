"""v8 §1.3 (t* sensitivity) + §1.2 (seeds {0,1,2} for the Table-1 main rows) job configuration.
R(t*): w = 0 for steps whose start time t_k < t*  (== --cfg_interval t*,1). t* in {0.03,0.05,0.1,0.15,0.2,0.3,0.5} x
NFE {2,3,4,10} x {CosyVoice2, CosyVoice3, IndexTTS-2}, en/zh seed 0. Each (model, NFE, step mask) is generated once and
mask-identical conditions reuse existing runs (masks from the official schedules: cosine for CosyVoice, linspace for
IndexTTS-2). Outputs out/E8v8/<config>/utt<k>/<set>/ (junction to C:\\speechFM_store\\out\\E8v8).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import e8p9_jobs as J9  # noqa: E402

P = J9.P
ROOT = P.ROOT
SETS = J9.SETS
TSTAR = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
NFES = [2, 3, 4, 10]
CONFIGS, ALIAS = {}, {}


def sched(model, n):
    u = np.arange(n) / n
    return list(1 - np.cos(np.pi / 2 * u)) if model in ("cv2", "cv3") else list(u)


def mask(model, n, tstar):
    return "".join("0" if t < tstar - 1e-9 else "1" for t in sched(model, n))


EXIST = {  # (model, nfe, mask) -> config with seed 0
    ("cv2", 2, "11"): "e82_cv2_all_n2_def", ("cv2", 2, "01"): "e82_cv2_all_n2_R", ("cv2", 2, "00"): "e5:cv2_cfg0_n2",
    ("cv2", 3, "111"): "e82_cv2_all_n3_def", ("cv2", 3, "011"): "e82_cv2_all_n3_R", ("cv2", 3, "001"): "e7:cv2_n3_m001",
    ("cv2", 3, "000"): "c_cv2_w0_n3",
    ("cv2", 4, "1111"): "e7:cv2_default_n4", ("cv2", 4, "0111"): "e7:cv2_n4_m0111", ("cv2", 4, "0011"): "e7:cv2_n4_m0011",
    ("cv2", 4, "0001"): "e7:cv2_n4_m0001",
    ("cv2", 10, "1" * 10): "e82_cv2_all_n10_def", ("cv2", 10, "0111111111"): "e7:cv2_n10_m0111111111",
    ("cv2", 10, "0001111111"): "e82_cv2_all_n10_R", ("cv2", 10, "0000011111"): "e7:cv2_n10_m0000011111",
    ("cv2", 10, "0000001111"): "e7:cv2_n10_m0000001111",
    ("cv3", 2, "11"): "e81_cv3_n2_def", ("cv3", 2, "01"): "e81_cv3_n2_R", ("cv3", 3, "111"): "e81_cv3_n3_def",
    ("cv3", 3, "011"): "e81_cv3_n3_R", ("cv3", 4, "1111"): "e81_cv3_n4_def", ("cv3", 4, "0011"): "e81_cv3_n4_R",
    ("cv3", 10, "1" * 10): "e81_cv3_n10_def", ("cv3", 10, "0001111111"): "e81_cv3_n10_R",
    ("idx", 2, "11"): "e81_idx_n2_def", ("idx", 2, "01"): "e81_idx_n2_R", ("idx", 3, "111"): "e81_idx_n3_def",
    ("idx", 3, "011"): "e81_idx_n3_R", ("idx", 4, "1111"): "e81_idx_n4_def", ("idx", 4, "0111"): "e81_idx_n4_R",
    ("idx", 10, "1" * 10): "B_idx_base_n10",
}
TSTAR_MAP = {}  # (model, nfe, tstar) -> config name


def add(name, model, K, extra):
    if model in ("cv2", "cv3"):
        ck ={"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}[model]

        def cmd(s, m, k, o):
            cache = f"cache/cosy2_tokens/{s}" if model == "cv2" else P.CV3_CACHE[s]
            return ["scripts/run_cosy.py", "--model_dir", ck, "--meta", m + ".meta.lst", "--stage", "decode",
                    "--cache_dir", cache, "--noise", f"utt:{k}", "--out_dir", o, *extra]
        env = "env_cosy"
    else:
        def cmd(s, m, k, o):
            return ["scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                    f"cache/index2/{s}", "--out_dir", o, "--noise", f"utt:{k}", *extra]
        env = "env_index"
    CONFIGS[name] = dict(env=env, K=K, cmd=cmd, sets=SETS, extra=extra)


# ---- §1.3 ----
for model in ["cv2", "cv3", "idx"]:
    for n in NFES:
        for ts in TSTAR:
            mk = mask(model, n, ts)
            key = (model, n, mk)
            if key in EXIST:
                TSTAR_MAP[(model, n, ts)] = EXIST[key]
                continue
            name = f"t_{model}_n{n}_m{mk}"
            if name not in CONFIGS:
                ex = ["--cfg", "0" if "1" not in mk else "0.7", "--nfe", str(n)] + (["--cfg_interval", f"{ts},1"] if "1" in mk else [])
                add(name, model, [0], ex)
            TSTAR_MAP[(model, n, ts)] = name

# ---- §1.2: seeds 1, 2 where missing (seed 0 = existing run) ----
SEED_NEW = {"s_cv3_n3_def": ("cv3", "e81_cv3_n3_def", ["--cfg", "0.7", "--nfe", "3"]),
            "s_cv3_n3_R": ("cv3", "e81_cv3_n3_R", ["--cfg", "0.7", "--nfe", "3", "--cfg_interval", "0.1,1"]),
            "s_idx_n3_def": ("idx", "e81_idx_n3_def", ["--cfg", "0.7", "--nfe", "3"]),
            "s_idx_n3_R": ("idx", "e81_idx_n3_R", ["--cfg", "0.7", "--nfe", "3", "--cfg_interval", "0.1,1"]),
            "s_cv2_midR1": ("cv2", "b_cv2_midR1", ["--cfg", "0.7", "--t_span", "0,0.5,1", "--solver", "mid1", "--cfg_interval", "0.1,1"]),
            "s_cv3_midR1": ("cv3", "b_cv3_midR1", ["--cfg", "0.7", "--t_span", "0,0.5,1", "--solver", "mid1", "--cfg_interval", "0.1,1"])}
for name, (model, base, ex) in SEED_NEW.items():
    add(name, model, [1, 2], ex)
    ALIAS[name] = base


def wav_dir(name, k, s):
    if name in ALIAS and k == 0:
        return J9.wav_dir(ALIAS[name], 0, s)
    if name in CONFIGS:
        return ROOT / "out" / "E8v8" / name / f"utt{k}" / s
    return J9.wav_dir(name, k, s)


def sets_for(name, k):
    return CONFIGS[name]["sets"]


n_lines = P.n_lines
