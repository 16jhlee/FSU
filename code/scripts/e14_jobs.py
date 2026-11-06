"""E14 (v10.6) generation jobs for queue_gen / queue_eval. Same runners, checkpoints and patches as E11 / E11-D / E13
(GLM-TTS patches/GLMTTS_E11.diff, CosyVoice patch chain, patches/IndexTTS_E13.diff); noise hash(utt|noise|k).
Outputs out/E14/<config>/utt<k>/<set>  (out/E14 = junction to C:\\speechFM_store\\out\\E14).
  E14-2  ga_10                      GLM all-null, N=2, first step only guided (w 0.7, 0), as E13 ga_* (seed 0 en/zh 300,
                                    seeds 1-7 en/zh 100); gachk_{11,00} = E13 ga_11 / ga_00 seed 1 en100 regenerated (bit check)
  E14-1  allnull_n2 / allnullR_n2   GLM all-null CFG / + R (ours), N=2, Seed-TTS hard 400 (E11-D AUX commands)
         allnull_midR               mid+R (ours) with the all-null branch at the midpoint evaluation, en 1,088 / zh 2,020 / hard 400
  E14-3b c3_n10_w0 / ix_n25_w0 / g_n10_w0   w = 0 sampler at the teacher's step count (CV3 10, IndexTTS-2 25, GLM 10),
                                    seed 0 en/zh 300, seeds 1-7 en/zh 100
  E14-4  g_w0zh_n{2,3,4,8}          GLM w = 0 curve on zh 300, seed 0 (N = 10 = g_n10_w0 seed 0 zh)
  E14-5  g_w0_n1                    GLM one step, w = 0, en/zh 300, seed 0
  E14-10 gs3_{base_n2,R_n2,midR}    GLM official (speaker-null) seeds 1, 2 on en/zh 300 (seed 0 = E11 block A/B folders);
                                    gs3chk = seed 1 en100 folders of E11 block C regenerated (bit check)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e11_jobs as J11  # noqa: E402
import e13_jobs as J13  # noqa: E402
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
S300, S100, EN100, K8_SETS, R_IV = J13.S300, J13.S100, J13.EN100, J13.K8_SETS, J13.R_IV
FULL = {"en_full": ("data/lists_full/en_all", "en"), "zh_full": ("data/lists_full/zh_all", "zh"),
        "hard": ("data/lists_full/hard_all", "zh")}
CACHE = {"en": "en", "zh": "zh", "en100": "en", "zh100": "zh", "en_full": "en", "zh_full": "zh", "hard": "hard"}
CONFIGS, LABEL = {}, {}


def glm(extra):
    return lambda s, m, k, o: ["scripts/run_glm.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                               f"cache/glm_tokens/{CACHE[s]}", "--out_dir", o, "--noise", f"utt:{k}", "--cfg", "0.7", *extra]


def add(name, label, runner, extra, sets_by_k):
    LABEL[name] = label
    # GLM-TTS decode beside a Seed-TTS hard / full-set evaluation overflowed 16 GB (2026-11-03, 1 h for 97 wavs) -> heavy
    CONFIGS[name] = dict(env=J13.ENV.get(runner, "env_glm"), K=sorted(sets_by_k), cmd=runner(extra), sets_by_k=sets_by_k,
                         extra=extra, heavy=runner is glm)


def full(*names):
    return [(n, *FULL[n]) for n in names]


AN = ["--cfg_branch", "allnull"]
# ---- E14-2 ----
add("ga_10", "first step only (10), all-null", glm, ["--nfe", "2", *AN, "--cfg_steps=0.7,0"], K8_SETS)
for c in ("11", "00"):
    add(f"gachk_{c}", f"E13 ga_{c} regenerated", glm, J13.CONFIGS[f"ga_{c}"]["extra"], {1: EN100})
# ---- E14-1 ----
add("allnull_n2", "all-null CFG", glm, ["--nfe", "2", *AN], {0: full("hard")})
add("allnullR_n2", "all-null CFG + R (ours)", glm, ["--nfe", "2", *AN, "--cfg_interval", R_IV], {0: full("hard")})
add("allnull_midR", "all-null CFG + mid+R (ours)", glm, ["--nfe", "1", "--solver", "midpoint", "--cfg_interval", R_IV, *AN],
    {0: full("en_full", "zh_full", "hard")})
# ---- E14-3b ----
add("c3_n10_w0", "w=0 N=10", J13.cv3, ["--cfg", "0", "--nfe", "10"], K8_SETS)
add("ix_n25_w0", "w=0 N=25", J13.ix, ["--cfg", "0", "--nfe", "25"], K8_SETS)
add("g_n10_w0", "w=0 N=10", glm, ["--nfe", "10", "--cfg", "0"], K8_SETS)
# ---- E14-4 / E14-5 ----
for n in (2, 3, 4, 8):
    add(f"g_w0zh_n{n}", "w=0", glm, ["--nfe", str(n), "--cfg", "0"], {0: S300[1:]})
add("g_w0_n1", "w=0 N=1", glm, ["--nfe", "1", "--cfg", "0"], {0: S300})
# ---- E14-10 (optional; last in the queue) ----
for key in ("base_n2", "R_n2", "midR"):
    ex = J11.CONFIGS[f"g_{key}"]["extra"]
    add(f"gs3_{key}", f"GLM {key} seeds 1-2", glm, ex, {1: S300, 2: S300})
add("gs3chk_base_n2", "E11 g_base_n2 seed 1 en100 regenerated", glm, J11.CONFIGS["g_base_n2"]["extra"], {1: EN100})
# ---- E14-6 / E14-7 dumps (GPU, through the generation queue; IndexTTS-2 = heavy -> never beside an evaluation) ----
DUMP_SETS = {0: [("en", "data/lists/en_100", "en")]}
for name, env, args in [("dump_f5", "env_f5dmo", ["e14_branches.py", "f5"]), ("dump_zip", "env_zip", ["e14_branches.py", "zip"]),
                        ("dump_idx", "env_index", ["e14_idx_closedform.py", "100"])]:
    CONFIGS[name] = dict(env=env, K=[0], sets_by_k=DUMP_SETS, prep=True,
                         cmd=lambda s, m, k, o, a=args: [str(ROOT / "scripts" / a[0]), *a[1:]])

def wav_dir(name, k, s):
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / "E14" / name / s
    return ROOT / "out" / "E14" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets_by_k"].get(k, [])


def metrics_for(name, k, s, lang):
    if "100" in s:
        return "wavlm,ecapa,eres2net"
    return "wer,wavlm,ecapa,eres2net,utmos" + (",werwh" if s == "zh_full" else "")


n_lines = P.n_lines
