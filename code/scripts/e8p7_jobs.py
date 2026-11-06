"""E8'-7 (v6.1 §B) external baselines, CosyVoice2 (NFE 2/3/4/10) + CosyVoice3 (NFE 2), en/zh seed 0.
Formulas, official code lines and translation: reports/E8p7_sources.md; patch patches/CosyVoice_E8p7.diff.
Conditions whose step mask is identical to an existing run are aliased (no regeneration), e.g. Kynk [0.1,0.9] at NFE 2
= R (ours) (mask 01), Sel-CFG input_text = E8' speaker-null, def_text+R = R′+R.
8-seed geometry for each method's best NFE 2 setting is a second phase (configs added after seed-0 results).
Outputs: out/E8p7/<config>/utt<k>/<set>/ (junction to C:\\speechFM_store\\out\\E8p7).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import e8p8_jobs as P8  # noqa: E402

P = P8.P
ROOT = P.ROOT
SETS = [x for x in P.SEEDTTS if x[0] in ("en", "zh")]
CONFIGS, ALIAS, LABEL = {}, {}, {}
CK = {"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}


def cache(model, s):
    return f"cache/cosy2_tokens/{s}" if model == "cv2" else P.CV3_CACHE[s]


def add(name, model, label, n, extra, alias=None):
    LABEL[name] = (model, label, n)
    if alias:
        ALIAS[name] = alias
        return

    def cmd(s, m, k, o):
        return ["scripts/run_cosy.py", "--model_dir", CK[model], "--meta", m + ".meta.lst", "--stage", "decode",
                "--cache_dir", cache(model, s), "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7", "--nfe", str(n), *extra]
    CONFIGS[name] = dict(env="env_cosy", K=[0], cmd=cmd, sets=SETS, extra=extra)


def mask(n, lo, hi):
    ts = 1 - np.cos(np.pi / 2 * np.arange(n) / n)
    return "".join("1" if lo <= t <= hi else "0" for t in ts)


BASE = {("cv2", n): f"e82_cv2_all_n{n}_def" for n in (2, 3, 10)}
BASE.update({("cv2", 4): "e7:cv2_default_n4", ("cv3", 2): "e81_cv3_n2_def"})
RCFG = {("cv2", 2): "e82_cv2_all_n2_R", ("cv2", 3): "e82_cv2_all_n3_R", ("cv2", 10): "e82_cv2_all_n10_R",
        ("cv2", 4): "e7:cv2_n4_m0011", ("cv3", 2): "e81_cv3_n2_R"}
for model, nfes in [("cv2", [2, 3, 10]), ("cv3", [2])]:
    for n in nfes:
        # (1) Kynkäänniemi guidance interval
        for lo, hi in [(0.1, 0.9), (0.2, 0.8), (0.1, 0.8)]:
            same_as_R = mask(n, lo, hi) == mask(n, 0.1, 1.0)
            add(f"{model}_kynk{lo}-{hi}_n{n}", model, f"Kynk [{lo},{hi}]", n, ["--cfg_interval", f"{lo},{hi}"],
                alias=RCFG[(model, n)] if same_as_R else None)
        # (3) MAMBO-G
        for a in [0.5, 1, 2, 4]:
            add(f"{model}_mambo{a}_n{n}", model, f"MAMBO-G({a})", n, ["--mambo_alpha", str(a)])
        # (4) Selective-CFG
        add(f"{model}_seltext_n{n}", model, "R′ (Sel-CFG input_text)", n, ["--cfg_branch", "speaker"],
            alias=f"e82_cv2_speaker_n{n}_def" if model == "cv2" else "B_cv3_Rp_n2")
        add(f"{model}_defltext_n{n}", model, "Sel-CFG def_text", n, ["--cfg_branch", "speaker", "--branch_switch_t", "0.08"])
for model, nfes in [("cv2", [2, 3, 4, 10]), ("cv3", [2])]:
    for n in nfes:  # (2) CFG-Zero*
        add(f"{model}_zi_n{n}", model, "CFG-Zero* zi", n, ["--zero_init", "1"])
        add(f"{model}_zis_n{n}", model, "CFG-Zero* zi+s*", n, ["--zero_init", "1", "--cfg_zero_star", "1"])
        add(f"{model}_s_n{n}", model, "CFG-Zero* s*", n, ["--cfg_zero_star", "1"])
for n in [2, 3]:  # def_text + R: first step(s) t<0.1 CFG off, then input_text == R′+R
    add(f"cv2_defltextR_n{n}", "cv2", "Sel-CFG def_text+R", n, [], alias=f"e82_cv2_speaker_n{n}_R")


# phase 2 (after seed-0 results, 2026-10-24): 8 seeds for each method's best NFE 2 setting, CosyVoice2
# (Kynk [*] = R and def_text+R = R′+R already have 8 seeds; CFG-Zero* zi collapses -> s* is its best)
for _n in ["cv2_s_n2", "cv2_mambo4_n2", "cv2_defltext_n2"]:
    CONFIGS[_n]["K"] = list(range(8))
GEOM7 = [("CosyVoice2", "R (ours) = Kynk [0.1,0.9]", "e82_cv2_all_n2_def", "e82_cv2_all_n2_R"),
         ("CosyVoice2", "CFG-Zero* s*", "e82_cv2_all_n2_def", "cv2_s_n2"),
         ("CosyVoice2", "MAMBO-G(4)", "e82_cv2_all_n2_def", "cv2_mambo4_n2"),
         ("CosyVoice2", "Sel-CFG def_text", "e82_cv2_all_n2_def", "cv2_defltext_n2"),
         ("CosyVoice2", "Sel-CFG def_text+R = R′+R", "e82_cv2_all_n2_def", "e82_cv2_speaker_n2_R")]


def wav_dir(name, k, s):
    if name in ALIAS:
        a = ALIAS[name]
        return P8.wav_dir(a, k, s) if not a.startswith("e7:") else P.e7_wav_dir(a[3:], k, s)
    if name in CONFIGS:
        return ROOT / "out" / "E8p7" / name / f"utt{k}" / s
    return P8.wav_dir(name, k, s)


def sets_for(name, k):
    return CONFIGS[name]["sets"]


n_lines = P.n_lines
