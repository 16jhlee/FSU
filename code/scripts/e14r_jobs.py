"""E14R-2 (v10.7 §2) generation jobs for queue_gen / queue_eval: external rules on IndexTTS-2 and GLM-TTS (all-null
branch, w = 0.7), N = 2, Seed-TTS-eval en/zh 300, seed 0. Same runners / caches / noise hash as E8'-1 and E13
(patches/IndexTTS_E14.diff, patches/GLMTTS_E14.diff); hyper-parameters = the CosyVoice rows of table II (not re-tuned).
Outputs out/E14R/<config>/utt0/<set>  (out/E14R = junction to E:\\speechFM_store\\out\\E14R).
Rows of table II that need no generation (aliases, results/E14_rules_regress.csv / logs/e14r_tfr_smoke_*.txt):
  guidance interval [0.1, 0.9] = FSU (same step mask at step starts 0 / 0.5 and 0 / 0.293), TFR on GLM = w = 0 (ga_00),
  Sel-CFG input_text: IndexTTS-2 B_idx_Rp_n2 (E8'-8), GLM = official speaker-null branch (g_base_n2).
PCI on IndexTTS-2 is not defined (prompt block of the state is held at 0 in training and inference) -> not run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e13_jobs as J13  # noqa: E402
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
S300 = J13.S300
JRR_W = "0.70,1.20,0.95"
RULES = [("defl", "Sel-CFG def_text", ["--cfg_branch", "speaker", "--branch_switch_t", "0.08"]),
         ("s", "CFG-Zero* s*", ["--cfg_zero_star", "1"]), ("zi", "CFG-Zero* zero-init", ["--zero_init", "1"]),
         ("tvs", "TV-CFG symmetric (released)", ["--tv", "symmetric"]),
         ("tvsf", "TV-CFG symmetric (last-delta fix)", ["--tv", "symmetric_fix"]), ("tvu", "TV-CFG stepup", ["--tv", "stepup"]),
         ("mambo4", "MAMBO-G(4)", ["--mambo_alpha", "4"]), ("jrr", "JRR-CFG", ["--jrr", JRR_W])]
CONFIGS, LABEL = {}, {}


def add(name, label, runner, extra, heavy):
    LABEL[name] = label
    CONFIGS[name] = dict(env=J13.ENV[runner], K=[0], cmd=runner(extra), sets_by_k={0: S300}, extra=extra, heavy=heavy)


for key, label, args in RULES + [("tfr", "TFR", ["--tfr", "1"])]:  # TFR is not an alias on IndexTTS-2 (mel diff 3-4e-4)
    add(f"ix_{key}", label, J13.ix, ["--nfe", "2", *args], True)
for key, label, args in RULES + [("pci", "PCI", ["--pci", "1"])]:
    # all-null base branch first; Sel-CFG def_text's later --cfg_branch speaker overrides it (argparse: last value)
    add(f"ga_{key}", label, J13.glm, ["--nfe", "2", "--cfg_branch", "allnull", *args], True)


def wav_dir(name, k, s):
    return ROOT / "out" / "E14R" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets_by_k"].get(k, [])


def metrics_for(name, k, s, lang):
    return "wer,wavlm,ecapa,eres2net,utmos"


n_lines = P.n_lines
