"""E16 (v10.9) generation jobs for queue_gen / queue_eval: the empty "SIM LSPC" cells of Table II, LibriSpeech-PC test-clean
1,127 pairs (data/lists/lspc_all), seed 0, noise hash(utt|noise|0).
  CosyVoice 3   prep_cv3_lspc (LM tokens, cache/cosy3_tokens/lspc, shared by every condition), then
                cv3_{base_n2,R_n2,midR,teacher} = e10_jobs f_cv3_* commands on the lspc set and
                cv3_{tvsf,seltext,mambo4} = f_cv3_base_n2 + the E15 rule arguments
  GLM-TTS       glm_{allnull_n2,allnullR_n2,allnull_midR} = e14_jobs arguments with the E11-D LSPC token cache (heavy);
                teacher = the official speaker-null NFE 10 run of E11-D (out/E11D_b/d_teacher/utt0/lspc, not regenerated)
Queue order: GLM (heavy, before any evaluation exists) -> CosyVoice 3 tokens -> CosyVoice 3 decodes.
Outputs out/E16/<config>/utt0/lspc (out/E16 = junction to E:\\speechFM_store\\out\\E16).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e10_jobs as J10  # noqa: E402
import e14_jobs as J14  # noqa: E402
import e15_jobs as J15  # noqa: E402
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
LSPC = J10.LSPC
P.CV3_CACHE["lspc"] = "cache/cosy3_tokens/lspc"
J14.CACHE["lspc"] = "lspc"
CONFIGS, LABEL = {}, {}

for key in ("allnull_n2", "allnullR_n2", "allnull_midR"):
    c = J14.CONFIGS[key]
    CONFIGS[f"glm_{key}"] = dict(env=c["env"], K=[0], sets=LSPC, cmd=J14.glm(c["extra"]), extra=c["extra"], heavy=True)
    LABEL[f"glm_{key}"] = ("GLM-TTS", J14.LABEL[key])
CONFIGS["prep_cv3_lspc"] = dict(env="env_cosy", K=[0], sets=LSPC, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", J10.CK["cv3"], "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", J10.cache("cv3", s), "--prompt_prefix", P.CV3_PREFIX])
for key in ("base_n2", "R_n2", "midR", "teacher"):
    c = J10.CONFIGS[f"f_cv3_{key}"]
    CONFIGS[f"cv3_{key}"] = dict(env=c["env"], K=[0], sets=LSPC, cmd=c["cmd"], extra=c["extra"])
    LABEL[f"cv3_{key}"] = ("CosyVoice3", J10.LABEL[f"f_cv3_{key}"][1])
base = J10.CONFIGS["f_cv3_base_n2"]
for rule, (label, args) in J15.RULES.items():
    CONFIGS[f"cv3_{rule}"] = dict(env=base["env"], K=[0], sets=LSPC, extra=base["extra"] + args,
                                  cmd=lambda s, m, k, o, f=base["cmd"], a=args: [*f(s, m, k, o), *a])
    LABEL[f"cv3_{rule}"] = ("CosyVoice3", label)


def wav_dir(name, k, s):
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / "E16" / name / s
    return ROOT / "out" / "E16" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


def metrics_for(name, k, s, lang):
    return "wer,wavlm,ecapa,eres2net,utmos"


n_lines = P.n_lines
