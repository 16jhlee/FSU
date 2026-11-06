"""E15 (v10.8) generation jobs for queue_gen / queue_eval: the top three public external rules of Table I
(TV-CFG corrected = --tv symmetric_fix, Selective-CFG input-text = --cfg_branch speaker, MAMBO-G = --mambo_alpha 4; values of
Table I, not re-tuned) on the Table II full sets and the Fig. 9 robustness conditions, N = 2, seed 0.
Every config = an existing baseline command of the same set + the rule's arguments (argparse: the last value wins), so the
token caches, lengths, noise hash(utt|noise|k), prompt lists and evaluation settings are those of the baseline:
  E15-1  full en 1,088 / zh 2,020          e10_jobs f_{cv2,cv3,idx}_base_n2, e11d_jobs d_base_n2 (GLM official speaker-null)
  E15-3  robustness (16 conditions)        CosyVoice2/3 mm_ko/ja/de (e8p_jobs e85_*_n2_def), GLM mm_de (e11d l_base_n2),
                                           CosyVoice2 long_en/zh offline / streaming (e8v8s s_{off,str}_n2_def),
                                           CosyVoice2 prompt 1/3/10 s, 20/10 dB (e8v8r r_n2_def; SIM reference *_ref lists)
  E15-2  hard 400 (four decoders), LibriSpeech-PC 1,127 (CosyVoice2, IndexTTS-2, GLM)
GLM Sel-CFG input-text = the official speaker-null branch (alias of Official, not generated).
Earlier runs are not regenerated: the analysis-subset utterances of the same rule (300 en/zh, hard 100) are copied in
first by scripts/e15_seed_subset.py after the regression check (identical command / cache / noise), and the runners
skip existing wavs. Queue order E15-1 -> E15-3 -> E15-2 (v10.8 §2). Outputs out/E15/<config>/utt0/<set>
(out/E15 = junction to E:\\speechFM_store\\out\\E15).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e10_jobs as J10  # noqa: E402
import e11d_jobs as J11D  # noqa: E402
import e8p_jobs as P  # noqa: E402
import e8v8r_jobs as JR  # noqa: E402
import e8v8s_jobs as JS  # noqa: E402

ROOT = P.ROOT
RULES = {"tvsf": ("TV-CFG (corrected)", ["--tv", "symmetric_fix"]),
         "seltext": ("Selective-CFG (input-text)", ["--cfg_branch", "speaker"]),
         "mambo4": ("MAMBO-G (alpha = 4)", ["--mambo_alpha", "4"])}
CONFIGS, LABEL, SRC, BLOCK = {}, {}, {}, {}


def pick(mod, base, names):
    return [x for x in mod.sets_for(base, 0) if x[0] in names]


# (block, tag, decoder, source module, baseline config, sets)
GROUPS = [
    ("E15-1", "full_cv2", "CosyVoice2", J10, "f_cv2_base_n2", ("en", "zh")),
    ("E15-1", "full_cv3", "CosyVoice3", J10, "f_cv3_base_n2", ("en", "zh")),
    ("E15-1", "full_idx", "IndexTTS-2", J10, "f_idx_base_n2", ("en", "zh")),
    ("E15-1", "full_glm", "GLM-TTS", J11D, "d_base_n2", ("en", "zh")),
    ("E15-3", "mm_cv2", "CosyVoice2", P, "e85_cv2_n2_def", ("mm_ko", "mm_ja", "mm_de")),
    ("E15-3", "mm_cv3", "CosyVoice3", P, "e85_cv3_n2_def", ("mm_ko", "mm_ja", "mm_de")),
    ("E15-3", "mm_glm", "GLM-TTS", J11D, "l_base_n2", ("mm_de",)),
    ("E15-3", "long_off_cv2", "CosyVoice2", JS, "s_off_n2_def", ("long_en", "long_zh")),
    ("E15-3", "long_str_cv2", "CosyVoice2", JS, "s_str_n2_def", ("long_en", "long_zh")),
    ("E15-3", "rb_cv2", "CosyVoice2", JR, "r_n2_def", tuple(JR.NAMES)),
    ("E15-2", "hard_cv2", "CosyVoice2", J10, "f_cv2_base_n2", ("hard", "lspc")),
    ("E15-2", "hard_cv3", "CosyVoice3", J10, "f_cv3_base_n2", ("hard",)),
    ("E15-2", "hard_idx", "IndexTTS-2", J10, "f_idx_base_n2", ("hard", "lspc")),
    ("E15-2", "hard_glm", "GLM-TTS", J11D, "d_base_n2", ("hard", "lspc")),
]
for block, tag, dec, mod, base, names in GROUPS:
    for rule, (label, args) in RULES.items():
        if dec == "GLM-TTS" and rule == "seltext":
            continue
        name = f"{tag}_{rule}"
        c = mod.CONFIGS[base]
        CONFIGS[name] = dict(env=c["env"], K=[0], sets=pick(mod, base, names), extra=args,
                             cmd=lambda s, m, k, o, f=c["cmd"], a=args: [*f(s, m, k, o), *a],
                             heavy=dec in ("GLM-TTS", "IndexTTS-2"))
        LABEL[name], SRC[name], BLOCK[name] = (dec, rule, label), (mod, base), block


def wav_dir(name, k, s):
    return ROOT / "out" / "E15" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


def metrics_for(name, k, s, lang):
    """the baseline's metric set (same evaluation as the Table II / Fig. 9 rows)"""
    mod, base = SRC[name]
    if hasattr(mod, "metrics_for"):
        return mod.metrics_for(base, k, s, lang)
    return "wer,wavlm,ecapa,eres2net,utmos" + (",cer2" if lang in ("ko", "ja") else "")


def eval_meta(meta):
    """prompt-robustness lists: SIM against the clean 10 s reference (e8v8r_jobs.eval_meta)"""
    return JR.eval_meta(meta) if Path(meta).name in JR.NAMES else meta


n_lines = P.n_lines
