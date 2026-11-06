"""E17 (v10.10) generation jobs for queue_gen / queue_eval: the empty cells of the robustness table (Fig. 9 conditions,
100 utterances each, seed 0, noise hash(utt|noise|0)).
  FSU-Mid (= mid+R (ours): --nfe 1 --solver midpoint --cfg_interval 0.1,1, as E8'-8 A_cv*_midR and Table II) on
    mm_cv{2,3}_fsumid        CosyVoice2 / 3 mm_de          (E8'-5 command, caches cache/cosy{2,3}_tokens/mm_de)
    long_{off,str}_cv2_fsumid CosyVoice2 long_en / long_zh  (v8 §2.2 command; streaming = --stream 1)
    rb_cv2_fsumid            CosyVoice2 rb_len1/3/10, rb_snr20/10 (v8 §2.3 command; SIM against the clean 10 s *_ref lists)
  GLM-TTS all-null mm_de (E11-L command, cache cache/glm_tokens/mm_de):
    mmglm_an_{fsumid,tvsf,mambo4}  all-null + FSU-Mid (E14-1 arguments), all-null + TV-CFG corrected / MAMBO-G(4) (E15 arguments)
  GLM Sel-CFG input-text = the speaker-null branch = the official branch -> alias, not generated.
The builders below also give the official / FSU commands used by the regression check (scripts/e17_regress.py).
Outputs out/E17/<config>/utt0/<set> (out/E17 = junction to E:\\speechFM_store\\out\\E17).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e11d_jobs as J11D  # noqa: E402
import e14_jobs as J14  # noqa: E402
import e15_jobs as J15  # noqa: E402
import e8p_jobs as P  # noqa: E402
import e8v8r_jobs as JR  # noqa: E402
import e8v8s_jobs as JS  # noqa: E402

ROOT = P.ROOT
R_IV = P.R_IV
MID = ["--nfe", "1", "--solver", "midpoint", "--cfg_interval", R_IV]
MM_DE = [x for x in P.MM if x[0] == "mm_de"]
CK = {"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}
J14.CACHE["mm_de"] = "mm_de"


def cv_mm(model, extra):  # = e8p_jobs cosy / cosy3 (E8'-5 e85_*), e8p8_jobs cosy (A_cv*_midR)
    cache = (lambda s: f"cache/cosy2_tokens/{s}") if model == "cv2" else (lambda s: P.CV3_CACHE[s])
    return lambda s, m, k, o: ["scripts/run_cosy.py", "--model_dir", CK[model], "--meta", m + ".meta.lst", "--stage", "decode",
                               "--cache_dir", cache(s), "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7", *extra]


def long(mode, extra):  # = e8v8s_jobs s_{off,str}_*
    st = ["--stream", "1"] if mode == "str" else []
    return lambda s, m, k, o: ["scripts/run_cosy.py", "--model_dir", CK["cv2"], "--meta", m + ".meta.lst", "--stage", "decode",
                               "--cache_dir", f"cache/cosy2_tokens/{s}", "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7",
                               *extra, *st]


def rb(extra):  # = e8v8r_jobs r_*
    return lambda s, m, k, o: ["scripts/run_cosy.py", "--model_dir", CK["cv2"], "--meta", m + ".meta.lst", "--stage", "decode",
                               "--cache_dir", f"cache/cosy2_tokens/{s}", "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7",
                               *extra]


def glm_de(extra):  # = e11d_jobs l_* (same argument list as e14_jobs.glm)
    return J14.glm(extra)


CONFIGS, LABEL = {}, {}


def add(name, dec, label, cmd, sets, extra, heavy=False):
    CONFIGS[name] = dict(env="env_glm" if dec == "GLM-TTS" else "env_cosy", K=[0], cmd=cmd, sets=sets, extra=extra,
                         heavy=heavy)
    LABEL[name] = (dec, label)


# GLM first (heavy; no evaluation exists yet), then CosyVoice
AN = ["--cfg_branch", "allnull"]
add("mmglm_an_fsumid", "GLM-TTS", "all-null + FSU-Mid (ours)", glm_de(J14.CONFIGS["allnull_midR"]["extra"]), MM_DE,
    J14.CONFIGS["allnull_midR"]["extra"], heavy=True)
for rule in ("tvsf", "mambo4"):
    ex = ["--nfe", "2", *AN, *J15.RULES[rule][1]]
    add(f"mmglm_an_{rule}", "GLM-TTS", f"all-null + {J15.RULES[rule][0]}", glm_de(ex), MM_DE, ex, heavy=True)
add("mm_cv2_fsumid", "CosyVoice2", "FSU-Mid (ours)", cv_mm("cv2", MID), MM_DE, MID)
add("mm_cv3_fsumid", "CosyVoice3", "FSU-Mid (ours)", cv_mm("cv3", MID), MM_DE, MID)
for mode in ("off", "str"):
    add(f"long_{mode}_cv2_fsumid", "CosyVoice2", "FSU-Mid (ours)", long(mode, MID), JS.SETS, MID)
add("rb_cv2_fsumid", "CosyVoice2", "FSU-Mid (ours)", rb(MID), JR.SETS, MID)


def wav_dir(name, k, s):
    return ROOT / "out" / "E17" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"]


def metrics_for(name, k, s, lang):  # = the metric sets of the comparator rows (E8'-5 / E11-L / v8)
    if name.startswith("mmglm"):
        return J11D.metrics_for(name, k, s, lang)
    return "wer,wavlm,ecapa,eres2net,utmos" + (",cer2" if lang in ("ko", "ja") else "")


def eval_meta(meta):
    return JR.eval_meta(meta) if Path(meta).name in JR.NAMES else meta


n_lines = P.n_lines
