"""E10 (v10, TASLP) job configuration, used by queue_gen.py / queue_eval.py / e10_analyze.py.
Outputs out/E10/<block>/<config>/utt<k>/<set>/ (out/E10 is a junction to C:\\speechFM_store\\out\\E10).

E10-2 full sets (block 'full'): Seed-TTS-eval en 1,088 / zh 2,020 / hard 400 (data/lists_full) + LibriSpeech-PC test-clean
  1,127 pairs (data/lists/lspc_all, CosyVoice2 and IndexTTS-2), seed 0. Same token caches as the subset (the subset
  utterances are already cached; tokens are seeded per utterance) and noise hash(utt|noise|k).
  Conditions: base_n2, R_n2, midR (mid+R, 3 calls), base_n3, R_n3, teacher (official NFE 10 / 25).
E10-3 first-step w sign sweep (block 'wsign'): CosyVoice2/3 NFE 2 cosine, w = (w1, 0.7) via --cfg_steps; seed 0 on the
  300 subsets (full metrics), seeds 1-7 on en/zh 100 (embeddings). w1 = 0.7 / 0 / -0.433 reuse baseline / R (ours) /
  TV-CFG stepup (E8'-10; stepup's first scale g/3 - 1 = -0.4333).
E10-4 fill (block 'fill'): R'+R NFE 3 (CosyVoice3, IndexTTS-2), mid+R +1 Euler (IndexTTS-2; CosyVoice2/3 exist in E8'-9),
  R'+R NFE 2 seeds 1-7 (CosyVoice3, IndexTTS-2; CosyVoice2 exists), and WER/UTMOS for seeds 1, 2 of R' / R'+R.
Evaluation marker '_q10' (QMARK): metrics_for() decides the metric set; eval_metrics.py only adds missing columns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p9_jobs as J9  # noqa: E402
import e8p10_jobs as J10  # noqa: E402

P = J9.P8.P
ROOT = P.ROOT
QMARK = "_q10"
FULL = [("en", "data/lists_full/en_all", "en"), ("zh", "data/lists_full/zh_all", "zh"),
        ("hard", "data/lists_full/hard_all", "zh")]
LSPC = [("lspc", "data/lists/lspc_all", "en")]
S300 = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh")]
S100 = [("en100", "data/lists/en_100", "en"), ("zh100", "data/lists/zh_100", "zh")]
CK = {"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}
CACHE = {"cv2": lambda s: f"cache/cosy2_tokens/{s}", "cv3": lambda s: P.CV3_CACHE[s if s != "en100" else "en"],
         "idx": lambda s: f"cache/index2/{s}"}
R_IV = "0.1,1"
CONFIGS, ALIAS, LABEL, BLOCK = {}, {}, {}, {}


def cache(model, s):
    return CACHE[model]({"en100": "en", "zh100": "zh"}.get(s, s))


def add(block, name, model, label, nfe, extra, K=(0,), sets=FULL, sets_k=None):
    """sets: sets for seed 0; sets_k: sets for seeds >= 1 (default = sets)."""
    BLOCK[name], LABEL[name] = block, (model, label, nfe)
    if model in ("cv2", "cv3"):
        def cmd(s, m, k, o):
            return ["scripts/run_cosy.py", "--model_dir", CK[model], "--meta", m + ".meta.lst", "--stage", "decode",
                    "--cache_dir", cache(model, s), "--noise", f"utt:{k}", "--out_dir", o, *extra]
        env = "env_cosy"
    else:
        def cmd(s, m, k, o):
            return ["scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                    cache(model, s), "--out_dir", o, "--noise", f"utt:{k}", *extra]
        env = "env_index"
    CONFIGS[name] = dict(env=env, K=list(K), cmd=cmd, sets=sets, sets_k=sets_k or sets, extra=extra)


# ---------------- prep: LLM tokens (CosyVoice2/3) and GPT / s2mel inputs (IndexTTS-2) for the full sets ----------------
CONFIGS["prep_cv2_full"] = dict(env="env_cosy", K=[0], sets=FULL, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", CK["cv2"], "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", cache("cv2", s)])
CONFIGS["prep_cv3_full"] = dict(env="env_cosy", K=[0], sets=FULL, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", CK["cv3"], "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", cache("cv3", s), "--prompt_prefix", P.CV3_PREFIX])
# out_dir = the E8' official-pipeline folder so that the 300/100 subset utterances (cache + wav present) are skipped
CONFIGS["prep_idx_full"] = dict(env="env_index", K=[0], sets=FULL + LSPC, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "cond", "--cache_dir", cache("idx", s),
    "--out_dir", f"out/E8p/idx_official_gptnoise/utt0/{s}"])

# ---------------- E10-2 full sets ----------------
TEACHER_NFE = {"cv2": 10, "cv3": 10, "idx": 25}
COND = [("base_n2", "baseline", 2, ["--cfg", "0.7", "--nfe", "2"]),
        ("R_n2", "R (ours)", 2, ["--cfg", "0.7", "--nfe", "2", "--cfg_interval", R_IV]),
        ("midR", "mid+R (ours)", 1, ["--cfg", "0.7", "--nfe", "1", "--solver", "midpoint", "--cfg_interval", R_IV]),
        ("base_n3", "baseline", 3, ["--cfg", "0.7", "--nfe", "3"]),
        ("R_n3", "R (ours)", 3, ["--cfg", "0.7", "--nfe", "3", "--cfg_interval", R_IV]),
        ("teacher", "teacher", None, None)]
CALLS = {"base_n2": 4, "R_n2": 3, "midR": 3, "base_n3": 6, "R_n3": 5}
ORDER = {"cv2": "prep_cv2_full", "cv3": "prep_cv3_full", "idx": "prep_idx_full"}
for model in ("cv2", "cv3", "idx"):
    for key, label, n, extra in COND:
        if key == "teacher":
            n = TEACHER_NFE[model]
            extra = ["--cfg", "0.7", "--nfe", str(n)]
        sets = FULL + (LSPC if model in ("cv2", "idx") else [])
        add("full", f"f_{model}_{key}", model, label, n, extra, sets=sets)
ALIAS[("f_cv2_teacher", "lspc")] = lambda k: J10.wav_dir("lspc_cv2_base_n10", 0, "lspc")  # E8'-10 JRR reproduction

# ---------------- E10-3 first-step w sign sweep ----------------
W1 = [-1.0, -0.7, -0.43, -0.2, 0.0, 0.2, 0.35, 0.7, 1.05]
WS_EXIST = {("cv2", 0.7): "e82_cv2_all_n2_def", ("cv2", 0.0): "e82_cv2_all_n2_R", ("cv2", -0.43): "e8p10:cv2_tvu_n2",
            ("cv3", 0.7): "e81_cv3_n2_def", ("cv3", 0.0): "e81_cv3_n2_R", ("cv3", -0.43): "e8p10:cv3_tvu_n2"}
WS_NAME = {}
for model in ("cv2", "cv3"):
    for w in W1:
        name = f"ws_{model}_w{w}"
        WS_NAME[(model, w)] = name
        if (model, w) == ("cv3", -0.43):  # seed 0 = E8'-10 cv3_tvu_n2; seeds 1-7 on the 100 sets here
            add("wsign", name, model, "TV-CFG stepup (w1 = -0.433)", 2, ["--cfg", "0.7", "--nfe", "2", "--tv", "stepup"],
                K=range(1, 8), sets=S100)
            continue
        if (model, w) in WS_EXIST:
            continue
        add("wsign", name, model, f"w1 = {w}", 2, ["--cfg", "0.7", "--nfe", "2", f"--cfg_steps={w},0.7"],
            K=range(8), sets=S300, sets_k=S100)

# ---------------- E10-4 fill ----------------
add("fill", "fill_cv3_RpR_n3", "cv3", "R′+R", 3, ["--cfg", "0.7", "--nfe", "3", "--cfg_branch", "speaker", "--cfg_interval", R_IV],
    sets=S300)
add("fill", "fill_idx_RpR_n3", "idx", "R′+R", 3, ["--cfg", "0.7", "--nfe", "3", "--cfg_branch", "speaker", "--cfg_interval", R_IV],
    sets=S300)
add("fill", "fill_idx_midR1", "idx", "mid+R +1 Euler (ours)", 2,
    ["--cfg", "0.7", "--t_span", "0,0.5,1", "--solver", "mid1", "--cfg_interval", R_IV], sets=S300)
for model, base in [("cv3", "B_cv3_RpR_n2"), ("idx", "B_idx_RpR_n2")]:  # seeds 1-7 (seed 0 = E8'-8)
    add("fill", f"fill_{model}_RpR_n2", model, "R′+R", 2,
        ["--cfg", "0.7", "--nfe", "2", "--cfg_branch", "speaker", "--cfg_interval", R_IV], K=range(1, 8), sets=S300)
    ALIAS[(f"fill_{model}_RpR_n2", "k0")] = base
# seeds 1, 2 full metrics for existing R′ / R′+R NFE 2 runs (no generation; directories already _done)
EVAL_EXTRA = {"x_cv2_Rp_n2": "e82_cv2_speaker_n2_def", "x_cv2_RpR_n2": "e82_cv2_speaker_n2_R",
              "x_cv3_Rp_n2": "B_cv3_Rp_n2", "x_idx_Rp_n2": "B_idx_Rp_n2"}
for name, base in EVAL_EXTRA.items():
    BLOCK[name] = "fill"
    CONFIGS[name] = dict(env="none", K=[1, 2], sets=S300, sets_k=S300, evalonly=True, cmd=None)


# E10-3 dump measurement (dist / t1 per w1; GPU, run by queue_gen like a prep job)
for _m in ("cv2", "cv3"):
    BLOCK[f"dist_{_m}"] = "wsign"
    CONFIGS[f"dist_{_m}"] = dict(env="env_cosy", K=[0], sets=[("en", "data/lists/en_100", "en")], prep=True,
                                 cmd=lambda s, m, k, o, _m=_m: ["scripts/e10_wsign_dist.py", _m, "50"])

# generation order: each model's token prep right before its full-set runs (evaluation can start early)
_order = []
for model in ("cv2", "cv3", "idx"):
    _order += [ORDER[model]] + [n for n in CONFIGS if n.startswith(f"f_{model}_")]
CONFIGS = {n: CONFIGS[n] for n in _order + [n for n in CONFIGS if n not in _order]}


def wav_dir(name, k, s):
    if (name, s) in ALIAS:
        return ALIAS[(name, s)](k)
    if (name, "k0") in ALIAS and k == 0:
        return J9.wav_dir(ALIAS[(name, "k0")], 0, s)
    if name in EVAL_EXTRA:
        return J9.wav_dir(EVAL_EXTRA[name], k, s)
    if name.startswith("e8p10:"):
        return J10.wav_dir(name[6:], k, s)
    if name in CONFIGS:
        c = CONFIGS[name]
        if c.get("prep"):
            return ROOT / "cache" / "_prep" / name / s
        return ROOT / "out" / "E10" / BLOCK[name] / name / f"utt{k}" / s
    return J9.wav_dir(name, k, s)


def sets_for(name, k):
    c = CONFIGS[name]
    return c["sets"] if k == 0 else c.get("sets_k", c["sets"])


def metrics_for(name, k, s, lang):
    """E10-2: + Whisper-large-v3 WER for zh (en: Whisper-large-v3 is already the official ASR). Seeds >= 1: embeddings,
    except the 3-seed rows (E10-4) which need WER/UTMOS for seeds 1, 2."""
    if k == 0:
        m = "wer,wavlm,ecapa,eres2net,utmos"
        if BLOCK.get(name) == "full" and s == "zh":  # hard: CER not reported (earlier decision)
            m += ",werwh"
        return m
    if BLOCK.get(name) == "fill" and k in (1, 2):
        return "wer,wavlm,ecapa,eres2net,utmos"
    return "wavlm,ecapa,eres2net"


n_lines = P.n_lines
