"""E13 (v10.5) generation jobs for queue_gen / queue_eval. Official checkpoints and sampler patches as before (CosyVoice
patch chain, patches/IndexTTS_E13.diff = per-step w for IndexTTS-2, GLM-TTS patches/GLMTTS_E11.diff); noise hash(utt|noise|k).
Existing folders are reused when complete (ALIAS: same command, same seed rule — see each comment); otherwise outputs go to
out/E13/<config>/utt<k>/<set>  (out/E13 = junction to C:\\speechFM_store\\out\\E13).
  E13-1 measure 5   : c2_n10_w0                               CosyVoice2 10-step w = 0, en/zh 100 x seeds 0-7
  E13-6(a) grid     : c2g_{w1}_{w2}                           CosyVoice2 N=2 (w1, w2) grid, en/zh 300, seed 0
  E13-6(b) controls : {c2,c3,ix,gs,ga}_{11,01,10,00,h,w03}    CV2 / CV3 / IndexTTS-2 / GLM speaker-null / GLM all-null,
                      N=2, en/zh 300 x seeds 0-2  (11 baseline, 01 R, 10 first step only, 00 no guidance, h = (0.35, 0.7),
                      w03 = global w 0.3); teachers for per-seed recovery: {c2,c3,ix,gs}_T and ga_T (= E13-5)
  E13-3             : c2_10 also on en 100 x seeds 3-7 (8-seed geometry)
  E13-2             : ix_w{w1}                                IndexTTS-2 N=2 uniform, (w1, official 0.7), en/zh 300, seed 0
  E13-5             : ga_T                                    GLM all-null teacher N=10, en/zh 300 seeds 0-2 + en 100 seeds 3-7
  E13-8             : f5s0_n{8,16}_{def,R}                    F5-TTS v0 sway 0 (uniform), en 300, seed 0
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
S300 = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh")]
S100 = [("en100", "data/lists/en_100", "en"), ("zh100", "data/lists/zh_100", "zh")]
EN100 = S100[:1]
EN300 = S300[:1]
R_IV = "0.1,1"
CONFIGS, LABEL, ALIAS = {}, {}, {}


def base(s):
    return s.replace("100", "")


def cv2(extra):
    return lambda s, m, k, o: ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst",
                               "--stage", "decode", "--cache_dir", f"cache/cosy2_tokens/{base(s)}", "--noise", f"utt:{k}",
                               "--out_dir", o, *extra]


def cv3(extra):
    return lambda s, m, k, o: ["scripts/run_cosy.py", "--model_dir", "ckpt/Fun-CosyVoice3-0.5B", "--meta", m + ".meta.lst",
                               "--stage", "decode", "--cache_dir", P.CV3_CACHE[base(s)], "--noise", f"utt:{k}",
                               "--out_dir", o, *extra]


def ix(extra):
    return lambda s, m, k, o: ["scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                               f"cache/index2/{base(s)}", "--out_dir", o, "--noise", f"utt:{k}", *extra]


def glm(extra):
    return lambda s, m, k, o: ["scripts/run_glm.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir",
                               f"cache/glm_tokens/{base(s)}", "--out_dir", o, "--noise", f"utt:{k}", "--cfg", "0.7", *extra]


def f5(extra):
    return lambda s, m, k, o: ["scripts/run_f5.py", "--exp", "F5TTS_Base", "--ckpt",
                               "ckpt/F5-TTS/F5TTS_Base/model_1200000.safetensors", "--meta", m + ".meta.lst", "--out_dir", o,
                               "--cfg", "2.0", "--use_epss", "0", "--noise", f"utt:{k}", *extra]


ENV = {cv2: "env_cosy", cv3: "env_cosy", ix: "env_index", glm: "env_glm", f5: "env_f5dmo"}


def add(name, label, runner, extra, sets_by_k, alias=None):
    """sets_by_k: {k: [sets]}; alias: (module_wav_dir, config) reused for (k, s) when that folder is complete."""
    LABEL[name] = label
    CONFIGS[name] = dict(env=ENV[runner], K=sorted(sets_by_k), cmd=runner(extra), sets_by_k=sets_by_k, extra=extra)
    if alias:
        ALIAS[name] = alias


def k_sets(k012=S300, extra_k=None):
    d = {k: k012 for k in (0, 1, 2)}
    for k in (extra_k or []):
        d[k] = EN100
    return d


# ---------------- E13-1 measurement 5 (v10.5.1): CosyVoice2 10-step w = 0, seeds 0-7 on en/zh 100 ----------------
# (the N = 4 / 8 comparators of v10.5 are cancelled: E8'-9c c_cv2_{def,R}_n{4,8} have 8 seeds)
add("c2_n10_w0", "w=0 N=10 (unguided teacher steps)", cv2, ["--cfg", "0", "--nfe", "10"], {k: S100 for k in range(8)})

# ---------------- E13-6(a) grid ----------------
W1G, W2G = ["0", "0.2", "0.35", "0.7"], ["0.3", "0.5", "0.7", "1.0"]
GRID_ALIAS = {("0.7", "0.7"): ("e8p", "e82_cv2_all_n2_def"), ("0", "0.7"): ("e8p", "e82_cv2_all_n2_R"),
              ("0.35", "0.7"): ("e10", "ws_cv2_w0.35")}
for a in W1G:
    for b in W2G:
        add(f"c2g_{a}_{b}", f"(w1, w2) = ({a}, {b})", cv2, ["--nfe", "2", f"--cfg_steps={a},{b}"], {0: S300},
            GRID_ALIAS.get((a, b)))

# ---------------- v10.5.1 §2: attribution controls, 100 utterances x 8 seeds (replaces v10.5 E13-6(b)) ----------------
# seed 0 on the 300 subsets (full metrics; folders already generated under v10.5 are reused), seeds 1-7 on en/zh 100
# (SIM / geometry from embeddings, as the existing 8-seed runs). Conditions that already have 8 seeds are not listed:
# CV2 11 / 01 / (0.35, 0.7) / 00 (= TFR), CV3 11 / 01 / (0.35, 0.7), IndexTTS-2 11 / 01, GLM speaker-null 11 / 01.
CTRL = {"11": ("baseline (11)", ["--cfg", "0.7"]), "01": ("R (ours) (01)", ["--cfg", "0.7", "--cfg_interval", R_IV]),
        "10": ("first step only (10)", ["--cfg_steps=0.7,0"]), "00": ("no guidance (00)", ["--cfg", "0"]),
        "h": ("(w1, w2) = (0.35, 0.7)", ["--cfg_steps=0.35,0.7"])}
AL = {("ga", "11"): ("e11", "g_allnull_n2"), ("ga", "01"): ("e11", "g_allnullR_n2"),  # seed 0 (300) reused
      ("ga", "10"): ("e14", "ga_10")}  # E13-10 (v10.6): same command / sets as E14-2 ga_10, all 8 seeds reused
NEW = {"c2": ["10"], "c3": ["10", "00"], "ix": ["10", "h", "00"], "ga": ["11", "01", "h", "00", "10"]}
DEC = {"c2": (cv2, ["--nfe", "2"]), "c3": (cv3, ["--nfe", "2"]), "ix": (ix, ["--nfe", "2"]),
       "ga": (glm, ["--nfe", "2", "--cfg_branch", "allnull"])}
K8_SETS = {0: S300, **{k: S100 for k in range(1, 8)}}
for d, conds in NEW.items():
    runner, ex = DEC[d]
    for c in conds:
        lab, steps = CTRL[c]
        add(f"{d}_{c}", lab, runner, ex + steps, K8_SETS, AL.get((d, c)))
# E13-5: GLM all-null teacher (N = 10): seed 0 on en/zh 300, seeds 1-7 on en/zh 100 (8-seed geometry, G3)
add("ga_T", "teacher (all-null)", glm, ["--nfe", "10", "--cfg_branch", "allnull"], K8_SETS)
# E13-10 (v10.6): mid+R (ours) with the all-null branch (E11 g_midR + --cfg_branch allnull); seed 0 = E14-1 allnull_midR
# utterances, regenerated here and compared (scripts/e13_10_check.py). heavy: GLM decode never beside an evaluation
add("ga_midR", "mid+R (ours)", glm, ["--nfe", "1", "--solver", "midpoint", "--cfg_interval", R_IV, "--cfg_branch", "allnull"],
    K8_SETS)
CONFIGS["ga_midR"]["heavy"] = True
# ---------------- E13-2 IndexTTS-2 first-step weight sweep ----------------
W1S = ["-1", "-0.7", "-0.4333", "-0.2", "0", "0.2", "0.35", "0.7", "1.05"]
IX_ALIAS = {"0": ("e8p", "e81_idx_n2_R"), "0.7": ("e8p", "e81_idx_n2_def")}
for w in W1S:
    add(f"ix_w{w}", f"w1 = {w}", ix, ["--nfe", "2", f"--cfg_steps={w},0.7"], {0: S300}, IX_ALIAS.get(w))
ALIAS["ix_w0.35"] = ("e13", "ix_h")  # same command as ix_h seed 0

# ---------------- E13-8 F5-TTS sway 0 ----------------
for n in (8, 16):
    add(f"f5s0_n{n}_def", f"baseline sway 0 N={n}", f5, ["--nfe", str(n), "--sway", "0"], {0: EN300})
    add(f"f5s0_n{n}_R", f"R (ours) sway 0 N={n}", f5, ["--nfe", str(n), "--sway", "0", "--cfg_interval", R_IV], {0: EN300})


def _alias_dir(kind, cfg, k, s):
    if kind == "e8p":  # as e8p8_analyze.W (no matplotlib import: queue envs)
        import e8p8_jobs as J8
        return J8.wav_dir(cfg, k, s)
    if kind == "e10":
        import e10_jobs as J10
        return J10.wav_dir(cfg, k, s)
    if kind == "e11":
        import e11_jobs as J11
        return J11.wav_dir(cfg, k, s)
    if kind == "e13":
        return ROOT / "out" / "E13" / cfg / f"utt{k}" / s
    if kind == "e14":
        import e14_jobs as J14
        return J14.wav_dir(cfg, k, s)
    if kind == "e7":
        from e7_jobs import wav_dir as e7w
        return e7w(cfg, k, s)
    raise KeyError(kind)


def _complete(d, meta):
    try:
        n = sum(1 for f in os.listdir(d) if f.endswith(".wav"))
    except OSError:
        return False
    return n >= P.n_lines(meta)


def wav_dir(name, k, s):
    if name in ALIAS:
        meta = dict((x[0], x[1]) for x in S300 + S100)[s]
        d = Path(_alias_dir(*ALIAS[name], k, s))
        if _complete(d, meta):
            return d
    return ROOT / "out" / "E13" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets_by_k"].get(k, [])


def metrics_for(name, k, s, lang):
    return "wer,wavlm,ecapa,eres2net,utmos" if k == 0 and "100" not in s else "wavlm,ecapa,eres2net"


n_lines = P.n_lines
