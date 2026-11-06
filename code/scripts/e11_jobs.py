"""E11 (v10.2) fourth token-conditioned model: GLM-TTS. Job configuration for queue_gen.py / queue_eval.py.
Official: NFE 10, cosine schedule t_k = 1 - cos(pi/2 k/N) (t=0 noise), Euler, CFG w 0.7 ((1+w)v_c - w v_u, two forward
passes), all-null unconditional branch (tokens + prompt-mel/speaker condition + speaker embedding zeroed).
Runner scripts/run_glm.py (patch patches/GLMTTS_E11.diff). Outputs out/E11/<config>/utt<k>/<set>/ (junction to
E:\\speechFM_store\\out\\E11; C: was low after the GLM-TTS weights and env_glm).
Block A  NFE {10 (teacher), 5, 3, 2, 1} baseline, en/zh 300, seed 0.
Block B  NFE 2 / 3 / 10: R (ours) (--cfg_interval 0.1,1), mid+R (ours) (midpoint 1 step, 3 calls), R' (speaker-null
         branch), R'+R, Kynk [0.1,0.9] (NFE 2, mask identity check); t* sensitivity (only masks that differ from R);
         (ii) diagnostic: w = 0 at NFE {2, 3, 4, 8, 10}.
Block C  8-seed geometry NFE 2 (baseline / R / mid+R) + teacher 8 seeds on en/zh 100; t1 sweep NFE 2 [0, t1, 1] baseline/R
         (en 300); w1 sweep (en 300). Dumps (g1 branches, gamma(t), 1-step closed form / rho) in scripts/e11_dumps.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p_jobs as P  # noqa: E402

ROOT = P.ROOT
SETS = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh")]
EN = SETS[:1]
S100 = [("en100", "data/lists/en_100", "en"), ("zh100", "data/lists/zh_100", "zh")]
CACHE = {"en": "cache/glm_tokens/en", "zh": "cache/glm_tokens/zh", "en100": "cache/glm_tokens/en", "zh100": "cache/glm_tokens/zh"}
R_IV = "0.1,1"
K8 = list(range(8))
CONFIGS, LABEL, ALIAS = {}, {}, {}


def add(name, label, nfe, extra, K=(0,), sets=SETS, sets_k=None):
    LABEL[name] = (label, nfe)

    def cmd(s, m, k, o):
        return ["scripts/run_glm.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir", CACHE[s],
                "--out_dir", o, "--noise", f"utt:{k}", "--cfg", "0.7", *extra]
    CONFIGS[name] = dict(env="env_glm", K=list(K), cmd=cmd, sets=sets, sets_k=sets_k or sets, extra=extra)


CONFIGS["prep_glm_tokens"] = dict(env="env_glm", K=[0], sets=SETS, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_glm.py", "--meta", m + ".meta.lst", "--stage", "tokens", "--cache_dir", CACHE[s]])

# ---- block A ----
for n in [10, 5, 3, 2, 1]:
    geo = n in (2, 10)
    add(f"g_base_n{n}", "teacher" if n == 10 else "baseline", n, ["--nfe", str(n)], K=K8 if geo else (0,),
        sets_k=S100 if geo else None)
# ---- block B ----
for n in [2, 3, 10]:
    add(f"g_R_n{n}", "R (ours)", n, ["--nfe", str(n), "--cfg_interval", R_IV], K=K8 if n == 2 else (0,),
        sets_k=S100 if n == 2 else None)
    if n in (2, 3):
        # The released config has speech_token_cfg = False: the OFFICIAL unconditional branch already is speaker-null
        # (tokens kept), so R′ (Sel-CFG input_text) == baseline and R′+R == R (aliases in the analysis). Instead the
        # all-null branch (tokens zeroed too, = CosyVoice's official branch) is run as the comparison pair.
        add(f"g_allnull_n{n}", "all-null CFG", n, ["--nfe", str(n), "--cfg_branch", "allnull"])
        add(f"g_allnullR_n{n}", "all-null CFG + R", n, ["--nfe", str(n), "--cfg_branch", "allnull", "--cfg_interval", R_IV])
add("g_midR", "mid+R (ours)", 1, ["--nfe", "1", "--solver", "midpoint", "--cfg_interval", R_IV], K=K8, sets_k=S100)
add("g_kynk_n2", "Kynk [0.1,0.9]", 2, ["--nfe", "2", "--cfg_interval", "0.1,0.9"], sets=EN)
# t* sensitivity, en (masks identical to R or to an existing run are aliased in the analysis)
TSTAR = {("n3", "m001"): ["--nfe", "3", "--cfg_interval", "0.2,1"], ("n10", "m0000011111"): ["--nfe", "10", "--cfg_interval", "0.2,1"],
         ("n10", "m0000001111"): ["--nfe", "10", "--cfg_interval", "0.3,1"]}
for (n, mk), ex in TSTAR.items():
    add(f"g_ts_{n}_{mk}", f"R(t*) mask {mk[1:]}", int(n[1:]), ex, sets=EN)
# (ii) diagnostic: w = 0 curve (NFE 2 = t* 0.3 at NFE 2)
for n in [2, 3, 4, 8, 10]:
    add(f"g_w0_n{n}", "w=0", n, ["--nfe", str(n), "--cfg", "0"], sets=EN)
# ---- block C ----
for t1 in [0.1, 0.2, 0.3, 0.5, 0.7]:
    for r in (False, True):
        add(f"g_t1_{t1}_{'R' if r else 'def'}", "R (ours)" if r else "baseline", 2,
            ["--t_span", f"0,{t1},1"] + (["--cfg_interval", R_IV] if r else []), sets=EN)
for w in [-0.43, -0.2, 0.2, 0.35, 1.05]:  # w1 = 0 -> g_R_n2, w1 = 0.7 -> g_base_n2
    add(f"g_ws_w{w}", f"w1 = {w}", 2, ["--nfe", "2", f"--cfg_steps={w},0.7"], sets=EN)


# block C dumps (g1 branches, gamma(t), one-step closed form / rho) — needs the en token cache
CONFIGS["dumps"] = dict(env="env_glm", K=[0], sets=[("en", "data/lists/en_100", "en")], prep=True,
                        cmd=lambda s, m, k, o: ["scripts/e11_dumps.py", "100"])


def wav_dir(name, k, s):
    if CONFIGS.get(name, {}).get("prep"):
        return ROOT / "cache" / "_prep" / name / s
    return ROOT / "out" / "E11" / name / f"utt{k}" / s


def sets_for(name, k):
    c = CONFIGS[name]
    return c["sets"] if k == 0 else c.get("sets_k", c["sets"])


n_lines = P.n_lines
