"""E7 (v4 §1): prescription rule — which steps get CFG, per model and NFE.

Guidance variants act on the START time t_k of each Euler step (repo time, t=0 noise):
  default (all steps) | first (w=0 only at t=0) | iv[a,b] (w>0 only for t_k in [a,b]) | ramp t* (w_k = w*min(1,t_k/t*))
Official schedules: CosyVoice2 cosine t_k = 1-cos(pi/2*k/N); ZipVoice teacher t-shift 0.5 t_k = 0.5u/(1-0.5u), u=k/N.
Variants that switch CFG on at exactly the same steps (same mask) are generated once; every requested variant is
mapped to its mask config (VARIANT_TO_CONFIG) for reporting.
Sets: en 300 / zh 300 / hard 100 at seed 0; 8 seeds (en/zh) for the core conditions.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e3_jobs import wav_dir as e3_wav_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETS = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh"), ("hard", "data/lists/hard_all", "zh")]
ZIPDIR = ROOT / "repos" / "ZipVoice"


def sched(model, n):
    if model == "cv2":
        return [1 - math.cos(math.pi / 2 * k / n) for k in range(n)]
    return [0.5 * (k / n) / (1 - 0.5 * (k / n)) for k in range(n)]


VARIANTS = {"first": (1e-6, 1.0), "iv0.2-0.8": (0.2, 0.8), "iv0.1-1": (0.1, 1.0), "iv0.2-1": (0.2, 1.0),
            "iv0.3-1": (0.3, 1.0)}
REQUEST = {("zip", n): ["first", "iv0.2-0.8", "iv0.1-1", "iv0.3-1"] for n in [2, 4, 8, 16]}
REQUEST.update({("cv2", n): ["first", "iv0.1-1", "iv0.2-1", "iv0.3-1", "iv0.2-0.8"] for n in [2, 3, 4, 10]})
CORE = {("zip", 4, "first"), ("zip", 4, "iv0.2-0.8"), ("cv2", 2, "first"), ("cv2", 3, "first")}

CONFIGS, VARIANT_TO_CONFIG = {}, {}


def mask(model, n, iv):
    return tuple(int(iv[0] <= t <= iv[1]) for t in sched(model, n))


def add_cosy(name, n, K, extra):
    def cmd(s, m, k, o, n=n, extra=extra):
        return ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst", "--stage",
                "decode", "--cache_dir", f"cache/cosy2_tokens/{s}", "--cfg", "0.7", "--noise", f"utt:{k}",
                "--out_dir", o, "--nfe", str(n), *extra]
    CONFIGS[name] = dict(fam="A", model="cosyvoice2", env="env_cosy", K=K, cmd=cmd, nfe=n, extra=extra)


def add_zip(name, n, K, extra):
    def cmd(s, m, k, o, n=n, extra=extra):
        return ["-m", "zipvoice.bin.infer_zipvoice", "--model-name", "zipvoice",
                "--model-dir", str(ROOT / "ckpt/ZipVoice/zipvoice"), "--vocoder-path", str(ROOT / "ckpt/vocos-mel-24khz"),
                "--test-list", str(ROOT / (m + ".zipvoice.tsv")), "--raw-evaluation", "True", "--guidance-scale", "1.0",
                "--seed-per-utt", "True", "--noise-key", str(k), "--skip-existing", "True", "--res-dir", str(ROOT / o),
                "--num-step", str(n), *extra]
    CONFIGS[name] = dict(fam="B", model="zipvoice", env="env_zip", K=K, cmd=cmd, nfe=n, extra=extra)


for (model, n), vs in REQUEST.items():
    for v in vs:
        mk = mask(model, n, VARIANTS[v])
        if all(mk):
            VARIANT_TO_CONFIG[(model, n, v)] = f"{model}_default_n{n}"
            continue
        name = f"{model}_n{n}_m{''.join(map(str, mk))}"
        VARIANT_TO_CONFIG[(model, n, v)] = name
        K = list(range(8)) if (model, n, v) in CORE else [0]
        if name in CONFIGS:
            CONFIGS[name]["K"] = sorted(set(CONFIGS[name]["K"]) | set(K))
            continue
        iv = VARIANTS[v]
        if model == "cv2":
            add_cosy(name, n, K, ["--cfg_interval", f"{iv[0]},{iv[1]}"])
        else:
            add_zip(name, n, K, ["--cfg-interval", f"{iv[0]},{iv[1]}"])
for n in [2, 3]:
    for ts in [0.2, 0.4]:
        add_cosy(f"cv2_ramp{ts}_n{n}", n, [0], ["--cfg_ramp", str(ts)])
        VARIANT_TO_CONFIG[("cv2", n, f"ramp{ts}")] = f"cv2_ramp{ts}_n{n}"
add_cosy("cv2_default_n4_hard", 4, [0], [])  # CosyVoice2 NFE4 default on hard (en/zh exist from E5)

# defaults / references (existing dirs)
DEFAULT = {"cv2_default_n2": "cv2_nfe2", "cv2_default_n3": "cv2_nfe3", "cv2_default_n10": "cv2_nfe10",
           "zip_default_n4": "zip_nfe4", "zip_default_n16": "zip_nfe16"}
E1_ONLY = {"zip_default_n2": "out/E1/B/zipvoice/nfe2/cfg1.0/utt0", "zip_default_n8": "out/E1/B/zipvoice/nfe8/cfg1.0/utt0"}
TEACHER = {"A": "cv2_default_n10", "B": "zip_default_n16"}


def wav_dir(name, k, s):
    if name in DEFAULT:
        return _e1_hard(DEFAULT[name]) if s == "hard" else e3_wav_dir(DEFAULT[name], k, s)
    if name in E1_ONLY:
        return ROOT / E1_ONLY[name] / s
    if name == "cv2_default_n4":
        return (ROOT / "out/E5/A/cosyvoice2/cv2_euler_n4" / f"utt{k}" / s) if s != "hard" else \
            wav_dir("cv2_default_n4_hard", 0, "hard")
    c = CONFIGS[name]
    return ROOT / "out" / "E7" / c["fam"] / c["model"] / name / f"utt{k}" / s


def _e1_hard(e3name):
    from e3_jobs import CONFIGS as E3C
    return ROOT / E3C[e3name]["e1"] / "hard"


def n_lines(meta):
    return sum(1 for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip())


def sets_for(name, k):
    """8-seed runs use en/zh only (geometry); seed 0 uses all three sets. cv2_default_n4_hard: hard only."""
    if name == "cv2_default_n4_hard":
        return [x for x in SETS if x[0] == "hard"]
    return SETS if k == 0 else SETS[:2]
