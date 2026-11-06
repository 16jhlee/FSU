"""E5 (v3 §3, approved corrections) — sampler-level correction algorithms. Configuration shared by e5_gen/e5_eval/e5_analyze.

Algorithms (repo time: t=0 noise, t=1 data):
  euler  original sampler; a1 = re-interpolate x-prediction with the original noise (identical to Euler for NFE<=2,
  so A1 runs at NFE 3/4); apg = norm-preserving guidance in x-prediction space (A2-b); iv = CFG only for
  t in [0.2, 0.8] (A2-c); A3 schedules for NFE 4: early [0,.1,.25,.5,1], uniform [0,.25,.5,.75,1],
  late [0,.5,.75,.9,1] (= v3 {1,.9,.75,.5}->0 etc. converted to repo time); hybrid = first k steps of the
  official 10-step cosine schedule then one jump to t=1; A4 restart = default 2-step sampler, re-noise output to
  t_r (repo 0.7 / 0.5 = v3 0.3 / 0.5) with fresh noise hash(utt|restart|k), one more step (total NFE 3).
Seeds: 8 (k=0..7) for the core conditions (A1, A1+A2-b, A1+A3-early, A4 and their new Euler baselines), seed 0 else.
Existing Euler baselines with 8 seeds come from E1/E3 (see BASE).
"""
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from e3_jobs import wav_dir as e3_wav_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETS = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh")]
ZIPDIR = ROOT / "repos" / "ZipVoice"
K8, K1 = list(range(8)), [0]
SCHED = {"early": "0,0.1,0.25,0.5,1", "uniform": "0,0.25,0.5,0.75,1", "late": "0,0.5,0.75,0.9,1"}
COS10 = [1 - math.cos(i / 10 * math.pi / 2) for i in range(11)]

CONFIGS = {}


def cosy(name, K, *extra, cfg="0.7"):
    def cmd(s, m, k, o):
        return ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst",
                "--stage", "decode", "--cache_dir", f"cache/cosy2_tokens/{s}", "--cfg", cfg,
                "--noise", f"utt:{k}", "--out_dir", o, *extra]
    CONFIGS[name] = dict(fam="A", model="cosyvoice2", env="env_cosy", K=K, cmd=cmd, extra=list(extra), cfg=cfg)


def zipv(name, model, w, K, *extra):
    def cmd(s, m, k, o):
        return ["-m", "zipvoice.bin.infer_zipvoice", "--model-name", model,
                "--model-dir", str(ROOT / "ckpt" / "ZipVoice" / model), "--vocoder-path",
                str(ROOT / "ckpt" / "vocos-mel-24khz"), "--test-list", str(ROOT / (m + ".zipvoice.tsv")),
                "--raw-evaluation", "True", "--guidance-scale", w, "--seed-per-utt", "True", "--noise-key", str(k),
                "--skip-existing", "True", "--res-dir", str(ROOT / o), *extra]
    CONFIGS[name] = dict(fam="B", model=model, env="env_zip", K=K, cmd=cmd, extra=list(extra), cfg=w)


# ---------------- A: CosyVoice2 ----------------
cosy("cv2_euler_n4", K8, "--nfe", "4")                                   # new Euler baseline (A3 default)
cosy("cv2_a1_n3", K8, "--nfe", "3", "--sampler", "a1")
cosy("cv2_a1_n4", K8, "--nfe", "4", "--sampler", "a1")
cosy("cv2_a1apg_n3", K8, "--nfe", "3", "--sampler", "a1", "--cfg_mode", "apg")
cosy("cv2_a1apg_n4", K8, "--nfe", "4", "--sampler", "a1", "--cfg_mode", "apg")
cosy("cv2_a1_early", K8, "--t_span", SCHED["early"], "--sampler", "a1")
cosy("cv2_rs07", K8, "--nfe", "2", "--restart_tr", "0.7")
cosy("cv2_rs05", K8, "--nfe", "2", "--restart_tr", "0.5")
for w in ["0", "0.3", "1.0", "1.5"]:                                     # A2-a / E2 grid (0.7 = E1)
    for n in [2, 10]:
        cosy(f"cv2_cfg{w}_n{n}", K1, "--nfe", str(n), cfg=w)
for n in [2, 3, 10]:
    cosy(f"cv2_apg_n{n}", K1, "--nfe", str(n), "--cfg_mode", "apg")      # A2-b
    cosy(f"cv2_iv_n{n}", K1, "--nfe", str(n), "--cfg_interval", "0.2,0.8")  # A2-c
for sname, ts in SCHED.items():
    cosy(f"cv2_euler_{sname}", K1, "--t_span", ts)                       # A3 Euler
    if sname != "early":
        cosy(f"cv2_a1_{sname}", K1, "--t_span", ts, "--sampler", "a1")   # A3 A1 (early is 8-seed above)
for kk in [1, 3, 5, 8]:                                                  # A3 hybrid
    cosy(f"cv2_hybrid_k{kk}", K1, "--t_span", ",".join(f"{x:.6f}" for x in COS10[:kk + 1] + [1.0]))

# ---------------- B: ZipVoice teacher (w 1.0) ----------------
zipv("zip_a1_n4", "zipvoice", "1.0", K8, "--num-step", "4", "--sampler", "a1")
zipv("zip_a1_early", "zipvoice", "1.0", K8, "--t-span", SCHED["early"], "--sampler", "a1")
for sname, ts in SCHED.items():
    zipv(f"zip_euler_{sname}", "zipvoice", "1.0", K1, "--t-span", ts)
    if sname != "early":
        zipv(f"zip_a1_{sname}", "zipvoice", "1.0", K1, "--t-span", ts, "--sampler", "a1")

# ---------------- B: ZipVoice-Distill (w 3.0) ----------------
zipv("zipd_euler_n3", "zipvoice_distill", "3.0", K8, "--num-step", "3")
zipv("zipd_a1_n3", "zipvoice_distill", "3.0", K8, "--num-step", "3", "--sampler", "a1")
zipv("zipd_a1_n4", "zipvoice_distill", "3.0", K8, "--num-step", "4", "--sampler", "a1")
zipv("zipd_rs07", "zipvoice_distill", "3.0", K8, "--num-step", "2", "--restart-tr", "0.7")
zipv("zipd_rs05", "zipvoice_distill", "3.0", K8, "--num-step", "2", "--restart-tr", "0.5")

# existing 8-seed Euler baselines / references (E1 k=0, E3 k=1..7)
BASE = {"cv2_euler_n10": "cv2_nfe10", "cv2_euler_n3": "cv2_nfe3", "cv2_euler_n2": "cv2_nfe2",
        "zip_euler_n16": "zip_nfe16", "zip_euler_n4": "zip_nfe4",
        "zipd_euler_n4": "zipd_nfe4", "zipd_euler_n2": "zipd_nfe2"}

# network evaluations per utterance (CFG doubles calls for CosyVoice2 and the ZipVoice teacher; distill has none)
def nfe_calls(name):
    c = CONFIGS.get(name)
    ex = " ".join(c["extra"]) if c else ""
    if "--t_span" in ex or "--t-span" in ex:
        ts = c["extra"][c["extra"].index("--t_span" if "--t_span" in ex else "--t-span") + 1]
        n = len(ts.split(",")) - 1
    elif name in BASE:
        n = int(name.rsplit("_n", 1)[1])
    else:
        flag = "--nfe" if "--nfe" in c["extra"] else "--num-step"
        n = int(c["extra"][c["extra"].index(flag) + 1])
    if "restart" in ex:
        n += 1
    fam_cfg = not name.startswith("zipd")
    return n, n * (2 if fam_cfg else 1)


def wav_dir(name, k, s):
    if name in BASE:
        return e3_wav_dir(BASE[name], k, s)
    c = CONFIGS[name]
    return ROOT / "out" / "E5" / c["fam"] / c["model"] / name / f"utt{k}" / s


def n_lines(meta):
    return sum(1 for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip())
