"""E8'-10 (v9.1 addendum) 2026 inference-time baselines. Formulas / code lines: reports/E8p10_baselines2026.md;
patches patches/CosyVoice_E8p10.diff, patches/ZipVoice_E8p10.diff.
CosyVoice2 / CosyVoice3: NFE 2/3/4/10, en/zh 300, seed 0; CosyVoice2 NFE 2 also seeds 1-7 (geometry).
  tfr   TFR (2607.07665)                     pci   PCI (2610.04757)
  tvs   TV-CFG symmetric (2509.22007)       tvu   TV-CFG stepup          tvsf  TV-CFG symmetric, last-step delta fixed (appendix)
  jrr   JRR-CFG (2606.25672) all steps      jrrs  JRR on the first ceil(N/4) steps (sparse; F5 0:8/32 ratio)
  Rpci  R (ours) + PCI                       Rtfr  R (ours) + TFR
  base / R: aliases of existing runs where they exist (E8'-2 / E7 / E8'-1), generated otherwise.
ZipVoice-Distill NFE 4/8 (w 3.0) and ZipVoice teacher NFE 4 (w 1.0): base vs PCI (PCI paper setting).
Outputs out/E8p10/<config>/utt<k>/<set>/ (junction to C:\\speechFM_store\\out\\E8p10).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e8p8_jobs as P8  # noqa: E402

P = P8.P
ROOT = P.ROOT
SETS = [x for x in P.SEEDTTS if x[0] in ("en", "zh")]
CONFIGS, ALIAS, LABEL = {}, {}, {}
CK = {"cv2": "ckpt/CosyVoice2-0.5B", "cv3": "ckpt/Fun-CosyVoice3-0.5B"}
R_ARGS = ["--cfg_interval", "0.1,1.0"]
JRR_W = "0.70,1.20,0.95"  # paper Table 1, CosyVoice2 (also used for CosyVoice3: no published weights)


def cache(model, s):
    return f"cache/cosy2_tokens/{s}" if model == "cv2" else P.CV3_CACHE[s]


def add(name, model, label, n, extra, K=(0,), alias=None):
    LABEL[name] = (model, label, n)
    if alias:
        ALIAS[name] = alias
        return

    def cmd(s, m, k, o):
        return ["scripts/run_cosy.py", "--model_dir", CK[model], "--meta", m + ".meta.lst", "--stage", "decode",
                "--cache_dir", cache(model, s), "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7", "--nfe", str(n), *extra]
    CONFIGS[name] = dict(env="env_cosy", K=list(K), cmd=cmd, sets=SETS, extra=extra)


BASE = {("cv2", 2): "e82_cv2_all_n2_def", ("cv2", 3): "e82_cv2_all_n3_def", ("cv2", 10): "e82_cv2_all_n10_def",
        ("cv2", 4): "e7:cv2_default_n4", ("cv3", 2): "e81_cv3_n2_def"}
RCFG = {("cv2", 2): "e82_cv2_all_n2_R", ("cv2", 3): "e82_cv2_all_n3_R", ("cv2", 10): "e82_cv2_all_n10_R",
        ("cv2", 4): "e7:cv2_n4_m0011", ("cv3", 2): "e81_cv3_n2_R"}
METHODS = [("tfr", "TFR", ["--tfr", "1"]), ("pci", "PCI", ["--pci", "1"]),
           ("tvs", "TV-CFG symmetric", ["--tv", "symmetric"]), ("tvu", "TV-CFG stepup", ["--tv", "stepup"]),
           ("tvsf", "TV-CFG symmetric (last-delta fix)", ["--tv", "symmetric_fix"]),
           ("jrr", "JRR-CFG", ["--jrr", JRR_W]), ("jrrs", "JRR-CFG sparse", None),
           ("Rpci", "R (ours) + PCI", R_ARGS + ["--pci", "1"]), ("Rtfr", "R (ours) + TFR", R_ARGS + ["--tfr", "1"])]
GEOM_K = list(range(8))
for model in ("cv2", "cv3"):
    for n in (2, 3, 4, 10):
        add(f"{model}_base_n{n}", model, "baseline", n, [], alias=BASE.get((model, n)))
        add(f"{model}_R_n{n}", model, "R (ours)", n, R_ARGS, alias=RCFG.get((model, n)))
        for key, label, extra in METHODS:
            if key == "jrrs":
                extra = ["--jrr", JRR_W, "--jrr_steps", str(math.ceil(n / 4))]
            geom = model == "cv2" and n == 2 and key not in ("tvsf", "jrrs")
            add(f"{model}_{key}_n{n}", model, label, n, extra, K=GEOM_K if geom else (0,))

# ZipVoice-Distill / ZipVoice teacher (PCI paper setting)
ZSETS = SETS


def zipv(name, model, w, n, extra):
    def cmd(s, m, k, o):
        return ["-m", "zipvoice.bin.infer_zipvoice", "--model-name", model,
                "--model-dir", str(ROOT / "ckpt" / "ZipVoice" / model), "--vocoder-path",
                str(ROOT / "ckpt" / "vocos-mel-24khz"), "--test-list", str(ROOT / (m + ".zipvoice.tsv")),
                "--raw-evaluation", "True", "--guidance-scale", w, "--seed-per-utt", "True", "--noise-key", str(k),
                "--skip-existing", "True", "--res-dir", o, "--num-step", str(n), *extra]
    LABEL[name] = (model, "PCI" if extra else "baseline", n)
    CONFIGS[name] = dict(env="env_zip", K=[0], cmd=cmd, sets=ZSETS, extra=extra)


for n in (4, 8):
    zipv(f"zipd_base_n{n}", "zipvoice_distill", "3.0", n, [])
    zipv(f"zipd_pci_n{n}", "zipvoice_distill", "3.0", n, ["--pci", "True"])
zipv("zip_base_n4", "zipvoice", "1.0", 4, [])
zipv("zip_pci_n4", "zipvoice", "1.0", 4, ["--pci", "True"])


# v9 §2.1 boundary (speech): IndexTTS-2 NFE 2 with t1 = 0.2, baseline vs R (adds to E8'-8 §4 t1 = 0.3 and the G4 sweep)
for _key, _extra in [("def", []), ("R", R_ARGS)]:
    def _icmd(s, m, k, o, _extra=_extra):
        return ["scripts/run_index.py", "--meta", m + ".meta.lst", "--stage", "decode", "--cache_dir", f"cache/index2/{s}",
                "--out_dir", o, "--noise", f"utt:{k}", "--cfg", "0.7", "--t_span", "0,0.2,1", *_extra]
    LABEL[f"idx_t02_{_key}"] = ("idx", "R (ours)" if _extra else "baseline", 2)
    CONFIGS[f"idx_t02_{_key}"] = dict(env="env_index", K=[0], cmd=_icmd, sets=SETS, extra=_extra)

# JRR paper reproduction first (addendum §1): CosyVoice2 10 steps on LibriSpeech-PC test-clean (1,127 pairs),
# official CFG 0.7 vs JRR (paper: SIM 0.6561 -> 0.6690). List: scripts/build_lspc_list.py
LSPC = [("lspc", "data/lists/lspc_all", "en")]
CONFIGS["prep_cv2_tokens_lspc"] = dict(env="env_cosy", K=[0], sets=LSPC, prep=True, cmd=lambda s, m, k, o: [
    "scripts/run_cosy.py", "--model_dir", CK["cv2"], "--meta", m + ".meta.lst", "--stage", "tokens",
    "--cache_dir", "cache/cosy2_tokens/lspc"])
for _key, _extra in [("base", []), ("jrr", ["--jrr", JRR_W])]:
    def _cmd(s, m, k, o, _extra=_extra):
        return ["scripts/run_cosy.py", "--model_dir", CK["cv2"], "--meta", m + ".meta.lst", "--stage", "decode",
                "--cache_dir", "cache/cosy2_tokens/lspc", "--noise", f"utt:{k}", "--out_dir", o, "--cfg", "0.7",
                "--nfe", "10", *_extra]
    LABEL[f"lspc_cv2_{_key}_n10"] = ("cv2", "JRR-CFG" if _extra else "baseline", 10)
    CONFIGS[f"lspc_cv2_{_key}_n10"] = dict(env="env_cosy", K=[0], cmd=_cmd, sets=LSPC, extra=_extra)


def wav_dir(name, k, s):
    if name in ALIAS:
        a = ALIAS[name]
        return P8.wav_dir(a, k, s) if not a.startswith("e7:") else P.e7_wav_dir(a[3:], k, s)
    return ROOT / "out" / "E8p10" / name / f"utt{k}" / s


def sets_for(name, k):
    return CONFIGS[name]["sets"] if name in CONFIGS else SETS


n_lines = P.n_lines
