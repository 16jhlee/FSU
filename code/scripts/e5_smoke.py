"""E5 sanity checks on 3 utterances (en): patches keep defaults bit-exact, A1 == Euler at NFE<=2,
APG == standard at w=0, restart runs. Prints max |diff| per check."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS, ROOT  # noqa: E402

META = "data/lists/en_3"
OUT = ROOT / "out" / "E5" / "smoke"
ZIP = ROOT / "repos" / "ZipVoice"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")


def cosy(name, *extra):
    o = OUT / name
    subprocess.check_call([ENVS["env_cosy"], "scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta",
                           META + ".meta.lst", "--stage", "decode", "--cache_dir", "cache/cosy2_tokens/en",
                           "--noise", "utt:0", "--out_dir", str(o), *extra], cwd=ROOT, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return o


def zipv(name, model, w, *extra):
    o = OUT / name
    e = dict(env, PYTHONPATH=str(ZIP))
    subprocess.check_call([ENVS["env_zip"], "-m", "zipvoice.bin.infer_zipvoice", "--model-name", model,
                           "--model-dir", str(ROOT / "ckpt/ZipVoice" / model), "--vocoder-path",
                           str(ROOT / "ckpt/vocos-mel-24khz"), "--test-list", str(ROOT / (META + ".zipvoice.tsv")),
                           "--res-dir", str(o), "--raw-evaluation", "True", "--guidance-scale", w,
                           "--seed-per-utt", "True", "--noise-key", "0", *extra], cwd=ZIP, env=e,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return o


def diff(a, b):
    m = 0.0
    for f in sorted(Path(a).glob("*.wav")):
        x, _ = sf.read(f)
        y, _ = sf.read(Path(b) / f.name)
        m = max(m, float(np.abs(x - y).max()) if x.shape == y.shape else np.inf)
    return m


checks = [
    ("CV2 euler NFE2 vs E1", cosy("cv2_e2", "--nfe", "2"), ROOT / "out/E1/A/cosyvoice2/nfe2/cfg0.7/utt0/en"),
    ("CV2 a1 NFE2 vs euler NFE2", cosy("cv2_a1_2", "--nfe", "2", "--sampler", "a1"), OUT / "cv2_e2"),
    ("CV2 apg w0 NFE3 vs std w0", cosy("cv2_apg0", "--nfe", "3", "--cfg", "0", "--cfg_mode", "apg"),
     cosy("cv2_std0", "--nfe", "3", "--cfg", "0")),
    ("CV2 interval[0,1] NFE3 vs E1 NFE3", cosy("cv2_iv", "--nfe", "3", "--cfg_interval", "0,1"),
     ROOT / "out/E1/A/cosyvoice2/nfe3/cfg0.7/utt0/en"),
    ("CV2 a1 NFE3 vs euler NFE3 (should differ)", cosy("cv2_a1_3", "--nfe", "3", "--sampler", "a1"),
     ROOT / "out/E1/A/cosyvoice2/nfe3/cfg0.7/utt0/en"),
    ("CV2 restart runs (vs NFE3, differs)", cosy("cv2_rs", "--nfe", "2", "--restart_tr", "0.5"),
     ROOT / "out/E1/A/cosyvoice2/nfe3/cfg0.7/utt0/en"),
    ("ZIP teacher euler NFE4 vs E1", zipv("zip_e4", "zipvoice", "1.0", "--num-step", "4"),
     ROOT / "out/E1/B/zipvoice/nfe4/cfg1.0/utt0/en"),
    ("ZIPD a1 NFE2 vs E1 euler NFE2", zipv("zipd_a1_2", "zipvoice_distill", "3.0", "--num-step", "2", "--sampler", "a1"),
     ROOT / "out/E1/B/zipvoice_distill/nfe2/cfg3.0/utt0/en"),
    ("ZIPD a1 NFE4 vs E1 euler NFE4 (should differ)",
     zipv("zipd_a1_4", "zipvoice_distill", "3.0", "--num-step", "4", "--sampler", "a1"),
     ROOT / "out/E1/B/zipvoice_distill/nfe4/cfg3.0/utt0/en"),
    ("ZIPD restart runs", zipv("zipd_rs", "zipvoice_distill", "3.0", "--num-step", "2", "--restart-tr", "0.5"),
     ROOT / "out/E1/B/zipvoice_distill/nfe2/cfg3.0/utt0/en"),
    ("ZIP t-span uniform NFE4 runs", zipv("zip_ts", "zipvoice", "1.0", "--t-span", "0,0.25,0.5,0.75,1"),
     ROOT / "out/E1/B/zipvoice/nfe4/cfg1.0/utt0/en"),
]
for name, a, b in checks:
    print(f"{name:45s} max|diff| = {diff(a, b):.3e}  (n={len(list(Path(a).glob('*.wav')))})", flush=True)
