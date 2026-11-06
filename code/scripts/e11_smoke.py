"""E11 gate 5 / patch check (tools env): GLM-TTS on 3 en utterances (data/lists/en_3).
  (a) tokens stage; (b) official decode twice -> bit-identical (determinism with injected noise hash(utt|noise|0));
  (c) patched path forced with --cfg_interval 0,1 (all evaluations keep w) == official path (patch reproduces the
      official sampler); (d) R, mid+R, allnull run; R differs from official.
python scripts/e11_smoke.py   -> printed max |diff| per utterance; outputs out/smoke/E11
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PY = str(Path(sys.executable).resolve().parents[1] / "env_glm" / "python.exe")
OUT = ROOT / "out" / "smoke" / "E11"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
META = "data/lists/en_3.meta.lst"


def run(*args):
    r = subprocess.call([PY, "scripts/run_glm.py", "--meta", META, *args], cwd=ROOT, env=env)
    assert r == 0, args


def diff(a, b):
    out = []
    for f in sorted((OUT / a).glob("*.wav")):
        x, y = sf.read(f)[0], sf.read(OUT / b / f.name)[0]
        out.append(float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf"))
    return out


run("--stage", "tokens", "--cache_dir", "cache/glm_tokens/en")
dec = ["--stage", "decode", "--cache_dir", "cache/glm_tokens/en", "--noise", "utt:0"]
run(*dec, "--out_dir", str(OUT / "off_a"))
run(*dec, "--out_dir", str(OUT / "off_b"))
run(*dec, "--out_dir", str(OUT / "patched_full"), "--cfg_interval", "0,1")
run(*dec, "--out_dir", str(OUT / "R_n2"), "--nfe", "2", "--cfg_interval", "0.1,1")
run(*dec, "--out_dir", str(OUT / "off_n2"), "--nfe", "2")
run(*dec, "--out_dir", str(OUT / "midR"), "--nfe", "1", "--solver", "midpoint", "--cfg_interval", "0.1,1")
run(*dec, "--out_dir", str(OUT / "allnull_n2"), "--nfe", "2", "--cfg_branch", "allnull")
print("determinism official x2:", diff("off_a", "off_b"))
print("patched path (interval 0,1) vs official:", diff("off_a", "patched_full"))
print("R NFE 2 vs official NFE 2 (must differ):", diff("off_n2", "R_n2"))
print("all-null NFE 2 vs official NFE 2 (must differ):", diff("off_n2", "allnull_n2"))
print("mid+R files:", len(list((OUT / "midR").glob("*.wav"))))
