"""E10 smoke test (tools env): job enumeration + 3-utterance runs of the new sampler options, outputs out/smoke/E10.
  --cfg_steps 0.7,0.7 must equal the official NFE 2 (bit-identical wav); --cfg_steps 0,0.7 must equal R (ours);
  IndexTTS-2 --solver mid1 runs; eval_metrics werwh runs on zh.
python scripts/e10_smoke.py
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import e10_jobs as J  # noqa: E402
from e1_jobs import ENVS  # noqa: E402

ROOT = J.ROOT
OUT = ROOT / "out" / "smoke" / "E10"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
ENVS["env_index"] = ENVS["env_cosy"].replace("env_cosy", "env_index")


def run(py, args, cwd=ROOT):
    print(" ".join(args), flush=True)
    r = subprocess.call([ENVS[py]] + args, cwd=cwd, env=env)
    assert r == 0, r


def same(a, b):
    out = []
    for f in sorted(Path(a).glob("*.wav")):
        x, y = sf.read(f)[0], sf.read(Path(b) / f.name)[0]
        out.append(float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf"))
    return out


def main():
    n = {}
    for name, c in J.CONFIGS.items():
        for k in c["K"]:
            for s, meta, _ in J.sets_for(name, k):
                n[J.BLOCK.get(name, "prep")] = n.get(J.BLOCK.get(name, "prep"), 0) + (0 if c.get("prep") or c.get("evalonly") else J.n_lines(meta))
    print("wavs to generate per block:", n)
    meta = "data/lists/en_3.meta.lst"
    cv2 = ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", meta, "--stage", "decode",
           "--cache_dir", "cache/cosy2_tokens/en", "--noise", "utt:0", "--cfg", "0.7", "--nfe", "2"]
    for tag, extra in [("off", []), ("steps", ["--cfg_steps", "0.7,0.7"]), ("R", ["--cfg_interval", "0.1,1"]),
                       ("steps0", ["--cfg_steps", "0,0.7"]), ("neg", ["--cfg_steps=-0.7,0.7"])]:
        run("env_cosy", cv2 + ["--out_dir", str((OUT / f"cv2_{tag}").relative_to(ROOT))] + extra)
    print("cfg_steps 0.7,0.7 vs official max|diff|:", same(OUT / "cv2_off", OUT / "cv2_steps"))
    print("cfg_steps 0,0.7 vs R max|diff|:", same(OUT / "cv2_R", OUT / "cv2_steps0"))
    print("cfg_steps -0.7,0.7 vs official max|diff| (must be > 0):", same(OUT / "cv2_off", OUT / "cv2_neg"))
    run("env_index", ["scripts/run_index.py", "--meta", meta, "--stage", "decode", "--cache_dir", "cache/index2/en",
                      "--out_dir", str(OUT / "idx_mid1"), "--noise", "utt:0", "--cfg", "0.7", "--t_span", "0,0.5,1",
                      "--solver", "mid1", "--cfg_interval", "0.1,1"])
    zh = ROOT / "data/lists/zh_all.meta.lst"
    (OUT / "zh3.meta.lst").write_text("".join(zh.read_text(encoding="utf-8").splitlines(True)[:3]), encoding="utf-8")
    run("env_cosy", ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", str(OUT / "zh3.meta.lst"),
                     "--stage", "decode", "--cache_dir", "cache/cosy2_tokens/zh", "--noise", "utt:0", "--cfg", "0.7",
                     "--nfe", "2", "--out_dir", str((OUT / "cv2_zh").relative_to(ROOT))])
    run("env_eval", ["scripts/eval_metrics.py", "--meta", str(OUT / "zh3.meta.lst"), "--wav_dir", str(OUT / "cv2_zh"),
                     "--lang", "zh", "--metrics", "wer,werwh"])
    print((OUT / "cv2_zh/_metrics/metrics.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
