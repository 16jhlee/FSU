"""E17 (v10.10 §2) regression check: the E17 command builders with the official / FSU arguments on the first 3 utterances of
every condition family -> WAV data chunk vs the existing comparator run (identical expected; otherwise max |d| of the
waveform and of the log-mel are reported) -> results/E17_regress.csv
python scripts/e17_regress.py   (env_eval; the decoders run through their envs; GLM = heavy -> nothing else on the GPU)
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import e11d_jobs as J11D  # noqa: E402
import e15_seed_subset as SS  # noqa: E402
import e17_jobs as J  # noqa: E402
import e8p_jobs as P  # noqa: E402
import e8v8r_jobs as JR  # noqa: E402
import e8v8s_jobs as JS  # noqa: E402

ROOT = J.ROOT
ENVS = Path(sys.executable).resolve().parents[1]
O, F = ["--nfe", "2"], ["--nfe", "2", "--cfg_interval", J.R_IV]
AN = ["--cfg_branch", "allnull"]
# (family, which, env, cmd, set, reference dir)
CHECKS = []
for m in ("cv2", "cv3"):
    for w, ex, ref in (("official", O, f"e85_{m}_n2_def"), ("fsu", F, f"e85_{m}_n2_R")):
        CHECKS.append((f"mm_{m}", w, "env_cosy", J.cv_mm(m, ex), "mm_de", P.wav_dir(ref, 0, "mm_de")))
for mode in ("off", "str"):
    for w, ex, ref in (("official", O, f"s_{mode}_n2_def"), ("fsu", F, f"s_{mode}_n2_R")):
        CHECKS.append((f"long_{mode}_cv2", w, "env_cosy", J.long(mode, ex), "long_en", JS.wav_dir(ref, 0, "long_en")))
for s in ("rb_len1", "rb_snr10"):
    for w, ex, ref in (("official", O, "r_n2_def"), ("fsu", F, "r_n2_R")):
        CHECKS.append(("rb_cv2", w, "env_cosy", J.rb(ex), s, JR.wav_dir(ref, 0, s)))
for w, ex, ref in (("official", [*O, *AN], "l_allnull_n2"), ("fsu", ["--nfe", "2", *AN, "--cfg_interval", J.R_IV], "l_allnullR_n2")):
    CHECKS.append(("mmglm_an", w, "env_glm", J.glm_de(ex), "mm_de", J11D.wav_dir(ref, 0, "mm_de")))


def lists3(s):
    src = {"mm_de": "data/lists/mm_de", "long_en": "data/lists/long_en"}.get(s, f"data/lists/{s}")
    dst = ROOT / "out" / "smoke" / "E17_regress" / "lists" / f"{s}_3"
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = (ROOT / (src + ".meta.lst")).read_text(encoding="utf-8").splitlines()[:3]
    dst.with_suffix(".meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(dst.relative_to(ROOT)).replace("\\", "/")


def mel(x, sr):
    m = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, hop_length=256, n_mels=80)(torch.tensor(x, dtype=torch.float32))
    return torch.log(m.clamp_min(1e-5)).numpy()


if __name__ == "__main__":
    rows = []
    for fam, w, env, cmd, s, ref in CHECKS:
        meta = lists3(s)
        o = ROOT / "out" / "smoke" / "E17_regress" / f"{fam}_{w}_{s}"
        if o.exists():
            shutil.rmtree(o)
        out_arg = str(o.relative_to(ROOT)) if env == "env_cosy" else str(o)
        r = subprocess.run([str(ENVS / env / "python.exe"), *cmd(s, meta, 0, out_arg)], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-1500:], r.stderr[-1500:])
        for u in SS.utts(meta):
            a, b = o / f"{u}.wav", Path(ref) / f"{u}.wav"
            row = dict(family=fam, which=w, set=s, rc=r.returncode, utt=u, reference=str(b.parent),
                       identical=a.exists() and b.exists() and SS.data(a) == SS.data(b), max_abs_wav="", max_abs_logmel="")
            if a.exists() and b.exists() and not row["identical"]:
                xa, sr = sf.read(a)
                xb, _ = sf.read(b)
                n = min(len(xa), len(xb))
                row["max_abs_wav"] = float(np.abs(xa[:n] - xb[:n]).max()) if len(xa) == len(xb) else f"len {len(xa)} vs {len(xb)}"
                row["max_abs_logmel"] = float(np.abs(mel(xa[:n], sr) - mel(xb[:n], sr)).max())
            rows.append(row)
            print(fam, w, s, r.returncode, u[:28], row["identical"], row["max_abs_wav"], row["max_abs_logmel"], flush=True)
    with open(ROOT / "results/E17_regress.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    bad = [r for r in rows if not r["identical"]]
    print("STOP: mismatch" if bad else "all identical", len(rows) - len(bad), "/", len(rows))
