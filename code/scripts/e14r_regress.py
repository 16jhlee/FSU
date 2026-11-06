"""E14R-2 regression checks (v10.7 §2): en 3 utterances (data/lists/en_3), seed 0.
(a) default / existing settings after patches/IndexTTS_E14.diff, GLMTTS_E14.diff reproduce the earlier outputs:
    audio data chunk identical and max |log-mel difference| (reported, must be 0.000)
(b) TFR alias: TFR output vs the w = 0 output of the same decoder, max |log-mel difference| < 1e-4 -> alias.
python scripts/e14r_regress.py   (env_eval; runs the decoders through their envs) -> results/E14_rules_regress.csv
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
import e11_jobs as J11  # noqa: E402
import e13_jobs as J13  # noqa: E402
import e8p8_analyze as M8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ENVS = Path(sys.executable).resolve().parents[1]
TMP = ROOT / "out" / "smoke" / "E14R_regress"
META = "data/lists/en_3.meta.lst"
UTTS = [l.split("|")[0] for l in (ROOT / META).read_text(encoding="utf-8").splitlines() if l.strip()]


def idx(extra):
    return ["env_index", "scripts/run_index.py", "--meta", META, "--stage", "decode", "--cache_dir", "cache/index2/en",
            "--noise", "utt:0", *extra]


def glm(extra):
    return ["env_glm", "scripts/run_glm.py", "--meta", META, "--stage", "decode", "--cache_dir", "cache/glm_tokens/en",
            "--noise", "utt:0", "--cfg", "0.7", *extra]


# name -> (command, reference folder or None)
RUNS = {
    "idx_default_n2": (idx(["--nfe", "2"]), M8.W("e81_idx_n2_def", 0, "en")),
    "idx_R_n2": (idx(["--nfe", "2", "--cfg_interval", "0.1,1"]), M8.W("e81_idx_n2_R", 0, "en")),
    "idx_midR": (idx(["--nfe", "1", "--solver", "midpoint", "--cfg_interval", "0.1,1"]), M8.W("A_idx_midR", 0, "en")),
    "glm_official_n2": (glm(["--nfe", "2"]), J11.wav_dir("g_base_n2", 0, "en")),
    "glm_allnull_n2": (glm(["--nfe", "2", "--cfg_branch", "allnull"]), J13.wav_dir("ga_11", 0, "en")),
    "glm_allnull_R_n2": (glm(["--nfe", "2", "--cfg_branch", "allnull", "--cfg_interval", "0.1,1"]), J13.wav_dir("ga_01", 0, "en")),
    "glm_allnull_midR": (glm(["--nfe", "1", "--solver", "midpoint", "--cfg_interval", "0.1,1", "--cfg_branch", "allnull"]),
                         J13.wav_dir("ga_midR", 0, "en")),
    "idx_w0_n2": (idx(["--nfe", "2", "--cfg", "0"]), None),
    "idx_tfr_n2": (idx(["--nfe", "2", "--tfr", "1"]), None),
    "glm_w0_n2": (glm(["--nfe", "2", "--cfg_branch", "allnull", "--cfg", "0"]), None),
    "glm_tfr_n2": (glm(["--nfe", "2", "--cfg_branch", "allnull", "--tfr", "1"]), None),
}
PAIRS_TFR = [("idx_tfr_n2", "idx_w0_n2"), ("glm_tfr_n2", "glm_w0_n2")]


def data(p):
    x = p.read_bytes()
    return x[x.index(b"data"):]


def logmel(p):
    y, sr = sf.read(str(p), dtype="float32")
    m = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)(torch.from_numpy(y))
    return torch.log(m.clamp_min(1e-5)).numpy()


def mel_diff(a, b):
    A, B = logmel(a), logmel(b)
    if A.shape != B.shape:
        return float("inf")
    return float(np.abs(A - B).max())


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    for name, (cmd, _) in RUNS.items():
        env, script, *args = cmd
        o = TMP / name
        r = subprocess.run([str(ENVS / env / "python.exe"), script, *args, "--out_dir", str(o)], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(name, "rc", r.returncode, flush=True)
        if r.returncode:
            print(r.stderr[-2000:])
            sys.exit(1)
    rows = []
    for name, (_, ref) in RUNS.items():
        if ref is None:
            continue
        for u in UTTS:
            a, b = TMP / name / f"{u}.wav", Path(ref) / f"{u}.wav"
            rows.append(dict(check="default unchanged", run=name, ref=str(ref), utt=u, identical=data(a) == data(b),
                             max_logmel_diff=mel_diff(a, b)))
    for x, y in PAIRS_TFR:
        for u in UTTS:
            a, b = TMP / x / f"{u}.wav", TMP / y / f"{u}.wav"
            d = mel_diff(a, b)
            rows.append(dict(check="TFR alias (< 1e-4)", run=x, ref=y, utt=u, identical=data(a) == data(b), max_logmel_diff=d))
    with open(ROOT / "results/E14_rules_regress.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["check"], r["run"], r["utt"][:28], r["identical"], f"{r['max_logmel_diff']:.6f}")


if __name__ == "__main__":
    main()
