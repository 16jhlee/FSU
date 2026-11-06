"""E0 table: mean +- bootstrap 95% CI (10k resamples) per model vs paper numbers."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [  # (label, dir, paper WER %, paper SIM-o, paper UTMOS, paper source)
    ("A CosyVoice2 (NFE10, cfg0.7)", "out/E0/A/cosyvoice2/nfe10/cfg0.7/default", 2.57, 0.652, None, "CosyVoice2 paper Tab.6"),
    ("A Fun-CosyVoice3-2512 (NFE10, cfg0.7)", "out/E0/A/cosyvoice3/nfe10/cfg0.7/default", 2.24, 0.718, None, "CosyVoice README"),
    ("B ZipVoice (NFE16, w1.0)", "out/E0/B/zipvoice/nfe16/cfg1.0/seed666", 1.70, 0.697, 3.82, "ZipVoice paper Tab.I"),
    ("B ZipVoice-Distill (NFE8, w3.0)", "out/E0/B/zipvoice_distill/nfe8/cfg3.0/seed666", 1.62, 0.670, 3.91, "ZipVoice paper Tab.I"),
    ("C F5-TTS v0 Base (NFE32, cfg2, sway-1)", "out/E0/C/f5tts_v0/nfe32/cfg2.0/utt", 1.83, 0.67, None, "F5-TTS paper Tab.2"),
    ("C F5-TTS v1 Base (NFE32, cfg2, sway-1)", "out/E0/C/f5tts_v1/nfe32/cfg2.0/utt", None, None, None, "no published value"),
    ("C DMOSpeech2 student-only (NFE4)", "out/E0/C/dmospeech2_student/nfe4/cfgfused/utt", 1.752, 0.698, None, "DMOSpeech2 paper Tab.2"),
    ("C DMOSpeech2 teacher-guided (official)", "out/E0/C/dmospeech2_tg/nfe16/cfg2.0/utt", 1.738, 0.699, None, "DMOSpeech2 paper Tab.2"),
]
rng = np.random.default_rng(0)


def ci(x, scale=1.0):
    x = np.asarray(x, float) * scale
    bs = rng.choice(x, (10000, len(x))).mean(1)
    return f"{x.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]" if scale == 1 else \
        f"{x.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"


def fmt(v, d):
    return "—" if v is None else f"{v:.{d}f}"


rows = []
for lab, d, pw, ps, pu, src in RUNS:
    df = pd.read_csv(ROOT / d / "_metrics" / "metrics.csv")
    w, s = df.wer.mean() * 100, df.sim_wavlm.mean()
    ok = "—" if pw is None else ("✅" if abs(w - pw) <= 1.0 and abs(s - ps) * 100 <= 1.0 else "❌")
    rows.append([lab, len(df), f"{fmt(pw, 2)} → {ci(df.wer, 100)}", f"{fmt(ps, 3)} → {ci(df.sim_wavlm)}",
                 ci(df.sim_ecapa), ci(df.sim_campp), f"{fmt(pu, 2)} → {df.utmos.mean():.2f}", ok, src])
hdr = ["model (setting)", "N", "WER % (paper -> ours [95% CI])", "SIM-o WavLM (paper -> ours)", "SIM ECAPA", "SIM CAM++",
       "UTMOS (paper -> ours)", "within ±1", "paper source"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
print("\n".join(lines))
