"""DMOSpeech2 runner (family C student), wrapping DMOSpeech2/src/infer.py::DMOInference.generate.

Modes:
  official      teacher-guided: teacher_steps=16, teacher_stopping_time=0.07, student_start_step=1
  student       student only:   teacher_steps=0, student_start_step=0, student grid --grid (pre-sway)
Noise: torch.manual_seed(hash(utt|noise|K)) before each utterance. That seeds the duration
predictor (argmax by default -> deterministic anyway), the teacher noise and every student
re-noising draw, so for a fixed K all NFE settings start from the same z at t=0.
Duration: official GRPO duration predictor (model_1500.pt); value is logged to _lengths.tsv.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "repos" / "DMOSpeech2" / "src"
sys.path.insert(0, str(SRC))  # vendored f5_tts must shadow the pip f5-tts package
os.chdir(SRC)

import torch  # noqa: E402
import torchaudio  # noqa: E402
import f5_tts  # noqa: E402
# vendored f5_tts is a namespace package and would merge with the pip (editable) F5-TTS;
# restrict it to the vendored copy only.
f5_tts.__path__ = [str(SRC / "f5_tts")]
from infer import DMOInference  # noqa: E402
import f5_tts.model  # noqa: E402
assert str(SRC) in f5_tts.model.__file__, f5_tts.model.__file__


def h(s, mod=2**31 - 1):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16) % mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["official", "student", "teacher"], required=True)
    ap.add_argument("--grid", default="0,0.25,0.5,0.75", help="student time grid before sway")
    ap.add_argument("--teacher_steps", type=int, default=32, help="teacher mode: Euler steps")
    ap.add_argument("--cfg", type=float, default=2.0, help="teacher mode: cfg_strength")
    ap.add_argument("--eta", type=float, default=0.0, help="teacher mode: 0 = deterministic ODE")
    ap.add_argument("--noise", default="utt")
    a = ap.parse_args()
    ck = ROOT / "ckpt" / "DMOSpeech2"
    dmo = DMOInference(student_checkpoint_path=str(ck / "model_85000.pt"),
                       duration_predictor_path=str(ck / "model_1500.pt"), device="cuda")
    if a.mode == "student":
        dmo.student_steps_override = [float(x) for x in a.grid.split(",")]
    k = a.noise.split(":")[1] if ":" in a.noise else "0"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "_lengths.tsv", "a", encoding="utf-8")
    rows = [l.split("|") for l in Path(a.meta).read_text(encoding="utf-8").splitlines() if l.strip()]
    for utt, ptext, pwav, text, *_ in rows:
        if (out / f"{utt}.wav").exists():
            continue
        torch.manual_seed(h(f"{utt}|noise|{k}"))
        dur = dmo.predict_duration(pwav, text, ptext)
        torch.manual_seed(h(f"{utt}|noise|{k}"))
        if a.mode == "official":
            wav = dmo.generate(text, pwav, prompt_text=ptext, duration=dur)
        elif a.mode == "teacher":
            # internal teacher (guidance_model.real_unet) alone: full Euler ODE, sway -1, CFG on;
            # student_start_step = len(grid) -> zero student steps
            wav = dmo.generate(text, pwav, prompt_text=ptext, duration=dur, teacher_steps=a.teacher_steps,
                               teacher_stopping_time=1.0, student_start_step=4, eta=a.eta, cfg_strength=a.cfg)
        else:
            wav = dmo.generate(text, pwav, prompt_text=ptext, duration=dur, teacher_steps=0, student_start_step=0)
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        torchaudio.save(str(out / f"{utt}.wav"), wav, 24000)
        tag = f"T{a.teacher_steps}/cfg{a.cfg}/eta{a.eta}" if a.mode == "teacher" else a.grid
        log.write(f"{utt}\t{a.mode}\t{tag}\t{a.noise}\t{dur}\t{wav.shape[-1]}\n")
        log.flush()
    log.close()


if __name__ == "__main__":
    main()
