"""E6 (v4 §2): vocoder upper bound. Ground-truth target recordings (Seed-TTS-eval en/zh subsets) are
   (gt)    used as-is,
   (hift)  re-synthesised through CosyVoice2's mel + HiFT (family A vocoder),
   (vocos) re-synthesised through Vocos mel-24kHz (family B/C vocoder),
then evaluated exactly like generated speech (eval_metrics.py: WER, SIM-o x3 vs the prompt, UTMOS).
Upper bound of SIM for a family = SIM-o of its vocoder resynthesis of real target speech.
Run with the tools env python:  python scripts/e6_vocoder.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS, ROOT  # noqa: E402


def key(p):  # same as resynth_prompts.key (that module imports torch, unavailable in the tools env)
    import hashlib
    return hashlib.sha1(str(Path(p).resolve()).lower().encode("utf-8")).hexdigest()[:16]

SEED = ROOT / "data/raw/seedtts/seedtts_testset"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
SETS = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh")]


def main():
    gt_metas = []
    for s, meta, _ in SETS:
        rows = [l.split("|") for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()]
        lines, out = [], ROOT / "out/E6/gt" / s
        out.mkdir(parents=True, exist_ok=True)
        for f in rows:
            gt = SEED / s / "wavs" / f"{f[0]}.wav"
            assert gt.exists(), gt
            lines.append("|".join([f[0], f[1], str(gt), f[3]]))
            if not (out / f"{f[0]}.wav").exists():
                shutil.copyfile(gt, out / f"{f[0]}.wav")
        gm = ROOT / f"data/lists/e6_gt_{s}.meta.lst"
        gm.write_text("\n".join(lines) + "\n", encoding="utf-8")
        gt_metas.append(str(gm))
    for voc, envn in [("hift", "env_cosy"), ("vocos", "env_zip")]:
        subprocess.check_call([ENVS[envn], "scripts/resynth_prompts.py", "--vocoder", voc, "--meta", *gt_metas],
                              cwd=ROOT, env=env)
        for (s, _, _), gm in zip(SETS, gt_metas):
            out = ROOT / "out/E6" / voc / s
            out.mkdir(parents=True, exist_ok=True)
            for l in Path(gm).read_text(encoding="utf-8").splitlines():
                u, _, gt, _ = l.split("|")
                src = ROOT / "data/cache/prompt_resynth" / voc / f"{key(gt)}.wav"
                if not (out / f"{u}.wav").exists():
                    shutil.copyfile(src, out / f"{u}.wav")
    for kind in ["gt", "hift", "vocos"]:
        for s, meta, lang in SETS:
            d = ROOT / "out/E6" / kind / s
            subprocess.check_call([ENVS["env_eval"], "scripts/eval_metrics.py", "--meta", meta + ".meta.lst",
                                   "--wav_dir", str(d), "--lang", lang, "--metrics", "wer,wavlm,ecapa,eres2net,utmos"],
                                  cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(kind, s, "evaluated", flush=True)


if __name__ == "__main__":
    main()
