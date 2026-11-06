"""v3 M-1 check: is the 1-step output seed-independent (= conditional mean)?
en 20 utterances (pilot list), NFE 1, seeds k=0..7, for CosyVoice2, ZipVoice teacher, ZipVoice-Distill.
Output: out/E3b/m1/<family>/<model>/nfe1/<cfg>/utt<k>/en/"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS, ROOT  # noqa: E402

META = "data/lists/en_20"
zipdir = ROOT / "repos" / "ZipVoice"


def jobs():
    for k in range(8):
        yield ("env_cosy", ROOT, ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", META + ".meta.lst",
                                  "--stage", "decode", "--cache_dir", "cache/cosy2_tokens/en", "--nfe", "1", "--cfg", "0.7",
                                  "--noise", f"utt:{k}", "--out_dir"], f"out/E3b/m1/A/cosyvoice2/nfe1/cfg0.7/utt{k}/en", {})
        for model, w in [("zipvoice", "1.0"), ("zipvoice_distill", "3.0")]:
            yield ("env_zip", zipdir,
                   ["-m", "zipvoice.bin.infer_zipvoice", "--model-name", model,
                    "--model-dir", str(ROOT / "ckpt/ZipVoice" / model), "--vocoder-path", str(ROOT / "ckpt/vocos-mel-24khz"),
                    "--test-list", str(ROOT / (META + ".zipvoice.tsv")), "--raw-evaluation", "True", "--num-step", "1",
                    "--guidance-scale", w, "--seed-per-utt", "True", "--noise-key", str(k), "--skip-existing", "True",
                    "--res-dir"], f"out/E3b/m1/B/{model}/nfe1/cfg{w}/utt{k}/en", {"PYTHONPATH": str(zipdir)})


for env, cwd, cmd, out, extra in jobs():
    o = ROOT / out
    if len(list(o.glob("*.wav"))) >= 20:
        continue
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1", **extra)
    with open(ROOT / "logs" / "E3" / "m1_gen.log", "a", encoding="utf-8") as f:
        rc = subprocess.call([ENVS[env]] + cmd + [str(o)], cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT)
    print(out, rc, len(list(o.glob("*.wav"))), flush=True)
