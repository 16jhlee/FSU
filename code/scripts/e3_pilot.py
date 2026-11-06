"""E3 pilot (v2 §2.2) generation: en 20 utterances, reference NFE, noise seeds k=0..7.
k=0 is regenerated on purpose and compared to the E1 wav as a determinism check.
Output: out/E3/pilot/<family>/<model>/nfe<N>/<cfg>/utt<k>/en/<utt>.wav"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e1_jobs import ENVS, ROOT  # noqa: E402

META = "data/lists/en_20"
K = range(8)
zipdir = ROOT / "repos" / "ZipVoice"


def jobs():
    for k in K:
        yield (f"cv2_k{k}", "env_cosy", ROOT,
               ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", META + ".meta.lst",
                "--stage", "decode", "--cache_dir", "cache/cosy2_tokens/en", "--nfe", "10", "--cfg", "0.7",
                "--noise", f"utt:{k}", "--out_dir"], f"out/E3/pilot/A/cosyvoice2/nfe10/cfg0.7/utt{k}/en", {})
        yield (f"zip_k{k}", "env_zip", zipdir,
               ["-m", "zipvoice.bin.infer_zipvoice", "--model-name", "zipvoice",
                "--model-dir", str(ROOT / "ckpt/ZipVoice/zipvoice"), "--vocoder-path", str(ROOT / "ckpt/vocos-mel-24khz"),
                "--test-list", str(ROOT / (META + ".zipvoice.tsv")), "--raw-evaluation", "True", "--num-step", "16",
                "--guidance-scale", "1.0", "--seed-per-utt", "True", "--noise-key", str(k), "--skip-existing", "True",
                "--res-dir"], f"out/E3/pilot/B/zipvoice/nfe16/cfg1.0/utt{k}/en", {"PYTHONPATH": str(zipdir)})
        yield (f"dmoT_k{k}", "env_f5dmo", ROOT,
               ["scripts/run_dmo.py", "--meta", str(ROOT / (META + ".meta.lst")), "--mode", "teacher",
                "--teacher_steps", "32", "--noise", f"utt:{k}", "--out_dir"],
               f"out/E3/pilot/C/dmo_teacher/nfe32/cfg2.0/utt{k}/en", {})


def main():
    log = ROOT / "logs" / "E3"
    log.mkdir(parents=True, exist_ok=True)
    for name, env, cwd, cmd, out, extra in jobs():
        o = ROOT / out
        if len(list(o.glob("*.wav"))) >= 20:
            continue
        e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_DISABLE_SYMLINKS_WARNING="1", **extra)
        with open(log / f"pilot_{name}.log", "a", encoding="utf-8") as f:
            rc = subprocess.call([ENVS[env]] + cmd + [str(o)], cwd=cwd, env=e, stdout=f, stderr=subprocess.STDOUT)
        print(name, rc, len(list(o.glob("*.wav"))), flush=True)


if __name__ == "__main__":
    main()
