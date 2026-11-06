"""E16 (v10.9 §2) regression check: the E16 commands on 3 utterances of the existing runs with the same sampler settings
-> the WAV data chunk must be identical (stop otherwise) -> results/E16_regress.csv
  CosyVoice 3 Official / FSU   vs out/E10/full/f_cv3_{base_n2,R_n2}/utt0/en       (data/lists/en_3)
  GLM all-null x3              vs out/E14/{allnull_n2,allnullR_n2,allnull_midR}/utt0/hard   (data/lists/hard_3)
python scripts/e16_regress.py   (env_eval; the decoders run through their envs; GLM = heavy -> nothing else on the GPU)
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e10_jobs as J10  # noqa: E402
import e14_jobs as J14  # noqa: E402
import e15_seed_subset as SS  # noqa: E402
import e16_jobs as J  # noqa: E402

ROOT = J.ROOT
ENVS = Path(sys.executable).resolve().parents[1]
CHECKS = [("cv3_base_n2", "en", "data/lists/en_3", J10.wav_dir("f_cv3_base_n2", 0, "en")),
          ("cv3_R_n2", "en", "data/lists/en_3", J10.wav_dir("f_cv3_R_n2", 0, "en"))]
CHECKS += [(f"glm_{k}", "hard", "data/lists/hard_3", J14.wav_dir(k, 0, "hard"))
           for k in ("allnull_n2", "allnullR_n2", "allnull_midR")]

if __name__ == "__main__":
    rows = []
    for name, s, meta, ref in CHECKS:
        c = J.CONFIGS[name]
        o = ROOT / "out" / "smoke" / "E16_regress" / name
        if o.exists():
            shutil.rmtree(o)
        out_arg = str(o.relative_to(ROOT)) if c["env"] == "env_cosy" else str(o)
        r = subprocess.run([str(ENVS / c["env"] / "python.exe"), *c["cmd"](s, meta, 0, out_arg)], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:])
        for u in SS.utts(meta):
            a, b = o / f"{u}.wav", Path(ref) / f"{u}.wav"
            rows.append(dict(config=name, rc=r.returncode, utt=u, reference=str(b.parent),
                             identical=a.exists() and b.exists() and SS.data(a) == SS.data(b)))
            print(name, r.returncode, u[:30], rows[-1]["identical"], flush=True)
    with open(ROOT / "results/E16_regress.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["identical"]]
    print("STOP: mismatch" if bad else "all identical", len(rows) - len(bad), "/", len(rows))
