"""E15 (v10.8 §3) reuse of the analysis-subset runs of the same rules (no regeneration of earlier experiments).
  check : the E15 full-set command on the first 3 subset utterances (en) -> audio data chunk must equal the subset run
          (stop otherwise) -> results/E15_regress.csv
  seed  : copy the subset wavs (en/zh 300, hard 100) into the E15 full-set folders (only utterances of the full list);
          the runners then generate the remaining utterances only -> results/E15_seeded.csv
python scripts/e15_seed_subset.py check|seed   (env_eval; check runs the decoders through their envs)
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e14r_jobs as JR  # noqa: E402
import e15_jobs as J  # noqa: E402
import e8p10_jobs as J10b  # noqa: E402
import e8p7_jobs as J7  # noqa: E402
import e8p8_analyze as M8  # noqa: E402
import e8p_jobs as P  # noqa: E402

ROOT = J.ROOT
ENVS = Path(sys.executable).resolve().parents[1]
# (E15 group tag, rule) -> path function (k, s) of the subset run
SUB = {("cv2", "seltext"): lambda s: P.wav_dir("e82_cv2_speaker_n2_def", 0, s),
       ("cv2", "tvsf"): lambda s: J10b.wav_dir("cv2_tvsf_n2", 0, s),
       ("cv2", "mambo4"): lambda s: J7.wav_dir("cv2_mambo4_n2", 0, s),
       ("cv3", "seltext"): lambda s: M8.W("B_cv3_Rp_n2", 0, s),
       ("cv3", "tvsf"): lambda s: J10b.wav_dir("cv3_tvsf_n2", 0, s),
       ("cv3", "mambo4"): lambda s: J7.wav_dir("cv3_mambo4_n2", 0, s),
       ("idx", "seltext"): lambda s: M8.W("B_idx_Rp_n2", 0, s),
       ("idx", "tvsf"): lambda s: JR.wav_dir("ix_tvsf", 0, s),
       ("idx", "mambo4"): lambda s: JR.wav_dir("ix_mambo4", 0, s)}


def data(p):
    x = p.read_bytes()
    return x[x.index(b"data"):]


def utts(meta):
    return [l.split("|")[0] for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()]


def check():
    rows = []
    for (dec, rule), sub in SUB.items():
        name = f"full_{dec}_{rule}"
        c = J.CONFIGS[name]
        o = ROOT / "out" / "smoke" / "E15_regress" / name
        if o.exists():
            shutil.rmtree(o)
        out_arg = str(o.relative_to(ROOT)) if c["env"] == "env_cosy" else str(o)
        r = subprocess.run([str(ENVS / c["env"] / "python.exe"), *c["cmd"]("en", "data/lists/en_3", 0, out_arg)], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        for u in utts("data/lists/en_3"):
            a, b = o / f"{u}.wav", Path(sub("en")) / f"{u}.wav"
            rows.append(dict(config=name, rc=r.returncode, utt=u, subset=str(b.parent),
                             identical=a.exists() and b.exists() and data(a) == data(b)))
            print(name, r.returncode, u[:30], rows[-1]["identical"], flush=True)
    with open(ROOT / "results/E15_regress.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["identical"]]
    print("STOP: mismatch" if bad else "all identical", len(rows) - len(bad), "/", len(rows))


def seed():
    """CosyVoice3 / IndexTTS-2 only: the CosyVoice2 check failed on the first utterance of the process (max |d| 1.2e-4,
    the same first-call difference exists between the E10 full baseline and the subset baseline) -> CosyVoice2 rule
    folders are generated in full instead of seeded (2026-11-03)."""
    rows = []
    for (dec, rule), sub in SUB.items():
        if dec == "cv2":
            continue
        for tag, sets in ((f"full_{dec}_{rule}", ("en", "zh")), (f"hard_{dec}_{rule}", ("hard",))):
            for s, meta, _ in J.sets_for(tag, 0):
                if s not in sets:
                    continue
                src, dst = Path(sub(s)), J.wav_dir(tag, 0, s)
                keep = set(utts(meta))
                dst.mkdir(parents=True, exist_ok=True)
                n = 0
                for f in src.glob("*.wav"):
                    if f.stem in keep and not (dst / f.name).exists():
                        shutil.copy2(f, dst / f.name)
                        n += 1
                rows.append(dict(config=tag, set=s, source=str(src), copied=n, full_list=len(keep)))
                print(tag, s, n, "/", len(keep), "from", src, flush=True)
    with open(ROOT / "results/E15_seeded.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    {"check": check, "seed": seed}[sys.argv[1]]()
