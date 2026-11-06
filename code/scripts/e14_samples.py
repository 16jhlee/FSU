"""Listening samples for E14 (v10.6) and E13-10: the first 5 utterances (meta order) of seed 0 per config and set, copied to
samples/<block>/<config>/<set>/<utt>.wav (comparators with the same utterances alongside).
python scripts/e14_samples.py   (env_eval)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e11_jobs as J11  # noqa: E402
import e11d_jobs as J11D  # noqa: E402
import e13_jobs as J13  # noqa: E402
import e14_jobs as J  # noqa: E402

ROOT = J.ROOT
N = 5
LISTS = {"en": "data/lists/en_all", "zh": "data/lists/zh_all", "en_full": "data/lists_full/en_all",
         "zh_full": "data/lists_full/zh_all", "hard": "data/lists_full/hard_all"}
SETS = {
    "E14": [  # (label, wav_dir function, config, set)
        *[(c, J.wav_dir, c, s) for c in ("ga_10", "g_n10_w0", "g_w0_n1") for s in ("en", "zh")],
        *[(c, J.wav_dir, c, "hard") for c in ("allnull_n2", "allnullR_n2", "allnull_midR")],
        *[("allnull_midR", J.wav_dir, "allnull_midR", s) for s in ("en_full", "zh_full")],
        *[(c, J.wav_dir, c, s) for c in ("c3_n10_w0", "ix_n25_w0") for s in ("en", "zh")],
        *[("teacher_g_base_n10", J11D.wav_dir, "d_teacher", s, ls) for s, ls in (("en", "en_full"), ("zh", "zh_full"), ("hard", "hard"))],
        *[(c, J13.wav_dir, c, s) for c in ("ga_11", "ga_00", "ga_T") for s in ("en", "zh")],
    ],
    "E13": [(c, J13.wav_dir, c, s) for c in ("ga_10", "ga_midR", "ga_11", "ga_01", "ga_T") for s in ("en", "zh")],
}


def main():
    for block, items in SETS.items():
        for label, wd, cfg, s, *ls in items:
            ls = ls[0] if ls else s
            try:
                src = Path(wd(cfg, 0, s))
            except KeyError as ex:
                print("skip", block, label, s, ex)
                continue
            utts = [l.split("|")[0] for l in (ROOT / (LISTS[ls] + ".meta.lst")).read_text(encoding="utf-8").splitlines()
                    if l.strip()][:N]
            dst = ROOT / "samples" / block / label / ls
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for u in utts:
                if (src / f"{u}.wav").exists():
                    shutil.copy2(src / f"{u}.wav", dst / f"{u}.wav")
                    n += 1
            print(block, label, s, n, "from", src)


if __name__ == "__main__":
    main()
