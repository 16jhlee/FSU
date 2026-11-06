"""E15 listening samples (v10.8 §5): per decoder, rule and condition the first 3 utterances (meta order) of the rule run
-> samples/E15/<decoder>/<rule>/<group>_<set>/, and once per condition the same utterances of Official / FSU / FSU-Mid /
teacher (where run) -> samples/E15/<decoder>/_comparators/<group>_<set>/<who>/ (not repeated per rule).
python scripts/e15_samples.py   (env_eval)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e15_analyze as E  # noqa: E402
import e15_jobs as J  # noqa: E402

ROOT = J.ROOT
N = 3


def main():
    n_all = 0
    for name in J.CONFIGS:
        tag, rule = name.rsplit("_", 1)
        dec = J.LABEL[name][0].replace(" ", "_")
        R = E.refs(tag)
        for s, meta, _ in J.sets_for(name, 0):
            utts = [l.split("|")[0] for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()][:N]
            for who, cfg in [(rule, f"e15:{name}")] + [(k, v) for k, v in R.items() if v]:
                src = Path(E.W(cfg, 0, s))
                dst = ROOT / "samples" / "E15" / dec / (rule if who == rule else "_comparators") / f"{tag}_{s}" / (
                    "" if who == rule else who)
                dst.mkdir(parents=True, exist_ok=True)
                for u in utts:
                    if (src / f"{u}.wav").exists():
                        shutil.copy2(src / f"{u}.wav", dst / f"{u}.wav")
                        n_all += 1
    print(n_all, "files")


if __name__ == "__main__":
    main()
