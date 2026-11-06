"""E14R-2 listening samples (v10.7 §2): per decoder and table II row (Official, FSU, FSU-Mid, teacher and every rule), the
first 3 utterances of en/zh 300 (seed 0) -> samples/E14/<decoder>/<rule>/<set>/<utt>.wav. Rows that are not defined
(IndexTTS-2 PCI) are skipped; aliases are copied from the run they alias.   python scripts/e14r_samples.py   (env_eval)
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e14r_analyze as E  # noqa: E402

ROOT = E.ROOT
N = 3
DIRS = {"IndexTTS-2": "IndexTTS-2", "GLM-TTS all-null": "GLM-TTS_allnull"}


def main():
    for model, spec in E.DEC.items():
        for key, label, cfg, how, _ in spec["rows"]:
            if cfg is None:
                continue
            for s, lst in (("en", "data/lists/en_all"), ("zh", "data/lists/zh_all")):
                src = Path(E.W(cfg, 0, s))
                utts = [l.split("|")[0] for l in (ROOT / (lst + ".meta.lst")).read_text(encoding="utf-8").splitlines()
                        if l.strip()][:N]
                dst = ROOT / "samples" / "E14" / DIRS[model] / key / s
                dst.mkdir(parents=True, exist_ok=True)
                n = sum(1 for u in utts if (src / f"{u}.wav").exists() and shutil.copy2(src / f"{u}.wav", dst / f"{u}.wav"))
                print(model, key, s, n, src)


if __name__ == "__main__":
    main()
