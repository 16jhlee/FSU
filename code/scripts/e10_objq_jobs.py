"""E10-1′ re-scoring job list (existing wavs, no inference): the folders behind Table 1, Table 7′, Table 9, the v8 tables
(§1.2 seeds 1-2 of the Table-1 rows, §2.2 streaming, §2.3 prompt robustness) and the 8-seed teacher runs used for the
predictor resolution ε; plus the E10 rows (full-set Table 1, E10-4 fill) once they exist.
python scripts/e10_objq_jobs.py [out=logs/objq_jobs.tsv]   (env_eval / tools) -> lines "meta<TAB>wav_dir<TAB>tag"
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e8p7_jobs as J7  # noqa: E402
import e8p9_jobs as J9  # noqa: E402
import e8p10_jobs as J10  # noqa: E402
import e8v8_jobs as JV  # noqa: E402
import e8v8r_jobs as JR  # noqa: E402
import e8v8s_jobs as JS  # noqa: E402
import e10_jobs as JE  # noqa: E402
from e9_tables import T1  # noqa: E402

ROOT = J9.ROOT
LISTS = {"en": "data/lists/en_all", "zh": "data/lists/zh_all", "hard": "data/lists/hard_all",
         "mm_ko": "data/lists/mm_ko", "mm_ja": "data/lists/mm_ja", "mm_de": "data/lists/mm_de"}
TEACHERS = ["e82_cv2_all_n10_def", "e81_cv3_n10_def", "e81_idx_n25_def"]
V8_ROWS = ["e82_cv2_all_n2_def", "e82_cv2_all_n2_R", "A_cv2_midR", "e82_cv2_all_n10_def", "e82_cv2_all_n3_def",
           "e82_cv2_all_n3_R", "s_cv2_midR1", "e81_cv3_n2_def", "e81_cv3_n2_R", "A_cv3_midR", "e81_cv3_n10_def",
           "s_cv3_n3_def", "s_cv3_n3_R", "s_cv3_midR1", "e81_idx_n2_def", "e81_idx_n2_R", "A_idx_midR", "e81_idx_n25_def",
           "s_idx_n3_def", "s_idx_n3_R"]


def jobs(include_e10=True):
    out = []

    def add(meta, d, tag):
        d = Path(d)
        if (d / "_done").exists() or any(d.glob("*.wav")):
            out.append((meta + ".meta.lst", str(d), tag))

    for model, d in T1.items():  # Table 1 (subset)
        cfgs = [d["teacher"][0]] + [c for items in d["rows"].values() for _, c, _ in items]
        for c in cfgs:
            for s in ("en", "zh"):
                add(LISTS[s], J9.wav_dir(c, 0, s), "T1")
    t7 = pd.read_csv(ROOT / "results/E8p7_table7.csv")  # Table 7′ = E8′-7 + E8′-10
    for c, s in t7[["config", "set"]].drop_duplicates().itertuples(index=False):
        add(LISTS[s], J7.wav_dir(c, 0, s), "T7")
    t10 = pd.read_csv(ROOT / "results/E8p10_table.csv")
    for c, s in t10[["config", "set"]].drop_duplicates().itertuples(index=False):
        add(LISTS[s], J10.wav_dir(c, 0, s), "T7")
    ops = pd.read_csv(ROOT / "results/E8p8_ops.csv")  # Table 9 (+ Table 1 ops) incl. hard / ko / ja
    for c, s in ops[["config", "set"]].drop_duplicates().itertuples(index=False):
        if s in LISTS:
            add(LISTS[s], J9.wav_dir(c, 0, s), "T9")
    pp = pd.read_csv(ROOT / "results/E8p_pairs.csv")
    for _, r in pp[(pp.label == "NFE 2") & pp.set.isin(["hard", "mm_ko", "mm_ja", "mm_de"])].iterrows():
        for c in (r["default"], r["R"]):
            add(LISTS[r.set], J9.wav_dir(c, 0, r.set), "T9")
    for s in ("mm_ko", "mm_ja", "mm_de"):
        for c in ("e85_cv2_n10_def", "e85_cv3_n10_def"):
            add(LISTS[s], J9.wav_dir(c, 0, s), "T9")
    for c in TEACHERS:  # predictor resolution ε: 8 teacher seeds (seed 0 is in T1)
        for k in range(1, 8):
            for s in ("en", "zh"):
                add(LISTS[s], J9.wav_dir(c, k, s), "eps")
    for c in V8_ROWS:  # v8 §1.2 seeds 1, 2
        for k in (1, 2):
            for s in ("en", "zh"):
                add(LISTS[s], JV.wav_dir(c, k, s), "v8seed")
    for name, c in JS.CONFIGS.items():  # v8 §2.2 streaming
        if not c.get("prep"):
            for s, m, _ in JS.sets_for(name, 0):
                add(m, JS.wav_dir(name, 0, s), "v8stream")
    for name, c in JR.CONFIGS.items():  # v8 §2.3 prompt robustness (seed 0)
        if not c.get("prep"):
            for s, m, _ in JR.sets_for(name, 0):
                add(m, JR.wav_dir(name, 0, s), "v8robust")
    if include_e10:
        for name in JE.CONFIGS:
            if JE.BLOCK.get(name) == "full":
                for s, m, _ in JE.sets_for(name, 0):
                    if s != "hard":
                        add(m, JE.wav_dir(name, 0, s), "E10full")
            if name.startswith("fill_"):
                for k in JE.CONFIGS[name]["K"]:
                    if k <= 2:
                        for s, m, _ in JE.sets_for(name, k):
                            add(m, JE.wav_dir(name, k, s), "E10fill")
    seen, uniq = set(), []
    for m, d, t in out:
        key = str(Path(d).resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append((m, d, t))
    return uniq


if __name__ == "__main__":
    J = jobs(include_e10=False)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "objq_jobs.tsv"
    out.write_text("".join(f"{m}\t{d}\t{t}\n" for m, d, t in J), encoding="utf-8")
    n = {}
    for m, d, t in J:
        n[t] = n.get(t, 0) + J9.P8.P.n_lines(m[:-9])
    print(len(J), "folders", n, sum(n.values()), "wavs ->", out)
