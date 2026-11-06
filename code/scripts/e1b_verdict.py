"""E1b 1.3: does the E1 conclusion hold with the main embedder set (WavLM, ECAPA, ERes2Net-en)?
Prints paired deltas with CI, Wilcoxon flag, and '(<eps)' when |mean delta| is below the noise floor."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
noise = pd.read_csv(ROOT / "results" / "E1b_noise.csv")
EPS = (noise.groupby("emb").dsim.quantile(0.95) * 100).to_dict()
MAIN = ["wavlm", "ecapa", "eres2net"]
KEEP = ["cv2_nfe5", "cv2_nfe3", "cv2_nfe2", "cv2_nfe1", "zipvoice_nfe8", "zipvoice_nfe4", "zipvoice_distill_nfe8",
        "zipvoice_distill_nfe4", "zipvoice_distill_nfe2", "dmo_student_nfe4", "dmo_student_nfe3",
        "dmo_student_nfe2", "dmo_student_nfe1"]


def main():
    S = pd.read_csv(ROOT / "results" / "E1_summary.csv")
    print("eps x100:", {k: round(v, 2) for k, v in EPS.items()})
    for s in ["en", "zh", "hard"]:
        print(f"\n== {s}")
        t = S[(S.set == s) & S.job.isin(KEEP)].set_index("job").reindex(KEEP).dropna(how="all")
        for job, r in t.iterrows():
            parts = []
            for e in MAIN:
                d, lo, hi, p = (r[f"{k}_sim_{e}"] * 100 for k in ["d", "lo", "hi"]) + (r[f"p_sim_{e}"],) \
                    if False else (r[f"d_sim_{e}"] * 100, r[f"lo_sim_{e}"] * 100, r[f"hi_sim_{e}"] * 100, r[f"p_sim_{e}"])
                flag = "*" if p < 0.05 else " "
                res = "(<eps)" if abs(d) < EPS[e] else ""
                parts.append(f"{e}:{d:+.2f}[{lo:+.2f},{hi:+.2f}]{flag}{res}")
            simr = ",".join(f"{r[f'd_simr_{e}'] * 100:+.2f}" for e in MAIN)
            wflag = "*" if r.p_wer < 0.05 else " "
            print(f"{job:24s} dWER {r.d_wer:+.2f}{wflag} | " + "  ".join(parts) + f" | dSIM-r {simr}")


if __name__ == "__main__":
    main()
