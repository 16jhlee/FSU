"""E11-D / E11-L analysis (GLM-TTS, v10.3).   python scripts/e11d_analyze.py   (env_eval)
-> results/E11_full_table1.csv (+ Holm), results/E11_full_subset_check.csv, results/E11_lang.csv
Recovery = (X - baseline NFE 2) / (teacher - baseline NFE 2) only when teacher - baseline >= eps 0.42 (else Δ only);
all-null + R is measured against the all-null baseline (its own recovery). Objective-quality columns join from
_metrics/objq.csv / objq_dnsmos.csv when present.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e11_jobs as J11  # noqa: E402
import e11d_jobs as J  # noqa: E402

ROOT = J.ROOT
OBJ = ["utmos", "utmosv2", "dnsmos_ovrl", "nisqa_mos"]
EPS = 0.42


def L(name, s):
    return EA.load_metrics(J.wav_dir(name, 0, s))


def block(prefix, sets, conds, aux):
    rows = []
    for s in sets:
        T = L(f"{prefix}_teacher", s)
        base2, base3 = L(f"{prefix}_base_n2", s), L(f"{prefix}_base_n3", s)
        an = L(f"{prefix}_allnull_n2", s) if aux else None
        for key in conds:
            X = L(f"{prefix}_{key}", s)
            if X is None:
                continue
            ref = {"R_n2": base2, "midR": base2, "R_n3": base3, "allnullR_n2": an}.get(key)
            r = dict(set=s, cond=key, method=J.LABEL[f"{prefix}_{key}"][0], calls=J.CALLS[key], n=len(X))
            r.update(EA.compare(X, ref, T if key != "teacher" else None))
            if ref is not None and T is not None:
                gap = T.sim_eres2net.mean() - ref.sim_eres2net.mean()
                r["teacher_minus_ref"] = gap
                r["recovery"] = (X.sim_eres2net.mean() - ref.sim_eres2net.mean()) / gap if gap >= EPS else np.nan
            if T is not None and key != "teacher":
                r["loss_vs_teacher"] = T.sim_eres2net.mean() - X.sim_eres2net.mean()
            rows.append(r)
    D = pd.DataFrame(rows)
    for c in [c for c in D.columns if c.endswith("_p")]:
        D[c + "_holm"] = EA.holm(D[c].values)
    return D


def main():
    pd.set_option("display.width", 260)
    conds = ["base_n2", "R_n2", "midR", "base_n3", "R_n3", "teacher", "allnull_n2", "allnullR_n2"]
    F = block("d", ["en", "zh", "lspc", "hard"], conds, aux=True)
    F.to_csv(ROOT / "results/E11_full_table1.csv", index=False)
    # determinism: subset utterances vs E11 block A/B runs
    pairs = {"base_n2": "g_base_n2", "R_n2": "g_R_n2", "midR": "g_midR", "base_n3": "g_base_n3", "R_n3": "g_R_n3",
             "teacher": "g_base_n10", "allnull_n2": "g_allnull_n2", "allnullR_n2": "g_allnullR_n2"}
    chk = []
    for key, g in pairs.items():
        for s in ("en", "zh"):
            X, S = L(f"d_{key}", s), EA.load_metrics(J11.wav_dir(g, 0, s))
            if X is None or S is None:
                continue
            c = S.index.intersection(X.index)
            chk.append(dict(cond=key, set=s, n_common=len(c), max_abs_d_sim=float(np.max(np.abs(S.loc[c, "sim_eres2net"] - X.loc[c, "sim_eres2net"]))),
                            sim_subset=S.loc[c, "sim_eres2net"].mean(), sim_full_on_subset=X.loc[c, "sim_eres2net"].mean(), sim_full=X.sim_eres2net.mean()))
    C = pd.DataFrame(chk)
    C.to_csv(ROOT / "results/E11_full_subset_check.csv", index=False)
    # ko / ja unsupported (LLM does not terminate), see e11d_jobs.MM
    Lg = block("l", [s for s, _, _ in J.MM], ["base_n2", "R_n2", "midR", "teacher", "allnull_n2", "allnullR_n2"], aux=True)
    Lg.to_csv(ROOT / "results/E11_lang.csv", index=False)
    cols = ["set", "cond", "n", "sim_eres2net", "vs_base_d_sim_eres2net", "vs_base_d_sim_eres2net_p_holm", "recovery", "loss_vs_teacher",
            "wer", "vs_base_d_wer", "vs_base_d_wer_p", "wer_wh", "utmos"] + [c for c in ("utmosv2", "dnsmos_ovrl", "nisqa_mos") if c in F]
    print(F[[c for c in cols if c in F]].round(3).to_string())
    print(C.round(4).to_string())
    lc = ["set", "cond", "n", "sim_eres2net", "vs_base_d_sim_eres2net", "vs_base_d_sim_eres2net_p", "recovery", "loss_vs_teacher", "wer",
          "vs_base_d_wer_p", "utmos"]
    if "wer_sv" in Lg:
        lc.append("wer_sv")
    print(Lg[[c for c in lc if c in Lg]].round(3).to_string())


if __name__ == "__main__":
    main()
