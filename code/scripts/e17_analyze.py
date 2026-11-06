"""E17 analysis (v10.10 §2-4): the robustness table (E15-3 conditions) with the empty cells filled.
Same rows and columns as results/E15_rules_robust.csv; changes:
  fsumid  = E17 run (how = run, paired against official / fsu / teacher) for CosyVoice2/3 mm_de, CosyVoice2 long x4, prompt x5
            (the existing ref rows stay for CosyVoice2/3 ko/ja and GLM de speaker-null)
  group mm_glm_an (new) = GLM-TTS all-null de: official = all-null CFG (E11-L l_allnull_n2), fsu = all-null + FSU (l_allnullR_n2),
            fsumid / tvsf / mambo4 = E17 runs, teacher = speaker-null NFE 10 (l_teacher, as in Table II),
            seltext = alias of the speaker-null official branch (l_base_n2)
Holm over the generated rows (rules + E17 FSU-Mid) of each condition, per comparison and metric (so the rule rows' Holm p can
differ slightly from E15). recovery = (X - official) / (teacher - official) on ERes2Net SIM.
Outputs results/E17_robust_fill.csv, results/E17_verdicts.csv.   python scripts/e17_analyze.py (env_eval)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e15_analyze as E15  # noqa: E402
import e15_jobs as J15  # noqa: E402
import e16_analyze as E16  # noqa: E402  (10k bootstrap stat; registers e16 / e11d)
import e17_jobs as J  # noqa: E402

ROOT = E15.ROOT
E15.RES["e17"] = J.wav_dir
METRICS = E15.METRICS
NEW_MID = {("mm_cv2", "mm_de"): "e17:mm_cv2_fsumid", ("mm_cv3", "mm_de"): "e17:mm_cv3_fsumid",
           **{("long_off_cv2", s): "e17:long_off_cv2_fsumid" for s in ("long_en", "long_zh")},
           **{("long_str_cv2", s): "e17:long_str_cv2_fsumid" for s in ("long_en", "long_zh")},
           **{("rb_cv2", s): "e17:rb_cv2_fsumid" for s in J15.JR.NAMES}}


def items_for(tag, s):
    if tag == "mm_glm_an":
        return [("official", "all-null CFG", "e11d:l_allnull_n2", "ref"), ("fsu", "all-null + FSU (ours)", "e11d:l_allnullR_n2", "ref"),
                ("fsumid", "all-null + FSU-Mid (ours)", "e17:mmglm_an_fsumid", "run"),
                ("teacher", "teacher (speaker-null)", "e11d:l_teacher", "ref"),
                ("tvsf", J15.RULES["tvsf"][0], "e17:mmglm_an_tvsf", "run"),
                ("seltext", J15.RULES["seltext"][0], "e11d:l_base_n2", "alias"),
                ("mambo4", J15.RULES["mambo4"][0], "e17:mmglm_an_mambo4", "run")]
    R = E15.refs(tag)
    out = []
    for k in ("official", "fsu", "fsumid", "teacher"):
        if k == "fsumid" and (tag, s) in NEW_MID:
            out.append((k, k, NEW_MID[(tag, s)], "run"))
        else:
            out.append((k, k, R[k], "ref"))
    out += [(rule, J15.RULES[rule][0], f"e15:{tag}_{rule}", "run") for rule in J15.RULES if f"{tag}_{rule}" in J15.CONFIGS]
    if tag == "mm_glm":
        out.append(("seltext", J15.RULES["seltext"][0], R["official"], "alias"))
    return out


def groups():
    for tag in dict.fromkeys(n.rsplit("_", 1)[0] for n in J15.CONFIGS):
        first = next(n for n in J15.CONFIGS if n.rsplit("_", 1)[0] == tag)
        if J15.BLOCK[first] == "E15-3":
            yield tag, J15.LABEL[first][0], J15.sets_for(first, 0)
    yield "mm_glm_an", "GLM-TTS all-null", [x for x in J.MM_DE]


def table():
    rows = []
    for tag, dec, sets in groups():
        for s, meta, lang in sets:
            its = items_for(tag, s)
            cfg = {k: c for k, _, c, _ in its}
            B, T = E15.loaded(cfg["official"], s), E15.loaded(cfg["teacher"], s)
            gap = (T.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100 if T is not None and B is not None else np.nan
            for key, label, c, how in its:
                r = dict(block="E17" if c and c.startswith("e17:") else "E15-3", decoder=dec, group=tag, set=s, lang=lang,
                         key=key, method=label, how=how, config=c)
                D = E15.loaded(c, s) if c else None
                if D is None:
                    rows.append(r)
                    continue
                r["n"] = len(D)
                for m in METRICS:
                    r[m] = D[m].mean() * (100 if m.startswith("sim") else 1)
                r["teacher_minus_base"] = gap
                r["recovery"] = (r["sim_eres2net"] - B.sim_eres2net.mean() * 100) / gap if np.isfinite(gap) else np.nan
                if how == "run":
                    for tg in ("official", "fsu", "fsumid", "teacher"):
                        Rf = E15.loaded(cfg[tg], s) if cfg.get(tg) and tg != key else None
                        if Rf is None:
                            continue
                        ix = Rf.index.intersection(D.index)
                        for m in METRICS:
                            est, lo, hi, p = E16.stat((D.loc[ix, m] - Rf.loc[ix, m]) * (100 if m.startswith("sim") else 1))
                            r[f"vs_{tg}_d_{m}"], r[f"vs_{tg}_d_{m}_lo"], r[f"vs_{tg}_d_{m}_hi"], r[f"vs_{tg}_d_{m}_p"] = est, lo, hi, p
                rows.append(r)
    T = pd.DataFrame(rows)
    for (g, s), idx in T[T.how == "run"].groupby(["group", "set"]).groups.items():
        for tg in ("official", "fsu", "fsumid"):
            for m in ("sim_eres2net", "wer"):
                col = f"vs_{tg}_d_{m}_p"
                if col in T:
                    T.loc[idx, col + "_holm"] = E15.holm(T.loc[idx, col])
    return T


def verdicts(T):
    V = []
    for (g, s), x in T.groupby(["group", "set"]):
        h = x.set_index("key").sim_eres2net
        if (g, s) in NEW_MID and pd.notna(h.get("fsumid")) and pd.notna(h.get("fsu")):
            V.append(dict(check="R1 FSU-Mid > FSU (CosyVoice, new cells)", group=g, set=s, value=h["fsumid"] - h["fsu"],
                          ok=bool(h["fsumid"] > h["fsu"])))
        if g in ("mm_cv2", "mm_cv3") and s == "mm_de" and pd.notna(h.get("fsumid")):
            V.append(dict(check="R2 FSU-Mid >= Sel-CFG - 0.5 (de)", group=g, set=s, value=h["fsumid"] - h["seltext"],
                          ok=bool(h["fsumid"] >= h["seltext"] - 0.5)))
        if g == "mm_glm_an" and pd.notna(h.get("fsumid")):
            V.append(dict(check="R3a GLM all-null de FSU-Mid >= FSU", group=g, set=s, value=h["fsumid"] - h["fsu"],
                          ok=bool(h["fsumid"] >= h["fsu"])))
            for rule in ("tvsf", "mambo4"):
                V.append(dict(check=f"R3b GLM all-null de {rule} < FSU", group=g, set=s, value=h[rule] - h["fsu"],
                              ok=bool(h[rule] < h["fsu"])))
    return pd.DataFrame(V)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    T = table()
    T.to_csv(ROOT / "results/E17_robust_fill.csv", index=False)
    V = verdicts(T)
    V.to_csv(ROOT / "results/E17_verdicts.csv", index=False)
    cols = ["decoder", "group", "set", "key", "how", "sim_eres2net", "recovery", "vs_fsu_d_sim_eres2net", "vs_fsu_d_sim_eres2net_p_holm",
            "wer", "vs_official_d_wer", "vs_official_d_wer_p_holm", "utmos"]
    print(T[[c for c in cols if c in T]].round(3).to_string(index=False))
    if len(V):
        print(V.round(3).to_string(index=False))
        print(V.groupby("check").ok.agg(["sum", "count"]).to_string())
