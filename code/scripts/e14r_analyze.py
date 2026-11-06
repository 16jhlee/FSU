"""E14R-2 analysis (v10.7 §2-4): table II rows on IndexTTS-2 and GLM-TTS (all-null branch, w 0.7), N = 2, en/zh 300,
seed 0, same machinery as e8p10_analyze.py (paired by utterance; SIM x100 ERes2Net main, WavLM / ECAPA alongside; WER %;
UTMOS; d = method - reference with bootstrap 95% CI and Wilcoxon p). Holm over the rule rows of each decoder x set
(vs Official, SIM ERes2Net and WER). recovery = (X - Official) / (teacher - Official) on ERes2Net SIM.
  IndexTTS-2: Official e81_idx_n2_def, FSU e81_idx_n2_R, FSU-Mid A_idx_midR, teacher e81_idx_n25_def (N = 25),
              Sel-CFG input_text B_idx_Rp_n2; guidance interval = FSU (alias), PCI not defined.
  GLM-TTS all-null: Official ga_11, FSU ga_01, FSU-Mid ga_midR, teacher ga_T (all-null N = 10); aliases: guidance
              interval = ga_01, TFR = ga_00 (w = 0), Sel-CFG input_text = g_base_n2 (official speaker-null branch).
Outputs results/E14_rules_idx.csv, E14_rules_glm.csv, E14_rules_summary.csv (table II layout incl. CosyVoice 2/3 rows
from E8p7_table7.csv / E8p10_table.csv), E14_rules_verdicts.csv.   python scripts/e14r_analyze.py   (env_eval)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e11_jobs as J11  # noqa: E402
import e13_jobs as J13  # noqa: E402
import e14r_jobs as JR  # noqa: E402
import e8p8_analyze as M8  # noqa: E402
import e8p_analyze as A  # noqa: E402

ROOT = A.ROOT
METRICS = [f"sim_{e}" for e in A.EMBS] + ["wer", "utmos"]


def W(name, k, s):
    for pre, fn in (("e14r:", JR.wav_dir), ("e13:", J13.wav_dir), ("e11:", J11.wav_dir)):
        if name.startswith(pre):
            return fn(name[len(pre):], k, s)
    return M8.W(name, k, s)


A.wdir = W

# (rule key, table II label, config or None, how: run / alias / ours / ref / undefined, calls)
DEC = {
    "IndexTTS-2": dict(official="e81_idx_n2_def", teacher="e81_idx_n25_def", rows=[
        ("official", "Official CFG", "e81_idx_n2_def", "ref", 4),
        ("fsu", "FSU = R (ours)", "e81_idx_n2_R", "ours", 3),
        ("fsumid", "FSU-Mid = mid+R (ours)", "A_idx_midR", "ours", 3),
        ("interval", "Guidance interval [0.1, 0.9]", "e81_idx_n2_R", "alias", 3),
        ("seltext", "Selective-CFG (input-text)", "B_idx_Rp_n2", "run", 4),
        ("defl", "Selective-CFG (default)", "e14r:ix_defl", "run", 4),
        ("s", "CFG-Zero* scale (s*)", "e14r:ix_s", "run", 4),
        ("zi", "CFG-Zero* zero-init", "e14r:ix_zi", "run", 4),
        ("tvs", "TV-CFG (released)", "e14r:ix_tvs", "run", 4),
        ("tvsf", "TV-CFG (corrected)", "e14r:ix_tvsf", "run", 4),
        ("tvu", "TV-CFG (step-up)", "e14r:ix_tvu", "run", 4),
        ("mambo4", "MAMBO-G (alpha = 4)", "e14r:ix_mambo4", "run", 4),
        ("tfr", "Terminal-fitted repair", "e14r:ix_tfr", "run", 4),
        ("jrr", "JRR-CFG", "e14r:ix_jrr", "run", 8),
        ("pci", "Prompt-consistency", None, "undefined", None),
        ("teacher", "teacher (N = 25)", "e81_idx_n25_def", "ref", 50)]),
    "GLM-TTS all-null": dict(official="e13:ga_11", teacher="e13:ga_T", rows=[
        ("official", "Official CFG", "e13:ga_11", "ref", 4),
        ("fsu", "FSU = R (ours)", "e13:ga_01", "ours", 3),
        ("fsumid", "FSU-Mid = mid+R (ours)", "e13:ga_midR", "ours", 3),
        ("interval", "Guidance interval [0.1, 0.9]", "e13:ga_01", "alias", 3),
        ("seltext", "Selective-CFG (input-text)", "e11:g_base_n2", "alias", 4),
        ("defl", "Selective-CFG (default)", "e14r:ga_defl", "run", 4),
        ("s", "CFG-Zero* scale (s*)", "e14r:ga_s", "run", 4),
        ("zi", "CFG-Zero* zero-init", "e14r:ga_zi", "run", 4),
        ("tvs", "TV-CFG (released)", "e14r:ga_tvs", "run", 4),
        ("tvsf", "TV-CFG (corrected)", "e14r:ga_tvsf", "run", 4),
        ("tvu", "TV-CFG (step-up)", "e14r:ga_tvu", "run", 4),
        ("mambo4", "MAMBO-G (alpha = 4)", "e14r:ga_mambo4", "run", 4),
        ("tfr", "Terminal-fitted repair", "e13:ga_00", "alias", 2),
        ("jrr", "JRR-CFG", "e14r:ga_jrr", "run", 8),
        ("pci", "Prompt-consistency", "e14r:ga_pci", "run", 4),
        ("teacher", "teacher (all-null N = 10)", "e13:ga_T", "ref", 20)]),
}


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    o = np.argsort(q)
    adj = np.minimum(np.maximum.accumulate((len(q) - np.arange(len(q))) * q[o]), 1.0)
    r = np.empty_like(q)
    r[o] = adj
    out[ok] = r
    return out


def table(model, spec):
    recs = []
    for key, label, cfg, how, calls in spec["rows"]:
        for s in ("en", "zh"):
            r = dict(model=model, method=label, key=key, how=how, nfe=2, set=s, config=cfg, calls=calls)
            D = A.load(cfg, s) if cfg else None
            if D is None or any(m not in D for m in METRICS):  # not run / evaluation still running
                recs.append(r)
                continue
            r["n"] = len(D)
            for m in METRICS:
                r[m] = D[m].mean() * (100 if m.startswith("sim") else 1)
            for tag, ref in (("vs_base", spec["official"]), ("vs_R", dict((x[0], x[2]) for x in spec["rows"])["fsu"]),
                             ("vs_teacher", spec["teacher"])):
                if ref == cfg:
                    continue
                Rf = A.load(ref, s)
                c = Rf.index.intersection(D.index)
                for m in METRICS:
                    est, lo, hi, p = A.stat((D.loc[c, m] - Rf.loc[c, m]) * (100 if m.startswith("sim") else 1))
                    r[f"{tag}_d_{m}"], r[f"{tag}_d_{m}_lo"], r[f"{tag}_d_{m}_hi"], r[f"{tag}_d_{m}_p"] = est, lo, hi, p
            T, B = A.load(spec["teacher"], s), A.load(spec["official"], s)
            gap = (T.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100
            r["teacher_minus_base"] = gap
            r["recovery"] = (r["sim_eres2net"] - B.sim_eres2net.mean() * 100) / gap
            recs.append(r)
    T = pd.DataFrame(recs)
    rules = T.how.isin(["run", "ours"])  # aliases repeat an existing test; Official / teacher are references
    for s in ("en", "zh"):
        sel = rules & (T.set == s)
        for m in ("sim_eres2net", "wer"):
            T.loc[sel, f"vs_base_d_{m}_p_holm"] = holm(T.loc[sel, f"vs_base_d_{m}_p"])
    return T


def summary(T_idx, T_glm):
    """table II layout: rule x decoder, 'SIM en / zh' and 'recovery en / zh' (CosyVoice rows from the E8'-7 / E8'-10 tables)"""
    lab = {"Kynk [0.1,0.9]": "interval", "R′ (Sel-CFG input_text)": "seltext", "Sel-CFG def_text": "defl", "CFG-Zero* s*": "s",
           "CFG-Zero* zi": "zi", "TV-CFG symmetric": "tvs", "TV-CFG symmetric (last-delta fix)": "tvsf", "TV-CFG stepup": "tvu",
           "MAMBO-G(4)": "mambo4", "TFR": "tfr", "JRR-CFG": "jrr", "PCI": "pci", "R (ours)": "fsu", "baseline": "official"}
    cv = []
    for f in ("results/E8p7_table7.csv", "results/E8p10_table.csv"):
        C = pd.read_csv(ROOT / f)
        C = C[(C.nfe == 2) & C.method.isin(lab)].assign(key=lambda d: d.method.map(lab))
        cv.append(C)
    C = pd.concat(cv).drop_duplicates(["model", "key", "set"])
    rows = []
    order = [r[0] for r in DEC["IndexTTS-2"]["rows"]]
    names = dict((r[0], r[1]) for r in DEC["IndexTTS-2"]["rows"])
    for key in order:
        r = dict(key=key, rule=names[key].replace(" (N = 25)", ""))
        for model, D in (("CosyVoice2", C[C.model == "CosyVoice2"]), ("CosyVoice3", C[C.model == "CosyVoice3"]),
                         ("IndexTTS-2", T_idx), ("GLM-TTS all-null", T_glm)):
            g = D[D.key == key].set_index("set")
            for s in ("en", "zh"):
                if s in g.index and pd.notna(g.loc[s].get("sim_eres2net", np.nan)):
                    r[f"{model} SIM {s}"] = g.loc[s, "sim_eres2net"]
                    r[f"{model} rec {s}"] = g.loc[s].get("recovery", np.nan)
                    r[f"{model} WER {s}"] = g.loc[s, "wer"]
                    r[f"{model} UTMOS {s}"] = g.loc[s, "utmos"]
                    if "how" in g:
                        r[f"{model} how"] = g.loc[s, "how"]
        rows.append(r)
    return pd.DataFrame(rows)


def verdicts(T):
    """Q1-Q6 on IndexTTS-2 and GLM-TTS."""
    V = []
    ext = ["interval", "seltext", "defl", "s", "zi", "tvs", "tvsf", "tvu", "mambo4", "tfr", "jrr", "pci"]
    for model, g in T.groupby("model"):
        for s, h in g.groupby("set"):
            rec = h.set_index("key").recovery
            fsu, mid = rec["fsu"], rec["fsumid"]
            ge = [k for k in ext if k in rec and np.isfinite(rec[k]) and rec[k] >= fsu - 1e-9]
            noshrink = {k: rec[k] for k in ("defl", "s", "jrr", "pci") if k in rec and np.isfinite(rec[k])}
            wer_tvs = h.set_index("key").wer.get("tvs", np.nan)
            V.append(dict(model=model, set=s, fsu_recovery=fsu, fsumid_recovery=mid,
                          Q1_rules_ge_fsu=" ".join(ge), Q1=set(ge) <= {"interval"},
                          Q2_max_external=max((rec[k] for k in ext if k in rec and np.isfinite(rec[k])), default=np.nan),
                          Q2=bool(mid > max((rec[k] for k in ext if k in rec and np.isfinite(rec[k])), default=-np.inf)),
                          Q3_max=max(noshrink.values(), default=np.nan),
                          Q3=bool(max(noshrink.values(), default=np.inf) < 0.30),
                          Q4_wer_tvs=wer_tvs, Q4=bool(wer_tvs > 30), zi_recovery=rec.get("zi", np.nan),
                          zi_wer=h.set_index("key").wer.get("zi", np.nan)))
    return pd.DataFrame(V)


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 200)
    Ti = table("IndexTTS-2", DEC["IndexTTS-2"])
    Tg = table("GLM-TTS all-null", DEC["GLM-TTS all-null"])
    Ti.to_csv(ROOT / "results/E14_rules_idx.csv", index=False)
    Tg.to_csv(ROOT / "results/E14_rules_glm.csv", index=False)
    S = summary(Ti, Tg)
    S.to_csv(ROOT / "results/E14_rules_summary.csv", index=False)
    V = verdicts(pd.concat([Ti, Tg]))
    V.to_csv(ROOT / "results/E14_rules_verdicts.csv", index=False)
    cols = ["model", "set", "method", "how", "calls", "sim_eres2net", "recovery", "vs_base_d_sim_eres2net_p_holm", "sim_wavlm",
            "sim_ecapa", "wer", "vs_base_d_wer", "vs_base_d_wer_p_holm", "utmos"]
    print(pd.concat([Ti, Tg])[cols].round(3).to_string(index=False))
    print(V.round(3).to_string(index=False))
