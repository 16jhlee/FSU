"""E15 analysis (v10.8 §2-4): the three top public rules (TV-CFG corrected, Selective-CFG input-text, MAMBO-G(4)) beside the
fixed comparators (Official CFG, FSU = R (ours), FSU-Mid = mid+R (ours), teacher) on the Table II sets and the Fig. 9
robustness conditions, N = 2, seed 0. Same machinery as e14r_analyze.py: paired by utterance, SIM x100 (ERes2Net main,
WavLM / ECAPA alongside), WER %, UTMOS; d = rule - reference with bootstrap 95% CI and Wilcoxon p; Holm over the rules of
each decoder x set (per comparison). recovery = (X - Official) / (teacher - Official) on ERes2Net SIM (Table II teachers).
GLM Sel-CFG input-text = the official speaker-null branch (alias of Official).
§3 check: mean SIM of the full-set run on the analysis-subset utterances vs the Table I subset run (|d| <= 0.01).
Outputs results/E15_rules_full.csv, E15_rules_robust.csv, E15_check300.csv, E15_verdicts.csv.
python scripts/e15_analyze.py   (env_eval)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e10_jobs as J10  # noqa: E402
import e11d_jobs as J11D  # noqa: E402
import e15_jobs as J  # noqa: E402
import e15_seed_subset as SS  # noqa: E402
import e8p8_analyze as M8  # noqa: E402
import e8p_analyze as A  # noqa: E402
import e8p_jobs as P  # noqa: E402
import e8v8r_jobs as JR  # noqa: E402
import e8v8s_jobs as JS  # noqa: E402

ROOT = A.ROOT
METRICS = [f"sim_{e}" for e in A.EMBS] + ["wer", "utmos"]
RES = {"e10": J10.wav_dir, "e11d": J11D.wav_dir, "e8p": P.wav_dir, "m8": M8.W, "s": JS.wav_dir, "r": JR.wav_dir, "e15": J.wav_dir}


def W(name, k, s):
    pre, n = name.split(":", 1)
    return RES[pre](n, k, s)


A.wdir = W
RULES = list(J.RULES)


def refs(tag):
    """comparators of an E15 group: official / fsu / fsumid / teacher (None = not run for that condition)"""
    m = tag.split("_")[-1]
    if tag.startswith(("full_", "hard_")):
        if m == "glm":
            return dict(official="e11d:d_base_n2", fsu="e11d:d_R_n2", fsumid="e11d:d_midR", teacher="e11d:d_teacher")
        return dict(official=f"e10:f_{m}_base_n2", fsu=f"e10:f_{m}_R_n2", fsumid=f"e10:f_{m}_midR", teacher=f"e10:f_{m}_teacher")
    if tag.startswith("mm_"):
        if m == "glm":
            return dict(official="e11d:l_base_n2", fsu="e11d:l_R_n2", fsumid="e11d:l_midR", teacher="e11d:l_teacher")
        return dict(official=f"e8p:e85_{m}_n2_def", fsu=f"e8p:e85_{m}_n2_R", fsumid=f"m8:A_{m}_midR", teacher=f"e8p:e85_{m}_n10_def")
    if tag.startswith("long_"):
        mode = tag.split("_")[1]
        return dict(official=f"s:s_{mode}_n2_def", fsu=f"s:s_{mode}_n2_R", fsumid=None, teacher=f"s:s_{mode}_n10_def")
    return dict(official="r:r_n2_def", fsu="r:r_n2_R", fsumid=None, teacher="r:r_n10_def")


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q = p[ok]
        o = np.argsort(q)
        adj = np.minimum(np.maximum.accumulate((len(q) - np.arange(len(q))) * q[o]), 1.0)
        r = np.empty_like(q)
        r[o] = adj
        out[ok] = r
    return out


def loaded(name, s):
    D = A.load(name, s) if name else None
    return D if D is not None and all(m in D for m in METRICS) else None


def rows_for(tag, block, dec, sets):
    R = refs(tag)
    out = []
    for s, meta, lang in sets:
        T, B = loaded(R["teacher"], s), loaded(R["official"], s)
        gap = (T.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100 if T is not None and B is not None else np.nan
        items = [(k, f"{k}", R[k], "ref") for k in ("official", "fsu", "fsumid", "teacher")]
        items += [(rule, J.RULES[rule][0], f"e15:{tag}_{rule}", "run") for rule in RULES if f"{tag}_{rule}" in J.CONFIGS]
        if dec == "GLM-TTS":
            items.append(("seltext", J.RULES["seltext"][0], R["official"], "alias"))
        for key, label, cfg, how in items:
            r = dict(block=block, decoder=dec, group=tag, set=s, lang=lang, key=key, method=label, how=how, config=cfg)
            D = loaded(cfg, s)
            if D is None:
                out.append(r)
                continue
            r["n"] = len(D)
            for m in METRICS:
                r[m] = D[m].mean() * (100 if m.startswith("sim") else 1)
            r["teacher_minus_base"] = gap
            r["recovery"] = (r["sim_eres2net"] - B.sim_eres2net.mean() * 100) / gap if np.isfinite(gap) else np.nan
            if how == "run":
                for tg in ("official", "fsu", "fsumid", "teacher"):
                    Rf = loaded(R[tg], s)
                    if Rf is None:
                        continue
                    c = Rf.index.intersection(D.index)
                    for m in METRICS:
                        est, lo, hi, p = A.stat((D.loc[c, m] - Rf.loc[c, m]) * (100 if m.startswith("sim") else 1))
                        r[f"vs_{tg}_d_{m}"], r[f"vs_{tg}_d_{m}_lo"], r[f"vs_{tg}_d_{m}_hi"], r[f"vs_{tg}_d_{m}_p"] = est, lo, hi, p
            out.append(r)
    return out


def table(blocks):
    rows = []
    for tag in dict.fromkeys(n.rsplit("_", 1)[0] for n in J.CONFIGS):
        first = next(n for n in J.CONFIGS if n.rsplit("_", 1)[0] == tag)
        if J.BLOCK[first] not in blocks:
            continue
        rows += rows_for(tag, J.BLOCK[first], J.LABEL[first][0], J.sets_for(first, 0))
    T = pd.DataFrame(rows)
    run = T.how == "run"
    for (g, s), idx in T[run].groupby(["group", "set"]).groups.items():
        for tg in ("official", "fsu", "fsumid"):
            for m in ("sim_eres2net", "wer"):
                col = f"vs_{tg}_d_{m}_p"
                if col in T:
                    T.loc[idx, col + "_holm"] = holm(T.loc[idx, col])
    return T


def check300():
    """§3: full-set run restricted to the analysis-subset utterances vs the Table I subset run (same rule)"""
    rows = []
    for (dec, rule), sub in SS.SUB.items():
        for s in ("en", "zh"):
            F = loaded(f"e15:full_{dec}_{rule}", s)
            f = Path(sub(s)) / "_metrics" / "metrics.csv"
            if F is None or not f.exists():
                continue
            S = pd.read_csv(f).set_index("utt_id")
            c = S.index.intersection(F.index)
            rows.append(dict(decoder=dec, rule=rule, set=s, n=len(c), sim_full=F.loc[c, "sim_eres2net"].mean() * 100,
                             sim_tableI=S.loc[c, "sim_eres2net"].mean() * 100,
                             d=(F.loc[c, "sim_eres2net"].mean() - S.loc[c, "sim_eres2net"].mean()) * 100))
    C = pd.DataFrame(rows)
    if len(C):
        C["ok"] = C.d.abs() <= 0.01
    return C


def table_I():
    """Table I subset recoveries (E14R summary: CosyVoice 2/3 from E8'-7 / E8'-10, IndexTTS-2 from E14R / E8'-8)"""
    S = pd.read_csv(ROOT / "results/E14_rules_summary.csv").set_index("key")
    out = {}
    for dec, col in (("CosyVoice2", "CosyVoice2"), ("CosyVoice3", "CosyVoice3"), ("IndexTTS-2", "IndexTTS-2")):
        for rule in RULES:
            for s in ("en", "zh"):
                v = S.loc[rule].get(f"{col} rec {s}", np.nan) if rule in S.index else np.nan
                out[(dec, rule, s)] = v
    return out


def verdicts(F, Rb, C):
    V = []
    full = F[(F.block == "E15-1")]
    for (dec, s), g in full.groupby(["decoder", "set"]):
        h = g.set_index("key")
        mid, fsu = h.sim_eres2net.get("fsumid"), h.sim_eres2net.get("fsu")
        for rule in RULES:
            if rule not in h.index or pd.isna(h.loc[rule].get("sim_eres2net")):
                continue
            x = h.loc[rule]
            V.append(dict(check="F1 rule < FSU-Mid (full)", decoder=dec, set=s, rule=rule, value=x.sim_eres2net - mid,
                          ok=bool(x.sim_eres2net < mid)))
            if dec in ("CosyVoice2", "CosyVoice3"):
                V.append(dict(check="F2 rule < FSU (CosyVoice)", decoder=dec, set=s, rule=rule, value=x.sim_eres2net - fsu,
                              ok=bool(x.sim_eres2net < fsu)))
    TI = table_I()
    for (dec, rule, s), v in TI.items():
        g = full[(full.decoder == dec) & (full.set == s) & (full.key == rule)]
        if len(g) and np.isfinite(v) and pd.notna(g.recovery.iloc[0]):
            V.append(dict(check="F3 |rec_full - rec_TableI| <= 0.10", decoder=dec, set=s, rule=rule,
                          value=g.recovery.iloc[0] - v, ok=bool(abs(g.recovery.iloc[0] - v) <= 0.10)))
    for (grp, s), g in Rb.groupby(["group", "set"]):
        h = g.set_index("key")
        rs = [r for r in RULES if r in h.index and h.loc[r, "how"] == "run" and pd.notna(h.loc[r].get("sim_eres2net"))]
        if not rs or pd.isna(h.sim_eres2net.get("fsu")):
            continue
        best = max(h.loc[r, "sim_eres2net"] for r in rs)
        V.append(dict(check="F4 FSU > every rule (robustness)", decoder=h.decoder.iloc[0], set=f"{grp}:{s}", rule="best",
                      value=h.sim_eres2net["fsu"] - best, ok=bool(h.sim_eres2net["fsu"] > best)))
    for T in (F, Rb):
        for _, x in T[(T.how == "run") & T.sim_eres2net.notna()].iterrows():
            p = x.get("vs_official_d_wer_p_holm", np.nan)
            V.append(dict(check="F5 WER not worse than Official (Holm)", decoder=x.decoder, set=f"{x.group}:{x.set}",
                          rule=x.key, value=x.get("vs_official_d_wer", np.nan),
                          ok=bool(not (x.get("vs_official_d_wer", 0) > 0 and p < 0.05))))
    V = pd.DataFrame(V)
    if len(C):
        V = pd.concat([V, pd.DataFrame([dict(check="§3 first 300 SIM = Table I (±0.01)", decoder=r.decoder, set=r.set,
                                              rule=r.rule, value=r.d, ok=bool(r.ok)) for r in C.itertuples()])])
    return V


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    F = table({"E15-1", "E15-2"})
    Rb = table({"E15-3"})
    C = check300()
    F.to_csv(ROOT / "results/E15_rules_full.csv", index=False)
    Rb.to_csv(ROOT / "results/E15_rules_robust.csv", index=False)
    C.to_csv(ROOT / "results/E15_check300.csv", index=False)
    V = verdicts(F, Rb, C)
    V.to_csv(ROOT / "results/E15_verdicts.csv", index=False)
    cols = ["decoder", "group", "set", "method", "how", "sim_eres2net", "recovery", "vs_fsu_d_sim_eres2net",
            "vs_fsumid_d_sim_eres2net", "wer", "vs_official_d_wer_p_holm", "utmos"]
    print(F[[c for c in cols if c in F]].round(3).to_string(index=False))
    print(Rb[[c for c in cols if c in Rb]].round(3).to_string(index=False))
    print(C.round(4).to_string(index=False))
    print(V.groupby("check").ok.agg(["sum", "count"]).to_string())
