"""E9 (v7 §3.2) paper tables from the result CSVs / metric files -> reports/E9_tables.md (+ results/E9_table*.csv).
Tables 2 (prior work) and 6 (protocol) are written by hand in the same report (sources: novelty survey, E1b, E6).
env_eval.  python scripts/e9_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e8p9_jobs as J9  # noqa: E402
import e8p_analyze as A  # noqa: E402

ROOT = A.ROOT
A.wdir = J9.wav_dir
OUT = []


def md(df, cols=None):
    df = df[cols] if cols else df
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def mean(cfg, s, m):
    D = A.load(cfg, s)
    if D is None or m not in D:
        return np.nan
    return D[m].mean() * (100 if m.startswith("sim") else 1)


def f2(x):
    return "" if np.isnan(x) else f"{x:.2f}"


# ---------------- Table 1 ----------------
T1 = {
    "CosyVoice2": dict(teacher=("e82_cv2_all_n10_def", 20), rows={
        2: [("baseline", "e82_cv2_all_n2_def", 4), ("**R (ours)**", "e82_cv2_all_n2_R", 3),
            ("mid+R (ours)", "A_cv2_midR", 3), ("R′ (Sel-CFG input_text)", "e82_cv2_speaker_n2_def", 4)],
        3: [("baseline", "e82_cv2_all_n3_def", 6), ("**R (ours)**", "e82_cv2_all_n3_R", 5),
            ("mid+R +1 Euler (ours)", "b_cv2_midR1", 5), ("R′ (Sel-CFG input_text)", "e82_cv2_speaker_n3_def", 6)]}),
    "CosyVoice3": dict(teacher=("e81_cv3_n10_def", 20), rows={
        2: [("baseline", "e81_cv3_n2_def", 4), ("**R (ours)**", "e81_cv3_n2_R", 3), ("mid+R (ours)", "A_cv3_midR", 3),
            ("R′ (Sel-CFG input_text)", "B_cv3_Rp_n2", 4)],
        3: [("baseline", "e81_cv3_n3_def", 6), ("**R (ours)**", "e81_cv3_n3_R", 5), ("mid+R +1 Euler (ours)", "b_cv3_midR1", 5),
            ("R′ (Sel-CFG input_text)", "B_cv3_Rp_n3", 6)]}),
    "IndexTTS-2": dict(teacher=("e81_idx_n25_def", 50), rows={
        2: [("baseline", "e81_idx_n2_def", 4), ("**R (ours)**", "e81_idx_n2_R", 3), ("mid+R (ours)", "A_idx_midR", 3),
            ("R′ (Sel-CFG input_text)", "B_idx_Rp_n2", 4)],
        3: [("baseline", "e81_idx_n3_def", 6), ("**R (ours)**", "e81_idx_n3_R", 5), ("R′ (Sel-CFG input_text)", "B_idx_Rp_n3", 6)]}),
    "F5-TTS (text-cond.)": dict(teacher=("e81_f5_n32_def", 64), rows={
        8: [("baseline", "e81_f5_n8_def", 16), ("R (ours)", "e81_f5_n8_R", 13)],
        16: [("baseline", "e81_f5_n16_def", 32), ("R (ours)", "e81_f5_n16_R", 27)]}),
    "ZipVoice (text-cond.)": dict(teacher=("e7:zip_default_n16", 32), rows={
        4: [("baseline", "e7:zip_default_n4", 8), ("R (ours)", "e7:zip_n4_m0111", 7)],
        8: [("baseline", "e7:zip_default_n8", 16), ("R (ours)", "e7:zip_n8_m00111111", 14)]}),
}


def table1():
    rows = []
    for model, d in T1.items():
        tcfg, tcalls = d["teacher"]
        T = {s: mean(tcfg, s, "sim_eres2net") for s in ("en", "zh")}
        for n, items in d["rows"].items():
            B = {s: mean(items[0][1], s, "sim_eres2net") for s in ("en", "zh")}
            for meth, cfg, calls in items:
                r = dict(model=model, NFE=n, method=meth, calls=calls)
                for s in ("en", "zh"):
                    r[f"SIM {s}"] = mean(cfg, s, "sim_eres2net")
                r["WavLM en"], r["ECAPA en"] = mean(cfg, "en", "sim_wavlm"), mean(cfg, "en", "sim_ecapa")
                for s in ("en", "zh"):
                    r[f"WER {s}"] = mean(cfg, s, "wer")
                for s in ("en", "zh"):
                    r[f"UTMOS {s}"] = mean(cfg, s, "utmos")
                for s in ("en", "zh"):
                    gap = T[s] - B[s]
                    r[f"rec {s}"] = (r[f"SIM {s}"] - B[s]) / gap * 100 if gap > 0.42 and meth != "baseline" else np.nan
                rows.append(r)
        r = dict(model=model, NFE=f"teacher", method="teacher", calls=tcalls)
        for s in ("en", "zh"):
            r[f"SIM {s}"] = T[s]
        r["WavLM en"], r["ECAPA en"] = mean(tcfg, "en", "sim_wavlm"), mean(tcfg, "en", "sim_ecapa")
        for s in ("en", "zh"):
            r[f"WER {s}"], r[f"UTMOS {s}"] = mean(tcfg, s, "wer"), mean(tcfg, s, "utmos")
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E9_table1.csv", index=False)
    F = D.copy()
    for c in F.columns[4:]:
        F[c] = F[c].map(lambda x: "" if pd.isna(x) else (f"{x:.0f}%" if c.startswith("rec") else f"{x:.2f}"))
    return F


# ---------------- Table 3 ----------------
def table3():
    P = pd.read_csv(ROOT / "results/E8p_pairs.csv")
    Q = P[(P.group == "G3") & (P.set == "en")]
    rows = []
    for _, r in Q.iterrows():
        rows.append(dict(branch=r.branch, NFE=int(r.nfe), **{"SIM no rule": f"{r.def_sim_eres2net:.2f}",
                                                             "SIM + R": f"{r.R_sim_eres2net:.2f}",
                                                             "ΔSIM (R)": f"{r.d_sim_eres2net:+.2f}" + ("*" if r.d_sim_eres2net_p < .05 else ""),
                                                             "WER no rule": f"{r.def_wer:.2f}",
                                                             "ΔWER (R)": f"{r.d_wer:+.2f}" + ("*" if r.d_wer_p < .05 else "")}))
    T3a = pd.DataFrame(rows)
    g = []
    for model, f in [("CosyVoice2", "E8p_guidance.csv"), ("CosyVoice3", "E8p8_guidance_cv3.csv"), ("IndexTTS-2", "E8p8_guidance_idx.csv")]:
        D = pd.read_csv(ROOT / "results" / f)
        S = D.groupby("branch").mean(numeric_only=True)
        g.append({"model": model, "RMS g1 all": f"{S.loc['all', 'rms']:.2f}", "RMS g1 token-null": f"{S.loc['token', 'rms']:.2f}",
                  "RMS g1 speaker-null": f"{S.loc['speaker', 'rms']:.2f}",
                  "timbre share all / speaker": f"{S.loc['all', 'timbre_share'] * 100:.0f}% / {S.loc['speaker', 'timbre_share'] * 100:.0f}%",
                  "cos(all, token) / cos(all, speaker)": f"{S.loc['cos', 'cos_all_token']:.2f} / {S.loc['cos', 'cos_all_speaker']:.2f}"})
    return T3a, pd.DataFrame(g)


# ---------------- Table 4 ----------------
def table4():
    G = pd.read_csv(ROOT / "results/E8p8_geometry.csv")
    G = G[(G.emb == "eres2net") & G.measure.isin(["var_ratio", "ddcond"])]
    rows = []
    for model in ["CosyVoice2", "CosyVoice3", "IndexTTS-2"]:
        g = G[G.model == model]
        b = g[g.branch == "R (ours)"]
        cell = lambda q, m, c: " / ".join(f"{q[(q.measure == m) & (q.set == s)][c].iloc[0]:.3f}" for s in ["en", "zh"])  # noqa: E731
        rows.append(dict(model=model, method="baseline NFE2", **{"var ratio en / zh": cell(b, "var_ratio", "default"),
                                                                  "Δd_cond en / zh": cell(b, "ddcond", "default")}))
        for br in ["R (ours)", "R′ (Sel-CFG input_text)", "mid+R (ours)"]:
            q = g[g.branch == br]
            if len(q):
                rows.append(dict(model=model, method=br, **{"var ratio en / zh": cell(q, "var_ratio", "R"),
                                                             "Δd_cond en / zh": cell(q, "ddcond", "R")}))
    return pd.DataFrame(rows)


# ---------------- Table 5 ----------------
FEAT = ["f0_mean_st", "f0_sd_st", "rate_active"]


def pvar(cfg, s):
    A_ = pd.concat([pd.read_csv(J9.wav_dir(cfg, k, s) / "_metrics" / "acoustic.csv").assign(k=k) for k in range(8)])
    return A_.groupby("utt_id")[FEAT].var()


def table5():
    rows = []
    T = {s: pvar("e82_cv2_all_n10_def", s) for s in ("en", "zh")}
    G = pd.read_csv(ROOT / "results/E8p8_geometry.csv")
    G = G[(G.emb == "eres2net") & (G.model == "CosyVoice2")]
    for meth, cfg, br, col in [("CosyVoice2 baseline NFE 2", "e82_cv2_all_n2_def", "R (ours)", "default"),
                               ("CosyVoice2 R (ours) NFE 2", "e82_cv2_all_n2_R", "R (ours)", "R"),
                               ("CosyVoice2 R′ (Sel-CFG) NFE 2", "e82_cv2_speaker_n2_def", "R′ (Sel-CFG input_text)", "R"),
                               ("CosyVoice2 mid+R (ours) 3 calls", "A_cv2_midR", "mid+R (ours)", "R")]:
        q = G[(G.branch == br) & (G.set == "en")]
        P = pvar(cfg, "en")
        pr = [np.exp(np.nanmean(np.log((P[f] + 1e-9) / (T["en"][f] + 1e-9)))) for f in FEAT]
        rows.append({"method": meth, "ΔSIM vs teacher (en)": f"{mean(cfg, 'en', 'sim_eres2net') - mean('e82_cv2_all_n10_def', 'en', 'sim_eres2net'):+.2f}",
                     "var ratio": f"{q[q.measure == 'var_ratio'][col].iloc[0]:.2f}", "Δd_cond": f"{q[q.measure == 'ddcond'][col].iloc[0]:.3f}",
                     "prosody var ratio F0 mean / F0 SD / rate": " / ".join(f"{x:.2f}" for x in pr),
                     "UTMOS − teacher (en)": f"{mean(cfg, 'en', 'utmos') - mean('e82_cv2_all_n10_def', 'en', 'utmos'):+.2f}"})
    # A1 (E5) and DMOSpeech2 student (E3) from their summaries
    E5g = pd.read_csv(ROOT / "results/E5_geometry_summary.csv")
    E5p = pd.read_csv(ROOT / "results/E5_prosody.csv")
    E5s = pd.read_csv(ROOT / "results/E5_seed0_summary.csv")
    a = E5g[(E5g.alg == "cv2_a1_n3") & (E5g.set == "en") & (E5g.emb == "eres2net")]
    p = E5p[(E5p.alg == "cv2_a1_n3") & (E5p.set == "en")].set_index("feat")
    s = E5s[(E5s.config == "cv2_a1_n3") & (E5s.set == "en")].iloc[0]
    rows.append({"method": "CosyVoice2 A1 (re-interpolation) NFE 3", "ΔSIM vs teacher (en)": f"{s.vsTeacher_sim_eres2net * 100:+.2f}",
                 "var ratio": f"{np.exp(a[a.measure == 'lvr'].alg_est.iloc[0]):.2f}" if a[a.measure == 'lvr'].alg_est.iloc[0] < 0 else f"{a[a.measure == 'lvr'].alg_est.iloc[0]:.2f}",
                 "Δd_cond": f"{a[a.measure == 'ddcond'].alg_est.iloc[0]:.3f}",
                 "prosody var ratio F0 mean / F0 SD / rate": " / ".join(f"{p.loc[f, 'alg_vs_teacher']:.2f}" for f in FEAT),
                 "UTMOS − teacher (en)": f"{s.vsTeacher_utmos:+.2f}"})
    rows.append({"method": "DMOSpeech2 student NFE 2 (vs teacher 32; E3)", "ΔSIM vs teacher (en)": "+5.26",
                 "var ratio": "0.50", "Δd_cond": "0.064", "prosody var ratio F0 mean / F0 SD / rate": "0.26 / 0.35 / 0.11",
                 "UTMOS − teacher (en)": "+0.36"})  # E1_summary: student NFE 2 3.69 vs teacher NFE 32 3.33
    return pd.DataFrame(rows)


# ---------------- Table 7 ----------------
def table7():
    T = pd.read_csv(ROOT / "results/E8p7_table7.csv")
    keep = ["baseline", "R (ours)", "Kynk [0.1,0.9]", "Kynk [0.2,0.8]", "CFG-Zero* zi", "CFG-Zero* s*", "MAMBO-G(2)", "MAMBO-G(4)",
            "R′ (Sel-CFG input_text)", "Sel-CFG def_text", "Sel-CFG def_text+R"]
    rows = []
    for (model, n) in [("CosyVoice2", 2), ("CosyVoice3", 2), ("CosyVoice2", 10)]:
        for meth in keep:
            q = T[(T.model == model) & (T.nfe == n) & (T.method == meth)].set_index("set")
            if len(q) < 2:
                continue
            e, z = q.loc["en"], q.loc["zh"]
            d = lambda x, p: "" if pd.isna(x) else f"{x:+.2f}" + ("*" if p < .05 else "")  # noqa: E731
            rows.append({"model": model, "NFE": n, "method": meth, "calls": int(e.calls), "extra": e.extra if isinstance(e.extra, str) else "",
                         "SIM en / zh": f"{e.sim_eres2net:.2f} / {z.sim_eres2net:.2f}",
                         "Δ vs R en / zh": "—" if meth == "R (ours)" else f"{d(e.vs_R_d_sim_eres2net, e.vs_R_d_sim_eres2net_p)} / {d(z.vs_R_d_sim_eres2net, z.vs_R_d_sim_eres2net_p)}",
                         "WER en / zh": f"{e.wer:.2f} / {z.wer:.2f}", "UTMOS en / zh": f"{e.utmos:.2f} / {z.utmos:.2f}"})
    return pd.DataFrame(rows)


# ---------------- Table 8 ----------------
def table8():
    C = pd.read_csv(ROOT / "results/E8p9c_components.csv")
    N = pd.read_csv(ROOT / "results/E8p9c_nfe.csv")
    rows = []
    for model in ["CosyVoice2", "IndexTTS-2"]:
        for n in [2, 3, 4]:
            q = C[(C.model == model) & (C.nfe == n)].set_index("set")
            w = N[(N.model == model) & (N.cond == "w0")].set_index(["set", "nfe"])
            wd = [w.loc[(s, n), "loss"] - w.loc[(s, 25), "loss"] for s in ("en", "zh")]
            rows.append({"model": model, "NFE": n, "total loss en / zh": f"{q.loc['en', 'total']:.2f} / {q.loc['zh', 'total']:.2f}",
                         "(i) first-step extrapolation": f"{q.loc['en', 'i_first_step']:.2f} / {q.loc['zh', 'i_first_step']:.2f}",
                         "(ii) truncation": f"{q.loc['en', 'ii_truncation']:.2f} / {q.loc['zh', 'ii_truncation']:.2f}",
                         "w=0 NFE-dependence (n − 25)": f"{wd[0]:+.2f} / {wd[1]:+.2f}"})
    return pd.DataFrame(rows)


# ---------------- Table 9 ----------------
def table9():
    P = pd.read_csv(ROOT / "results/E8p_pairs.csv")
    O = pd.read_csv(ROOT / "results/E8p8_ops.csv")
    rows = []
    for s, lab in [("mm_ko", "ko"), ("mm_ja", "ja"), ("mm_de", "de"), ("hard", "Seed hard")]:
        for model in ["CosyVoice2", "CosyVoice3", "IndexTTS-2"]:
            g = "G7" if s.startswith("mm") else "G1"
            q = P[(P.group == g) & (P.model == model) & (P.set == s) & (P.label == "NFE 2")]
            if q.empty:
                continue
            r = q.iloc[0]
            mr = O[(O.model == model) & (O.set == s) & (O.method == "mid+R (ours)")]
            note = ""
            if (model, lab) in [("CosyVoice3", "ja"), ("CosyVoice2", "de")]:
                note = "CER/WER excluded (frontend/language)"
            rows.append({"set": lab, "model": model, "teacher": f"{r.T_sim_eres2net:.2f}", "baseline NFE2": f"{r.def_sim_eres2net:.2f}",
                         "R (ours)": f"{r.R_sim_eres2net:.2f}", "recovery": f"{r.rec_eres2net * 100:.0f}%",
                         "mid+R (ours)": f"{mr.sim_eres2net.iloc[0]:.2f}" if len(mr) else "",
                         "CER/WER baseline → R": f"{r.def_wer:.2f} → {r.R_wer:.2f}", "note": note})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t1 = table1()
    t3a, t3b = table3()
    t4, t5, t7, t8, t9 = table4(), table5(), table7(), table8(), table9()
    for i, t in [(3, t3a), (4, t4), (5, t5), (7, t7), (8, t8), (9, t9)]:
        t.to_csv(ROOT / f"results/E9_table{i}.csv", index=False)
    t3b.to_csv(ROOT / "results/E9_table3b.csv", index=False)
    parts = {"T1": md(t1), "T3a": md(t3a), "T3b": md(t3b), "T4": md(t4), "T5": md(t5), "T7": md(t7), "T8": md(t8), "T9": md(t9)}
    for k, v in parts.items():
        (ROOT / "results" / f"_E9_{k}.md").write_text(v, encoding="utf-8")
    print(md(t1))
    print(md(t5))
