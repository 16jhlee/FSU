"""E16 analysis (v10.9 §2-4): the LibriSpeech-PC cells of Table II for CosyVoice 3 (Official, three rules, FSU, FSU-Mid,
teacher NFE 10) and GLM-TTS all-null (all-null CFG, + FSU, + FSU-Mid; teacher = official speaker-null NFE 10 of E11-D;
the official speaker-null rows of E11-D are listed for context). Paired by utterance: SIM x100 (ERes2Net main, WavLM /
ECAPA alongside), WER %, UTMOS; d = method - reference with bootstrap 10k 95% CI and Wilcoxon p; Holm over the methods of
each decoder block (per comparison and metric). recovery = (X - base) / (teacher - base) on ERes2Net SIM, base = Official
(CosyVoice 3) or all-null CFG (GLM all-null), as in Table II.
Outputs results/E16_lspc_fill.csv, results/E16_verdicts.csv.
python scripts/e16_analyze.py   (env_eval)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import e11d_jobs as J11D  # noqa: E402
import e15_analyze as E15  # noqa: E402
import e16_jobs as J  # noqa: E402
import e8p_analyze as A  # noqa: E402

ROOT = A.ROOT
E15.RES["e16"] = J.wav_dir
E15.RES["e11d"] = J11D.wav_dir
METRICS = E15.METRICS
rng = np.random.default_rng(0)
# (decoder block, key, method, config, how)
BLOCKS = {
    "CosyVoice3": [("official", "Official CFG (2)", "e16:cv3_base_n2", "run"),
                   ("tvsf", "TV-CFG (corrected) (2)", "e16:cv3_tvsf", "run"),
                   ("seltext", "Sel.-CFG (input-text) (2)", "e16:cv3_seltext", "run"),
                   ("mambo4", "MAMBO-G (2)", "e16:cv3_mambo4", "run"),
                   ("fsu", "FSU (ours) (2)", "e16:cv3_R_n2", "run"),
                   ("fsumid", "FSU-Mid (ours) (2)", "e16:cv3_midR", "run"),
                   ("teacher", "Teacher (10)", "e16:cv3_teacher", "run")],
    "GLM-TTS all-null": [("official", "All-null CFG (2)", "e16:glm_allnull_n2", "run"),
                         ("fsu", "All-null + FSU (ours) (2)", "e16:glm_allnullR_n2", "run"),
                         ("fsumid", "All-null + FSU-Mid (ours) (2)", "e16:glm_allnull_midR", "run"),
                         ("teacher", "Teacher (10, speaker-null)", "e11d:d_teacher", "ref"),
                         ("spk_official", "Official CFG (2, speaker-null)", "e11d:d_base_n2", "ref"),
                         ("spk_fsu", "FSU (2, speaker-null)", "e11d:d_R_n2", "ref"),
                         ("spk_fsumid", "FSU-Mid (2, speaker-null)", "e11d:d_midR", "ref")],
}
VS = ("official", "fsu", "fsumid", "teacher")


def stat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (10000, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5]), (wilcoxon(x).pvalue if np.any(x != 0) else 1.0)


def table():
    rows = []
    for dec, items in BLOCKS.items():
        cfg = {k: c for k, _, c, _ in items}
        B, T = E15.loaded(cfg["official"], "lspc"), E15.loaded(cfg["teacher"], "lspc")
        gap = (T.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100 if T is not None and B is not None else np.nan
        for key, label, c, how in items:
            r = dict(decoder=dec, set="lspc", key=key, method=label, how=how, config=c)
            D = E15.loaded(c, "lspc")
            if D is None:
                rows.append(r)
                continue
            r["n"] = len(D)
            for m in METRICS:
                r[m] = D[m].mean() * (100 if m.startswith("sim") else 1)
            r["teacher_minus_base"] = gap
            r["recovery"] = (r["sim_eres2net"] - B.sim_eres2net.mean() * 100) / gap if np.isfinite(gap) else np.nan
            for tg in VS:
                Rf = E15.loaded(cfg[tg], "lspc")
                if tg == key or Rf is None:
                    continue
                ix = Rf.index.intersection(D.index)
                for m in METRICS:
                    est, lo, hi, p = stat((D.loc[ix, m] - Rf.loc[ix, m]) * (100 if m.startswith("sim") else 1))
                    r[f"vs_{tg}_d_{m}"], r[f"vs_{tg}_d_{m}_lo"], r[f"vs_{tg}_d_{m}_hi"], r[f"vs_{tg}_d_{m}_p"] = est, lo, hi, p
            rows.append(r)
    F = pd.DataFrame(rows)
    for dec, idx in F[F.how == "run"].groupby("decoder").groups.items():  # Holm over the generated methods only
        for tg in VS:
            for m in METRICS:
                col = f"vs_{tg}_d_{m}_p"
                if col in F:
                    F.loc[idx, col + "_holm"] = E15.holm(F.loc[idx, col])
    F["cell"] = [f"{s:.2f}" + (f" ({rc * 100:.0f}%)" if k not in ("official", "teacher") and not k.startswith("spk_")
                                and np.isfinite(rc) else "") if pd.notna(s) else ""
                 for s, rc, k in zip(F.get("sim_eres2net", np.nan), F.get("recovery", np.nan), F.key)]
    return F


def verdicts(F):
    V = []
    h = F[F.decoder == "CosyVoice3"].set_index("key")
    g = F[F.decoder == "GLM-TTS all-null"].set_index("key")
    if pd.notna(h.recovery.get("fsu")):
        V.append(dict(check="L1 CosyVoice 3 FSU recovery >= 80%", decoder="CosyVoice3", method="fsu",
                      value=h.recovery["fsu"], ok=bool(h.recovery["fsu"] >= 0.80)))
    for rule in ("tvsf", "seltext", "mambo4"):
        if pd.notna(h.sim_eres2net.get(rule)) and pd.notna(h.sim_eres2net.get("fsumid")):
            V.append(dict(check="L2 CosyVoice 3 FSU-Mid SIM > rule", decoder="CosyVoice3", method=rule,
                          value=h.sim_eres2net["fsumid"] - h.sim_eres2net[rule],
                          ok=bool(h.sim_eres2net["fsumid"] > h.sim_eres2net[rule])))
    if pd.notna(g.sim_eres2net.get("fsumid")) and pd.notna(g.sim_eres2net.get("teacher")):
        V.append(dict(check="L3 GLM all-null FSU-Mid SIM >= speaker-null teacher", decoder="GLM-TTS all-null",
                      method="fsumid", value=g.sim_eres2net["fsumid"] - g.sim_eres2net["teacher"],
                      ok=bool(g.sim_eres2net["fsumid"] >= g.sim_eres2net["teacher"])))
    for _, x in F[(F.how == "run") & (F.key != "official") & F.sim_eres2net.notna()].iterrows():
        d, p = x.get("vs_official_d_wer", np.nan), x.get("vs_official_d_wer_p_holm", np.nan)
        V.append(dict(check="L4 WER not worse than the base (Holm)", decoder=x.decoder, method=x.key, value=d,
                      ok=bool(not (d > 0 and p < 0.05))))
    return pd.DataFrame(V)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    F = table()
    F.to_csv(ROOT / "results/E16_lspc_fill.csv", index=False)
    V = verdicts(F)
    V.to_csv(ROOT / "results/E16_verdicts.csv", index=False)
    cols = ["decoder", "method", "n", "sim_eres2net", "sim_wavlm", "sim_ecapa", "recovery", "cell", "vs_official_d_sim_eres2net",
            "vs_official_d_sim_eres2net_p_holm", "vs_fsumid_d_sim_eres2net", "vs_teacher_d_sim_eres2net", "wer",
            "vs_official_d_wer", "vs_official_d_wer_p_holm", "utmos"]
    print(F[[c for c in cols if c in F]].round(3).to_string(index=False))
    print(V.round(3).to_string(index=False))
