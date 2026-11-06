"""Markdown tables for reports/E8p10_baselines2026.md from results/E8p7_table7.csv + results/E8p10_table.csv (no re-computation).
python scripts/e8p10_tables.py > logs/e8p10_tables.md"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YEAR = {"baseline": "", "R (ours)": "", "R (ours) + PCI": "", "R (ours) + TFR": "", "TFR": "2026.07", "PCI": "2026.10",
        "TV-CFG symmetric": "2025.09 / ICLR'26", "TV-CFG stepup": "2025.09 / ICLR'26", "TV-CFG symmetric (last-delta fix)": "2025.09 (ours fix)",
        "JRR-CFG": "2026.06", "JRR-CFG sparse": "2026.06", "Kynk [0.1,0.9]": "2024.04", "Kynk [0.2,0.8]": "2024.04",
        "Kynk [0.1,0.8]": "2024.04", "CFG-Zero* zi": "2025.03", "CFG-Zero* zi+s*": "2025.03", "CFG-Zero* s*": "2025.03",
        "MAMBO-G(0.5)": "2025.08", "MAMBO-G(1)": "2025.08", "MAMBO-G(2)": "2025.08", "MAMBO-G(4)": "2025.08",
        "R′ (Sel-CFG input_text)": "2025.09", "Sel-CFG def_text": "2025.09", "Sel-CFG def_text+R": "2025.09"}


def f(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def star(d, p):
    return "" if pd.isna(d) else f"{d:+.2f}{'*' if p < 0.05 else ''}"


def load():
    a = pd.read_csv(ROOT / "results/E8p7_table7.csv")
    a["src"] = "E8′-7"
    b = pd.read_csv(ROOT / "results/E8p10_table.csv")
    b["src"] = "E8′-10"
    b = b[~b.method.isin(["baseline", "R (ours)"]) | ~b.set.isin([])]
    return a, b


def wide(T):
    rows = []
    for (model, nfe, method), g in T.groupby(["model", "nfe", "method"], sort=False):
        e, z = g[g.set == "en"], g[g.set == "zh"]
        if e.empty or z.empty:
            continue
        e, z = e.iloc[0], z.iloc[0]
        rows.append(dict(model=model, nfe=nfe, method=method, year=YEAR.get(method, ""), src=e.src, calls=e.calls,
                         sim=f"{f(e.sim_eres2net)} / {f(z.sim_eres2net)}",
                         d_base=f"{star(e.get('vs_base_d_sim_eres2net'), e.get('vs_base_d_sim_eres2net_p'))} / "
                                f"{star(z.get('vs_base_d_sim_eres2net'), z.get('vs_base_d_sim_eres2net_p'))}",
                         d_R=f"{star(e.get('vs_R_d_sim_eres2net'), e.get('vs_R_d_sim_eres2net_p'))} / "
                             f"{star(z.get('vs_R_d_sim_eres2net'), z.get('vs_R_d_sim_eres2net_p'))}",
                         rec="" if pd.isna(e.get("recovery")) else f"{100 * e.recovery:.0f}% / {100 * z.recovery:.0f}%",
                         wavlm=f"{f(e.sim_wavlm)} / {f(z.sim_wavlm)}", ecapa=f"{f(e.sim_ecapa)} / {f(z.sim_ecapa)}",
                         wer=f"{f(e.wer)} / {f(z.wer)}", utmos=f"{f(e.utmos)} / {f(z.utmos)}"))
    return pd.DataFrame(rows)


def md(D, cols, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in D.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    a, b = load()
    A = wide(a)
    B = wide(b)
    allr = pd.concat([A, B[~B.method.isin(["baseline", "R (ours)"]) | ~B.set_index(["model", "nfe", "method"]).index.isin(
        A.set_index(["model", "nfe", "method"]).index)]]) if not A.empty else B
    allr = allr.drop_duplicates(subset=["model", "nfe", "method"], keep="first")
    order = {m: i for i, m in enumerate(["baseline", "R (ours)", "R (ours) + PCI", "R (ours) + TFR"])}
    allr["o"] = allr.method.map(lambda m: order.get(m, 10))
    allr = allr.sort_values(["model", "nfe", "o", "src", "method"])
    cols = ["model", "nfe", "method", "year", "calls", "sim", "d_base", "d_R", "rec", "wavlm", "ecapa", "wer", "utmos", "src"]
    head = ["model", "NFE", "method", "source", "calls", "SIM en / zh", "Δ vs baseline", "Δ vs R", "recovery en / zh", "WavLM", "ECAPA",
            "WER %", "UTMOS", "experiment"]
    print(md(allr, cols, head))
