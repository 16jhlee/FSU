"""E8'-10 analysis (v9.1 addendum): 2026 baselines vs baseline / R (ours), same machinery as e8p7_analyze.py. env_eval.
python scripts/e8p10_analyze.py
Paired by utterance (seed 0); SIM x100 (ERes2Net main; WavLM / ECAPA alongside), WER % (utterance mean, as e8p_analyze.load),
UTMOS. d = method - reference with bootstrap 95% CI and Wilcoxon p. recovery = (method - baseline) / (teacher - baseline)
on ERes2Net SIM, teacher = official 10 steps. Calls = network evaluations (CFG step 2, w=0 step 1, JRR step 4).
Outputs results/E8p10_table.csv, E8p10_geometry.csv, E8p10_zip.csv, E8p10_lspc.csv, E8p10_idx.csv, E8p10_tv_scales.csv
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e8p10_jobs as J  # noqa: E402
import e8p7_jobs as J7  # noqa: E402
import e8p_analyze as A  # noqa: E402
from e7_jobs import wav_dir as e7_wav_dir  # noqa: E402
import ast  # noqa: E402


class RC:  # tv_scales exactly as used by run_cosy.py (extracted from its source; run_cosy imports CosyVoice deps)
    _src = (Path(__file__).parent / "run_cosy.py").read_text(encoding="utf-8")
    _fn = next(n for n in ast.parse(_src).body if isinstance(n, ast.FunctionDef) and n.name == "tv_scales")
    _ns = {"np": np}
    exec(compile(ast.Module([_fn], []), "run_cosy.py", "exec"), _ns)
    tv_scales = staticmethod(_ns["tv_scales"])

ROOT = A.ROOT
MODEL = {"cv2": A.CV2, "cv3": A.CV3}
TEACHER = {"cv2": "e82_cv2_all_n10_def", "cv3": "e81_cv3_n10_def"}
METRICS = [f"sim_{e}" for e in A.EMBS] + ["wer", "utmos"]


def W(name, k, s):
    if name in J.CONFIGS or name in J.ALIAS:
        return J.wav_dir(name, k, s)
    if name.startswith("e7:"):
        return e7_wav_dir(name[3:], k, s)
    return J7.wav_dir(name, k, s)


A.wdir = W


def ts(n):
    return list(1 - np.cos(np.pi / 2 * np.arange(n) / n))


def calls(key, n):
    t = ts(n)
    if key in ("R", "Rpci", "Rtfr"):
        return sum(2 if x >= 0.1 else 1 for x in t)
    if key == "jrr":
        return 4 * n
    if key == "jrrs":
        k = math.ceil(n / 4)
        return 4 * k + 2 * (n - k)
    return 2 * n


def paired(ref, cfg, s):
    D, R = A.load(ref, s), A.load(cfg, s)
    if D is None or R is None:
        return {}
    c = D.index.intersection(R.index)
    return {m: A.stat((R.loc[c, m] - D.loc[c, m]) * (100 if m.startswith("sim") else 1)) for m in METRICS}


def key_of(name):
    return name.split("_")[1]


def table():
    recs = []
    for name, (model, label, n) in J.LABEL.items():
        if model not in MODEL:
            continue
        base, rr = f"{model}_base_n{n}", f"{model}_R_n{n}"
        base = J.ALIAS.get(base, base)
        rr = J.ALIAS.get(rr, rr)
        cfg = J.ALIAS.get(name, name)
        for s in ("en", "zh"):
            D = A.load(cfg, s)
            if D is None:
                continue
            r = dict(model=MODEL[model], method=label, key=key_of(name), nfe=n, set=s, config=name, calls=calls(key_of(name), n),
                     n=len(D))
            for m in METRICS:
                r[m] = D[m].mean() * (100 if m.startswith("sim") else 1)
            for tag, ref in (("vs_base", base), ("vs_R", rr), ("vs_teacher", TEACHER[model])):
                if ref == cfg:
                    continue
                for m, (est, lo, hi, p) in paired(ref, cfg, s).items():
                    r[f"{tag}_d_{m}"], r[f"{tag}_d_{m}_lo"], r[f"{tag}_d_{m}_hi"], r[f"{tag}_d_{m}_p"] = est, lo, hi, p
            T, B = A.load(TEACHER[model], s), A.load(base, s)
            gap = (T.sim_eres2net.mean() - B.sim_eres2net.mean()) * 100
            r["teacher_minus_base"] = gap
            r["recovery"] = (r["sim_eres2net"] - B.sim_eres2net.mean() * 100) / gap if gap > A.EPS["eres2net"] else np.nan
            recs.append(r)
    T = pd.DataFrame(recs)
    T.to_csv(ROOT / "results/E8p10_table.csv", index=False)
    return T


def simple(pairs, out):
    recs = []
    for group, s, ref, cfg in pairs:
        D, R = A.load(ref, s), A.load(cfg, s)
        if D is None or R is None:
            print("missing", group, s, ref, cfg)
            continue
        r = dict(group=group, set=s, ref=ref, config=cfg, n=len(R))
        for m in METRICS:
            sc = 100 if m.startswith("sim") else 1
            r[f"ref_{m}"], r[m] = D[m].mean() * sc, R[m].mean() * sc
            est, lo, hi, p = A.stat((R.loc[D.index.intersection(R.index), m] - D.loc[D.index.intersection(R.index), m]) * sc)
            r[f"d_{m}"], r[f"d_{m}_lo"], r[f"d_{m}_hi"], r[f"d_{m}_p"] = est, lo, hi, p
        recs.append(r)
    T = pd.DataFrame(recs)
    T.to_csv(ROOT / out, index=False)
    return T


def tv_table():
    rows = []
    for n in (2, 3, 4, 10):
        t = (1 - np.cos(np.linspace(0, 1, n + 1) * 0.5 * np.pi)).tolist()
        for kind in ("symmetric", "stepup", "symmetric_fix"):
            g = RC.tv_scales(t, 0.7, kind)
            rows.append(dict(nfe=n, schedule=kind, t_start=" ".join(f"{x:.3f}" for x in t[:-1]), g_k=" ".join(f"{x:.3f}" for x in g),
                             w_first=g[0] - 1, w_mean=float(np.mean(g)) - 1))
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "results/E8p10_tv_scales.csv", index=False)
    return T


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 500)
    T = table()
    cols = ["model", "nfe", "set", "method", "calls", "sim_eres2net", "vs_base_d_sim_eres2net", "vs_base_d_sim_eres2net_p",
            "vs_R_d_sim_eres2net", "vs_R_d_sim_eres2net_p", "recovery", "sim_wavlm", "sim_ecapa", "wer", "vs_base_d_wer_p", "utmos"]
    print(T.sort_values(["model", "nfe", "set", "method"])[cols].round(3).to_string(index=False))
    G = [(A.CV2, J.LABEL[f"cv2_{k}_n2"][1], "e82_cv2_all_n2_def", f"cv2_{k}_n2") for k in ("tfr", "pci", "tvs", "tvu", "jrr", "Rpci", "Rtfr")]
    O = A.geometry(G, W, "results/E8p10_geometry.csv")
    print(O[(O.emb == "eres2net") & O.measure.isin(["var_ratio", "ddcond"])].round(4).to_string(index=False))
    Z = simple([("ZipVoice-Distill NFE4 PCI", s, "zipd_base_n4", "zipd_pci_n4") for s in ("en", "zh")]
               + [("ZipVoice-Distill NFE8 PCI", s, "zipd_base_n8", "zipd_pci_n8") for s in ("en", "zh")]
               + [("ZipVoice NFE4 PCI", s, "zip_base_n4", "zip_pci_n4") for s in ("en", "zh")], "results/E8p10_zip.csv")
    print(Z.round(4).to_string(index=False))
    L = simple([("JRR LibriSpeech-PC 10 steps", "lspc", "lspc_cv2_base_n10", "lspc_cv2_jrr_n10")], "results/E8p10_lspc.csv")
    print(L.round(4).to_string(index=False))
    I = simple([("IndexTTS-2 NFE2 t1=0.2: R vs baseline", s, "idx_t02_def", "idx_t02_R") for s in ("en", "zh")]
               + [("IndexTTS-2 NFE2 t1=0.3: R vs baseline (E8'-8)", s, "C_idx_t03_def", "C_idx_t03_R") for s in ("en", "zh")]
               + [("IndexTTS-2 NFE2 default: R vs baseline (E8')", s, "e81_idx_n2_def", "e81_idx_n2_R") for s in ("en", "zh")],
               "results/E8p10_idx.csv")
    print(I.round(4).to_string(index=False))
    print(tv_table().round(3).to_string(index=False))
