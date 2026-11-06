"""E11 (v10.2) analysis for GLM-TTS.   python scripts/e11_analyze.py   (env_eval)
Outputs results/E11_blockA.csv, E11_blockB.csv, E11_tstar.csv, E11_w0.csv, E11_geometry.csv, E11_g1.csv, E11_gamma.csv,
E11_law.csv, E11_wsign.csv, E11_onestep.csv and a printed N1–N10 summary. Paired comparisons as everywhere (bootstrap
95% CI, Wilcoxon; SIM x100; WER %). Recovery = (X - baseline) / (teacher - baseline) when teacher - baseline > eps 0.42.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e11_jobs as J  # noqa: E402

ROOT = J.ROOT
M = "GLM-TTS"
EA.A.TEACHER[M] = "g_base_n10"
OBJ = ["utmos", "utmosv2", "dnsmos_ovrl", "nisqa_mos"]
COLS = ["sim_eres2net", "sim_wavlm", "sim_ecapa", "wer", "wer_wh"] + OBJ


def L(name, s, k=0):
    return EA.load_metrics(J.wav_dir(name, k, s))


def row(name, s, ref=None, T=None, **extra):
    X = L(name, s)
    if X is None:
        return None
    r = dict(config=name, set=s, n=len(X), **extra)
    r.update(EA.compare(X, ref, T))
    if ref is not None and T is not None:
        gap = T.sim_eres2net.mean() - ref.sim_eres2net.mean()
        r["teacher_minus_base"] = gap
        r["recovery"] = (X.sim_eres2net.mean() - ref.sim_eres2net.mean()) / gap if gap > 0.42 else np.nan
    if T is not None:
        for c in OBJ:
            if c in X and c in T:
                r[f"{c}_minus_teacher"] = X[c].mean() - T[c].mean()
    return r


def main():
    pd.set_option("display.width", 260)
    out = {}
    # ---- block A ----
    A = []
    for s in ("en", "zh"):
        T = L("g_base_n10", s)
        for n in (10, 5, 3, 2, 1):
            A.append(row(f"g_base_n{n}", s, None, T if n != 10 else None, nfe=n))
    A = pd.DataFrame([a for a in A if a])
    A.to_csv(ROOT / "results/E11_blockA.csv", index=False)
    out["A"] = A
    # ---- block B ----
    B = []
    for s in ("en", "zh"):
        T = L("g_base_n10", s)
        for n in (2, 3, 10):
            base = L(f"g_base_n{n}", s)
            for cfg, lab in [(f"g_R_n{n}", "R (ours)"), (f"g_allnull_n{n}", "all-null CFG"), (f"g_allnullR_n{n}", "all-null CFG + R")]:
                if n == 10 and "allnull" in cfg:
                    continue
                r = row(cfg, s, base, T if n != 10 else None, nfe=n, method=lab)
                if r:
                    B.append(r)
        r = row("g_midR", s, L("g_base_n2", s), T, nfe=2, method="mid+R (ours)")
        B.append(r)
        if s == "en":
            B.append(row("g_kynk_n2", s, L("g_R_n2", s), T, nfe=2, method="Kynk [0.1,0.9] (vs R)"))
    B = pd.DataFrame([b for b in B if b])
    # all-null + R recovery measured against the all-null baseline
    B.to_csv(ROOT / "results/E11_blockB.csv", index=False)
    out["B"] = B
    # ---- t* sensitivity (en) ----
    ts_map = {2: {0.05: "g_R_n2", 0.1: "g_R_n2", 0.2: "g_R_n2", 0.3: "g_w0_n2"},
              3: {0.05: "g_R_n3", 0.1: "g_R_n3", 0.2: "g_ts_n3_m001", 0.3: "g_ts_n3_m001"},
              10: {0.05: "g_R_n10", 0.1: "g_R_n10", 0.2: "g_ts_n10_m0000011111", 0.3: "g_ts_n10_m0000001111"}}
    TS = []
    T = L("g_base_n10", "en")
    for n, mp in ts_map.items():
        base = L(f"g_base_n{n}", "en")
        for t, cfg in mp.items():
            X = L(cfg, "en")
            TS.append(dict(nfe=n, tstar=t, config=cfg, sim=X.sim_eres2net.mean(), d_vs_base=X.sim_eres2net.mean() - base.sim_eres2net.mean(),
                           wer=X.wer.mean(), utmos=X.utmos.mean()))
    TS = pd.DataFrame(TS)
    TS.to_csv(ROOT / "results/E11_tstar.csv", index=False)
    out["tstar"] = TS
    # ---- (ii) diagnostic: w = 0 curve ----
    W0 = pd.DataFrame([dict(nfe=n, sim=L(f"g_w0_n{n}", "en").sim_eres2net.mean(), wer=L(f"g_w0_n{n}", "en").wer.mean()) for n in (2, 3, 4, 8, 10)])
    W0.to_csv(ROOT / "results/E11_w0.csv", index=False)
    out["w0"] = W0
    # ---- geometry (8 seeds, 100 utt) ----
    G = EA.geometry_on([(M, "baseline", "g_base_n2"), (M, "R (ours)", "g_R_n2"), (M, "mid+R (ours)", "g_midR")],
                       {"en": "data/lists/en_100", "zh": "data/lists/zh_100"}, J.wav_dir)
    G.to_csv(ROOT / "results/E11_geometry.csv", index=False)
    out["geo"] = G
    # ---- dumps: g1 branches, gamma, onestep ----
    Dd = pd.read_csv(ROOT / "results/E11_dumps.csv")
    g1 = Dd[Dd.check == "g1"].groupby("branch").value.mean()
    g1.to_csv(ROOT / "results/E11_g1.csv")
    gam = Dd[Dd.check == "gamma"].groupby("t").value.mean()
    gam = gam / gam.iloc[0]
    gam.to_csv(ROOT / "results/E11_gamma.csv")
    one = Dd[Dd.check == "onestep"][["rho_nn", "rho_mean", "sigma", "d_TT", "g_rms", "seed_dep"]].mean()
    one["rho_over_sigma"] = (Dd[Dd.check == "onestep"].rho_nn / Dd[Dd.check == "onestep"].sigma).mean()
    one["rho_over_g"] = (Dd[Dd.check == "onestep"].rho_nn / Dd[Dd.check == "onestep"].g_rms).mean()
    one.to_csv(ROOT / "results/E11_onestep.csv")
    out["g1"], out["gamma"], out["one"] = g1, gam, one
    # ---- E1 law (en): loss = SIM(teacher) - SIM(cond), E1 = w * g1(branch) * dt1 ----
    sT = L("g_base_n10", "en").sim_eres2net.mean()
    cos1 = lambda n: 1 - np.cos(np.pi / 2 / n)  # noqa: E731
    pts = [(f"g_base_n{n}", "NFE", "all", cos1(n)) for n in (1, 2, 3, 5)]
    pts += [(f"g_t1_{t}_def", "t1", "all", t) for t in (0.1, 0.2, 0.3, 0.5, 0.7)]
    pts += [(f"g_allnull_n{n}", "branch", "allnull", cos1(n)) for n in (2, 3)]
    law = []
    for cfg, kind, br, dt1 in pts:
        X = L(cfg, "en")
        if X is None:
            continue
        law.append(dict(config=cfg, kind=kind, branch=br, dt1=dt1, E1=0.7 * g1[br] * dt1, loss=sT - X.sim_eres2net.mean()))
    law = pd.DataFrame(law)
    law.to_csv(ROOT / "results/E11_law.csv", index=False)
    out["law"] = law
    # ---- w1 sweep (en) ----
    wcfg = {-0.43: "g_ws_w-0.43", -0.2: "g_ws_w-0.2", 0.0: "g_R_n2", 0.2: "g_ws_w0.2", 0.35: "g_ws_w0.35", 0.7: "g_base_n2",
            1.05: "g_ws_w1.05"}
    WS = pd.DataFrame([dict(w1=w, sim=L(c, "en").sim_eres2net.mean(), wer=L(c, "en").wer.mean(), utmos=L(c, "en").utmos.mean())
                       for w, c in wcfg.items() if L(c, "en") is not None])
    WS.to_csv(ROOT / "results/E11_wsign.csv", index=False)
    out["ws"] = WS
    # ---- objective-quality resolution eps_utt (teacher 8 seeds, en/zh 100; seed 0 restricted to the same 100) ----
    E = []
    for s, s100 in (("en", "en100"), ("zh", "zh100")):
        Ds = [L("g_base_n10", s if k == 0 else s100, k) for k in range(8)]
        u = [x for x in Ds[1].index if x in Ds[0].index]
        for p in ["utmos", "utmosv2", "dnsmos_ovrl", "nisqa_mos", "sim_eres2net"]:
            if all(D is not None and p in D for D in Ds):
                Mx = np.stack([D.loc[u, p].values for D in Ds])
                E.append(dict(set=s, predictor=p, n_utt=len(u), eps_utt=Mx.std(0, ddof=1).mean(), eps_sys=Mx.mean(1).std(ddof=1)))
    E = pd.DataFrame(E)
    E.to_csv(ROOT / "results/E11_objq_eps.csv", index=False)
    out["eps"] = E
    for k, v in out.items():
        print(f"===== {k}")
        print(v.round(4).to_string() if hasattr(v, "round") else v)
    rho, p = spearmanr(law.E1, law.loss)
    print(f"N8 Spearman(loss, E1) = {rho:+.3f} (p {p:.3g}, n {len(law)})")


if __name__ == "__main__":
    main()
