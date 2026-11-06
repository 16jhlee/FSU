"""E13-6a (v10.5.1): attribution controls rebuilt from the existing 8-seed runs, no generation. en_100 / zh_100 x seeds 0-7.
Per seed SIM = cos(generated embedding, prompt embedding) x 100 from _metrics/emb_<emb>.npz ("gen/<utt>", full generated
utterance) and the prompt cache data/cache/prompt_<emb>.npz — exactly eval_metrics.py's sim_<emb> (cos of the full
generated-segment embedding with the prompt-wav embedding). Check first: the seed-0 value must equal metrics.csv sim_<emb>
within 0.01 (x100) for every utterance (stop otherwise).
Conditions (reused runs; config names in the output):
  CosyVoice2: 11 e82_cv2_all_n2_def, 01 e82_cv2_all_n2_R, (0.35, 0.7) ws_cv2_w0.35, 00 = TFR at N=2 (cv2_tfr_n2, E8'-10),
              w1 sweep (E10-3, 9 values), N=4 1111 / 0011 (E8'-9c c_cv2_def_n4 / c_cv2_R_n4)
  CosyVoice3: 11 / 01 (E8'-1), (0.35, 0.7) ws_cv3_w0.35, w1 sweep (E10-3)
  IndexTTS-2: 11 / 01 (E8'-1), N=4 1111 / 0011 (c_idx_def_n4 / c_idx_R_n4)
  GLM-TTS (speaker-null): 11 g_base_n2, 01 g_R_n2 (E11)
  v10.6 E13-10 / v10.7 E14R-1 additions: mid = mid+R (ours) (A_*_midR, g_midR, ga_midR), T = teacher (CV2/CV3 N=10,
  IndexTTS-2 N=25, GLM g_base_n10 / ga_T), GLM all-null 10 = ga_10 (E14-2 run); existing rows unchanged
Outputs: results/E13_seed_sim.csv (utterance x seed x condition x embedder), results/E13_seed_summary.csv (condition mean,
seed SD), results/E13_seed_pairs.csv (paired differences over utterance x seed, utterance-bootstrap CI; per-seed signs),
results/E13_seed_wsweep.csv (argmax w1 per seed). python scripts/e13_seeds.py   (env_eval)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import e10_analyze as EA  # noqa: E402
import e10_jobs as J10  # noqa: E402
import e11_jobs as J11  # noqa: E402
import e8p10_jobs as J10b  # noqa: E402
import e8p8_analyze as M8  # noqa: E402
import e8p9_jobs as J9  # noqa: E402
import e13_jobs as J13  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMBS = ["eres2net", "wavlm", "ecapa"]
rng = np.random.default_rng(0)
W8 = M8.W
WS = lambda m, w: (EA.ws_config(m, w), EA.ws_wdir)  # noqa: E731
COND = {
    "CosyVoice2": {"11": ("e82_cv2_all_n2_def", W8), "01": ("e82_cv2_all_n2_R", W8), "h": ("ws_cv2_w0.35", EA.ws_wdir),
                   "00": ("cv2_tfr_n2", J10b.wav_dir), "N4_1111": ("c_cv2_def_n4", J9.wav_dir), "N4_0011": ("c_cv2_R_n4", J9.wav_dir),
                   "10": ("c2_10", J13.wav_dir),
                   **{f"w1={w}": WS("cv2", w) for w in J10.W1},
                   "mid": ("A_cv2_midR", W8), "T": ("e82_cv2_all_n10_def", W8)},
    "CosyVoice3": {"11": ("e81_cv3_n2_def", W8), "01": ("e81_cv3_n2_R", W8), "h": ("ws_cv3_w0.35", EA.ws_wdir),
                   "10": ("c3_10", J13.wav_dir), "00": ("c3_00", J13.wav_dir),
                   **{f"w1={w}": WS("cv3", w) for w in J10.W1},
                   "mid": ("A_cv3_midR", W8), "T": ("e81_cv3_n10_def", W8)},
    "IndexTTS-2": {"11": ("e81_idx_n2_def", W8), "01": ("e81_idx_n2_R", W8), "N4_1111": ("c_idx_def_n4", J9.wav_dir),
                   "N4_0011": ("c_idx_R_n4", J9.wav_dir), "10": ("ix_10", J13.wav_dir), "h": ("ix_h", J13.wav_dir),
                   "00": ("ix_00", J13.wav_dir), "mid": ("A_idx_midR", W8), "T": ("e81_idx_n25_def", W8)},
    "GLM-TTS speaker-null": {"11": ("g_base_n2", J11.wav_dir), "01": ("g_R_n2", J11.wav_dir), "mid": ("g_midR", J11.wav_dir),
                             "T": ("g_base_n10", J11.wav_dir)},
    "GLM-TTS all-null": {"11": ("ga_11", J13.wav_dir), "01": ("ga_01", J13.wav_dir), "h": ("ga_h", J13.wav_dir),
                         "00": ("ga_00", J13.wav_dir), "10": ("ga_10", J13.wav_dir), "mid": ("ga_midR", J13.wav_dir),
                         "T": ("ga_T", J13.wav_dir)},
}
PAIRS = [("01", "11"), ("01", "10"), ("01", "h"), ("h", "11"), ("01", "00"), ("10", "11"), ("N4_0011", "N4_1111"),
         ("mid", "01"), ("mid", "11"), ("mid", "T")]  # v10.6 E13-10 / v10.7 E14R-1: mid = mid+R (ours), T = teacher (8 seeds)


def cos100(a, b):
    return 100 * (a @ b) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b))


def boot_utt(D):
    """D: utterances x seeds of paired differences -> mean, utterance-bootstrap 95% CI."""
    m = D.mean(1)
    b = rng.choice(m, (5000, len(m))).mean(1)
    return float(m.mean()), *np.percentile(b, [2.5, 97.5])


def main():
    rows, checks = [], []
    for s, lst in (("en", "data/lists/en_100"), ("zh", "data/lists/zh_100")):
        meta = {l.split("|")[0]: l.split("|")[2] for l in (ROOT / (lst + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip()}
        utts = list(meta)
        for e in EMBS:
            prom = dict(np.load(ROOT / "data/cache" / f"prompt_{e}.npz"))
            for model, conds in COND.items():
                for c, (cfg, wd) in conds.items():
                    try:
                        X = EA.emb_stack(cfg, s, e, utts, wd)
                    except (FileNotFoundError, KeyError) as ex:
                        print("missing", model, c, cfg, s, e, str(ex)[:100])
                        continue
                    S = np.stack([cos100(X[u], prom[EA.A.pkey(meta[u])]) for u in utts])  # utts x 8
                    if e == "eres2net" or True:  # seed-0 check against metrics.csv
                        d0 = wd(cfg, 0, s) if (wd(cfg, 0, s) / "_metrics" / "metrics.csv").exists() else wd(cfg, 0, s + "100")
                        M = pd.read_csv(d0 / "_metrics" / "metrics.csv").set_index("utt_id")
                        ref = M.loc[utts, f"sim_{e}"].values * 100
                        checks.append(dict(model=model, cond=c, config=cfg, set=s, emb=e, max_abs_diff=float(np.max(np.abs(S[:, 0] - ref)))))
                    for i, u in enumerate(utts):
                        for k in range(8):
                            rows.append(dict(model=model, cond=c, config=cfg, set=s, emb=e, utt=u, seed=k, sim=S[i, k]))
            print(s, e, flush=True)
    C = pd.DataFrame(checks)
    C.to_csv(ROOT / "results/E13_seed_check.csv", index=False)
    print("seed-0 check: max |diff| =", round(C.max_abs_diff.max(), 6), "(must be <= 0.01)")
    if C.max_abs_diff.max() > 0.01:
        print(C.sort_values("max_abs_diff", ascending=False).head(10).to_string(index=False))
        print("STOP: embedding SIM does not reproduce metrics.csv")
        return
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E13_seed_sim.csv", index=False)
    seedmean = D.groupby(["model", "cond", "config", "set", "emb", "seed"]).sim.mean().reset_index()
    Sm = seedmean.groupby(["model", "cond", "config", "set", "emb"]).sim.agg(["mean", "std"]).reset_index()
    Sm.to_csv(ROOT / "results/E13_seed_summary.csv", index=False)
    P = []
    # existing rows first in the original order (the shared bootstrap rng must reach them in the same state), then the
    # v10.6 / v10.7 additions (pairs with a new condition: mid, T, GLM all-null 10)
    new_c = lambda model, c: c in ("mid", "T") or (model == "GLM-TTS all-null" and c == "10")  # noqa: E731
    jobs = [(key, a, b) for key in sorted(D.groupby(["model", "set", "emb"]).groups) for a, b in PAIRS]
    jobs = [j for j in jobs if not (new_c(j[0][0], j[1]) or new_c(j[0][0], j[2]))] + \
           [j for j in jobs if new_c(j[0][0], j[1]) or new_c(j[0][0], j[2])]
    Wg = {key: g.pivot_table(index=["utt", "seed"], columns="cond", values="sim") for key, g in D.groupby(["model", "set", "emb"])}
    for (model, s, e), a, b in jobs:
        W = Wg[model, s, e]
        if True:
            if a not in W or b not in W:
                continue
            dd = (W[a] - W[b]).unstack("seed")  # utts x 8
            m, lo, hi = boot_utt(dd.values)
            per_seed = dd.mean(0)
            P.append(dict(model=model, set=s, emb=e, comparison=f"{a} - {b}", d=m, lo=lo, hi=hi, n_utt=len(dd),
                          seeds_positive=int((per_seed > 0).sum()), seed_min=float(per_seed.min()), seed_max=float(per_seed.max())))
    P = pd.DataFrame(P)
    P.to_csv(ROOT / "results/E13_seed_pairs.csv", index=False)
    Wr = []
    for (model, s, e), g in seedmean[seedmean.cond.str.startswith("w1=")].groupby(["model", "set", "emb"]):
        g = g.assign(w1=g.cond.str[3:].astype(float))
        for k, gk in g.groupby("seed"):
            best = gk.loc[gk.sim.idxmax()]
            Wr.append(dict(model=model, set=s, emb=e, seed=k, argmax_w1=best.w1, sim_best=best.sim,
                           sim_w0=float(gk[gk.w1 == 0].sim.iloc[0])))
    Wr = pd.DataFrame(Wr)
    Wr.to_csv(ROOT / "results/E13_seed_wsweep.csv", index=False)
    pd.set_option("display.width", 250)
    print(P[P.emb == "eres2net"].round(3).to_string(index=False))
    print(Wr[Wr.emb == "eres2net"].groupby(["model", "set"]).apply(lambda x: f"argmax=0 in {int((x.argmax_w1 == 0).sum())}/8; "
                                                                      f"argmax {sorted(x.argmax_w1.tolist())}").to_string())


if __name__ == "__main__":
    main()
