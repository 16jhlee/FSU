"""v3 M-1 check: seed dependence of the 1-step output.
For en 20 utterances x 8 seeds:
  V_seed(NFE1) (L2-normalised embedding cloud variance, 3 main embedders) vs V_noise (pilot perturbation floor)
  and vs V_seed at the reference NFE (pilot), and seed-to-seed log-mel RMS distance (dB) at NFE1 vs reference.
M-1 predicts V_seed(NFE1) ~ V_noise and log-mel seed distance ~ 0."""
import itertools
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import EMBEDDERS, load16, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EMBS = ["wavlm", "ecapa", "eres2net"]
NFE1 = {"A_cosyvoice2": "out/E3b/m1/A/cosyvoice2/nfe1/cfg0.7/utt{k}/en",
        "B_zipvoice": "out/E3b/m1/B/zipvoice/nfe1/cfg1.0/utt{k}/en",
        "B_zipvoice_distill": "out/E3b/m1/B/zipvoice_distill/nfe1/cfg3.0/utt{k}/en"}
REF = {"A_cosyvoice2": ("A_cosyvoice2_nfe10", "out/E3/pilot/A/cosyvoice2/nfe10/cfg0.7/utt{k}/en"),
       "B_zipvoice": ("B_zipvoice_nfe16", "out/E3/pilot/B/zipvoice/nfe16/cfg1.0/utt{k}/en"),
       "B_zipvoice_distill": ("B_zipvoice_nfe16", "out/E3/pilot/B/zipvoice/nfe16/cfg1.0/utt{k}/en")}


def cvar(E):
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    return len(U) / (len(U) - 1) * np.mean(np.sum((U - U.mean(0)) ** 2, axis=1))


def logmel(p):
    w, sr = sf.read(p, dtype="float32")
    return librosa.power_to_db(librosa.feature.melspectrogram(y=w, sr=sr, n_fft=1024, hop_length=256, n_mels=80,
                                                              fmax=8000) + 1e-10)


def mel_seed_dist(pat, u):
    M = [logmel(ROOT / pat.format(k=k) / f"{u}.wav") for k in range(8)]
    n = min(m.shape[1] for m in M)
    return np.mean([np.sqrt(np.mean((M[a][:, :n] - M[b][:, :n]) ** 2)) for a, b in itertools.combinations(range(8), 2)])


def main():
    utts = [r["utt"] for r in read_meta(ROOT / "data/lists/en_20.meta.lst")]
    pilot = pd.read_csv(ROOT / "results/E3_pilot.csv")
    pe = np.load(ROOT / "results/E3_pilot_emb.npz")
    recs = []
    for e in EMBS:
        emb = EMBEDDERS[e]()
        for m, pat in NFE1.items():
            refm, refpat = REF[m]
            for u in utts:
                E1 = np.stack([emb(load16(str(ROOT / pat.format(k=k) / f"{u}.wav"))) for k in range(8)])
                pr = pilot[(pilot.emb == e) & (pilot.model == refm) & (pilot.utt == u)].iloc[0]
                recs.append(dict(model=m, emb=e, utt=u, V_nfe1=cvar(E1), V_ref=cvar(pe[f"{e}/{refm}/{u}"]),
                                 V_noise=pr.V_noise))
        del emb
    D = pd.DataFrame(recs)
    mel = []
    for m, pat in NFE1.items():
        for u in utts:
            mel.append(dict(model=m, utt=u, mel_nfe1=mel_seed_dist(pat, u), mel_ref=mel_seed_dist(REF[m][1], u)))
    Mdf = pd.DataFrame(mel)
    D.to_csv(ROOT / "results/E3b_m1.csv", index=False)
    Mdf.to_csv(ROOT / "results/E3b_m1_mel.csv", index=False)
    S = D.groupby(["model", "emb"]).agg(V_nfe1=("V_nfe1", "median"), V_ref=("V_ref", "median"),
                                        V_noise=("V_noise", "median"))
    S["nfe1/noise"] = S.V_nfe1 / S.V_noise
    S["nfe1/ref"] = S.V_nfe1 / S.V_ref
    pd.set_option("display.width", 200)
    print(S.round(5).to_string())
    print(Mdf.groupby("model")[["mel_nfe1", "mel_ref"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
