"""E12-1 (v10.4): where does the displacement go? CosyVoice2, en_100, seeds 0/1, mel space (generated region, 80 x L).
Per utterance u, bases from teacher outputs (official NFE 10, cosine schedule, w = 0.7), each the mean over noise seeds
k = 0, 1 with the generated-region noise ALIGNED across variants (prompt-region noise drawn separately, generated-region
noise = hash(u|noise|k) for every variant, so variant differences are not noise-layout differences):
  speaker basis d_s : same tokens of u, prompts (prompt_token / prompt_feat / embedding) of u + 7 other en_100 utterances
                      (8 speakers); top-3 principal directions of the 8 centred outputs.
  content basis d_tau: prompt of u, tokens of u + 3 other en_100 sentences, resized to len(tokens_u) (truncated or tiled:
                      the generated region must keep its length for a frame-wise basis; deviation from the instruction's
                      "same speaker's other sentences" — Seed-TTS-eval has one sentence per prompt); top-3 directions;
                      Gram-Schmidt-orthogonalised against d_s.
Displacements decomposed (energy shares in d_s, d_tau, residual):
  (a) first step t1 w g_b, b = all-null / token-null / speaker-null (g_b = v_c - v_u(b) at t = 0, t1 = 1 - cos(pi/4),
      default noise layout as run_cosy);
  (b) final output: baseline NFE 2 - R NFE 2 (same noise; the waveforms of e12_jobs c1_*);
  controls: random Gaussian; teacher seed difference (k0 - k1, own prompt); held-out speaker (9th prompt) - own;
      held-out content (5th token sequence) - own.
  (c) artifact metrics of the final mels (E3b definitions): hf share (>= 4 kHz), frame flux, both for baseline and R.
python scripts/e12_direction.py [n_utt=100]   (env_cosy, GPU)  -> results/E12_direction.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v8_theory_cosy import Runner  # noqa: E402
from run_cosy import h, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T10 = list(1 - np.cos(np.pi / 2 * np.arange(11) / 10))
T2 = list(1 - np.cos(np.pi / 2 * np.arange(3) / 2))
T1 = float(T2[1])
W = 0.7
CENT = None


def smooth(m):
    import librosa
    global CENT
    if CENT is None:
        CENT = librosa.mel_frequencies(n_mels=82, fmin=0, fmax=8000)[1:-1]
    e = np.exp(2 * m.astype(np.float64))
    hf = e[CENT >= 4000].sum() / e.sum()
    flux = np.var(np.linalg.norm(np.diff(m, axis=1), axis=0))
    return hf, flux


def run(R, seg, noise, ts, w, branch="all", iv=None):
    dec, m = R.dec, R.m
    dec.inference_cfg_rate, dec.t_span_override, dec.cfg_branch, dec.cfg_interval = w, ts, branch, iv
    dec.rand_noise = noise
    dec.dump_states = []
    with torch.inference_mode():
        m.flow.inference(token=seg["token"].to(m.device, dtype=torch.int32),
                         token_len=torch.tensor([seg["token"].shape[1]], dtype=torch.int32).to(m.device),
                         prompt_token=seg["prompt_token"].to(m.device),
                         prompt_token_len=torch.tensor([seg["prompt_token"].shape[1]], dtype=torch.int32).to(m.device),
                         prompt_feat=seg["prompt_feat"].to(m.device),
                         prompt_feat_len=torch.tensor([seg["prompt_feat"].shape[1]], dtype=torch.int32).to(m.device),
                         embedding=seg["embedding"].to(m.device), streaming=False, finalize=True)
    st, dec.dump_states = dec.dump_states, None
    P = seg["prompt_feat"].shape[1]
    cut = lambda a: a[0, :, P:].double().numpy()  # noqa: E731
    return dict(final=cut(st[-1]["x"] + (ts[-1] - ts[-2]) * st[-1]["v"]), vc=cut(st[0]["vc"]), vu=cut(st[0]["vu"]))


def default_noise(u, k):
    return torch.randn([1, 80, 50 * 300], generator=torch.Generator().manual_seed(h(f"{u}|noise|{k}")))


def aligned_noise(u, k, P):
    gp = torch.Generator().manual_seed(h(f"{u}|pnoise|{k}"))
    pn = torch.randn([1, 80, 2000], generator=gp)[:, :, :P]
    return torch.cat([pn, default_noise(u, k)], 2)[:, :, :50 * 300]


def resize_tokens(tok, n):
    reps = int(np.ceil(n / tok.shape[1]))
    return tok.repeat(1, reps)[:, :n]


def basis(V, k=3):
    X = np.stack([v.ravel() for v in V])
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt[:k].T  # D x k orthonormal


def shares(d, Qs, Qt):
    d = d.ravel()
    e = d @ d
    ps, pt = Qs.T @ d, Qt.T @ d
    es, et = float(ps @ ps), float(pt @ pt)
    return es / e, et / e, 1 - (es + et) / e, e / d.size


def main():
    n_utt = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    meta = read_meta(ROOT / "data/lists/en_100.meta.lst")
    utts = [r[0] for r in meta]
    R = Runner("cv2")
    segs = {u: torch.load(ROOT / R.cache / f"{u}.pt")[0] for u in utts}
    rng = np.random.default_rng(0)
    rows = []
    for i, u in enumerate(utts[:n_utt]):
        su = segs[u]
        spk = [u] + [utts[(i + j) % len(utts)] for j in range(1, 8)]
        spk_ho = utts[(i + 8) % len(utts)]
        con = [u] + [utts[(i + j) % len(utts)] for j in (10, 20, 30)]
        con_ho = utts[(i + 40) % len(utts)]
        n_tok = su["token"].shape[1]

        def teacher(prompt_of, token_of):
            sp = segs[prompt_of]
            seg = dict(token=su["token"] if token_of == u else resize_tokens(segs[token_of]["token"], n_tok),
                       prompt_token=sp["prompt_token"], prompt_feat=sp["prompt_feat"], embedding=sp["embedding"])
            P = sp["prompt_feat"].shape[1]
            outs = [run(R, seg, aligned_noise(u, k, P), T10, W)["final"] for k in (0, 1)]
            return outs
        Ts = {p: teacher(p, u) for p in spk + [spk_ho]}
        Tc = {c: teacher(u, c) for c in con[1:] + [con_ho]}
        Tc[u] = Ts[u]
        L = min(min(v[0].shape[1] for v in Ts.values()), min(v[0].shape[1] for v in Tc.values()))
        mean = lambda v: (v[0][:, :L] + v[1][:, :L]) / 2  # noqa: E731
        Qs = basis([mean(Ts[p]) for p in spk])
        Ct = basis([mean(Tc[c]) for c in con])
        Ct = Ct - Qs @ (Qs.T @ Ct)
        Qt, _ = np.linalg.qr(Ct)
        D = 80 * L
        ctrl = {"random": rng.standard_normal((80, L)), "seed_diff": Ts[u][0][:, :L] - Ts[u][1][:, :L],
                "heldout_speaker": mean(Ts[spk_ho]) - mean(Ts[u]), "heldout_content": mean(Tc[con_ho]) - mean(Ts[u])}
        for name, d in ctrl.items():
            fs, ft, fr, en = shares(d, Qs, Qt)
            rows.append(dict(utt=u, k=-1, kind="control", what=name, f_spk=fs, f_con=ft, f_res=fr, energy=en, L=L, D=D))
        for k in (0, 1):
            nz = default_noise(u, k)
            for b in ("all", "token", "speaker"):
                o = run(R, su, nz, [0.0, 1.0], W, b)
                d = T1 * W * (o["vc"] - o["vu"])[:, :L]
                fs, ft, fr, en = shares(d, Qs, Qt)
                rows.append(dict(utt=u, k=k, kind="first_step", what=f"{b}-null", f_spk=fs, f_con=ft, f_res=fr, energy=en,
                                 L=L, D=D))
            xb = run(R, su, nz, T2, W)["final"]
            xr = run(R, su, nz, T2, W, iv=[0.1, 1.0])["final"]
            fs, ft, fr, en = shares((xb - xr)[:, :L], Qs, Qt)
            hb, fb = smooth(xb)
            hr, fr_ = smooth(xr)
            pr = (xb - xr)[:, :L].ravel()
            res = pr - Qs @ (Qs.T @ pr) - Qt @ (Qt.T @ pr)
            rows.append(dict(utt=u, k=k, kind="final", what="baseline-R", f_spk=fs, f_con=ft, f_res=fr, energy=en, L=L, D=D,
                             res_energy=float(res @ res) / D, hf_base=hb, hf_R=hr, flux_base=fb, flux_R=fr_))
        print(i, u, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results/E12_direction.csv", index=False)
    D = pd.DataFrame(rows)
    print(D.groupby(["kind", "what"])[["f_spk", "f_con", "f_res", "energy"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
