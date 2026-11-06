"""E3b §2.2 (v3, with approved corrections): dump CosyVoice2 flow states for the support test (M-P2) and
mel-level smoothness (M-P1). env_cosy. Uses patches CosyVoice_sampler.diff + CosyVoice_dump_states.diff.

Repo time convention: z_t = (1-t) eps + t x (t=0 noise, t=1 data), Euler z' = z + dt v, x_hat(z,t) = z + (1-t) v.
Per utterance (en 100, cached LLM tokens, per-utterance noise seeds):
  NFE2 (official cosine schedule, t1 = 1-cos(pi/4) = 0.2929), seed 0:
     x0hat  = eps + v(eps, 0)              (first-step prediction = NFE1 output)
     z_e    = (1-t1) eps + t1 x0hat        (Euler intermediate state)
     nfe2   = z_e + (1-t1) v(z_e, t1)      (final NFE2 output = prediction at the intermediate state)
  NFE10 (official cosine; its 5th grid point is exactly t1), seeds 0..7:
     x10_k  = final mel;  z_ode = trajectory state at t1 (seed 0)
  On-path references (seed 0): z_lin = (1-t1) eps + t1 x10_0 (straight interpolation, as in v3),
     xhat_lin = z_lin + (1-t1) v(z_lin, t1),  xhat_ode = z_ode + (1-t1) v(z_ode, t1)
Saved (generated region only, before the vocoder): out/E3b/states/<utt>.npz
"""
import argparse
import math
import sys
import uuid as uuidlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_cosy  # noqa: E402  (pins wetext, imports CosyVoice)
from run_cosy import AutoModel, h, read_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T1 = 1 - math.cos(math.pi / 4)


def cosine(n):
    t = torch.linspace(0, 1, n + 1)
    return (1 - torch.cos(t * 0.5 * torch.pi)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/lists/en_100.meta.lst")
    ap.add_argument("--cache_dir", default="cache/cosy2_tokens/en")
    ap.add_argument("--out_dir", default="out/E3b/states")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=0.7, help="E5 extra: run with 0 to test the CFG-extrapolation account")
    a = ap.parse_args()
    cv = AutoModel(model_dir=str(ROOT / "ckpt" / "CosyVoice2-0.5B"))
    assert cv.frontend.text_frontend in ("ttsfrd", "wetext")
    m, dec = cv.model, cv.model.flow.decoder
    dec.inference_cfg_rate = a.cfg
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    skipped = []

    def run(seg, utt, k, t_span, dump):
        g = torch.Generator().manual_seed(h(f"{utt}|noise|{k}"))
        dec.rand_noise = torch.randn([1, 80, 50 * 300], generator=g)
        dec.t_span_override = t_span
        dec.dump_states = [] if dump else None
        with torch.inference_mode():
            feat, _ = m.flow.inference(token=seg["token"].to(m.device, dtype=torch.int32),
                                       token_len=torch.tensor([seg["token"].shape[1]], dtype=torch.int32).to(m.device),
                                       prompt_token=seg["prompt_token"].to(m.device),
                                       prompt_token_len=torch.tensor([seg["prompt_token"].shape[1]], dtype=torch.int32).to(m.device),
                                       prompt_feat=seg["prompt_feat"].to(m.device),
                                       prompt_feat_len=torch.tensor([seg["prompt_feat"].shape[1]], dtype=torch.int32).to(m.device),
                                       embedding=seg["embedding"].to(m.device), streaming=False, finalize=True)
        st = dec.dump_states
        dec.dump_states = None
        return feat[0].float().cpu().numpy(), st, (dec.dump_args if dump else None)

    def one_step(z_full, args):
        """Euler step from t1 to 1 on a full-length state, with the same conditions (= x_hat at t1)."""
        dec.dump_states = None
        with torch.inference_mode():
            x = dec.solve_euler(z_full.to(args["mu"].device, args["mu"].dtype),
                                t_span=torch.tensor([T1, 1.0], device=args["mu"].device, dtype=args["mu"].dtype),
                                mu=args["mu"], mask=args["mask"], spks=args["spks"], cond=args["cond"],
                                streaming=args["streaming"])
        return x[0].float().cpu().numpy()

    for utt, *_ in read_meta(ROOT / a.meta):
        f = out / f"{utt}.npz"
        if f.exists():
            continue
        segs = torch.load(ROOT / a.cache_dir / f"{utt}.pt")
        if len(segs) != 1:
            skipped.append(utt)
            continue
        seg = segs[0]
        P = seg["prompt_feat"].shape[1]  # prompt frames are excluded from all saved mels
        nfe2, st2, _ = run(seg, utt, 0, [0.0, T1, 1.0], True)
        assert abs(st2[1]["t"] - T1) < 1e-4, st2[1]["t"]
        eps, v0 = st2[0]["x"][0], st2[0]["v"][0]
        x0hat = (eps + v0).numpy()
        z_e = st2[1]["x"][0].numpy()
        x10, st10, args10 = run(seg, utt, 0, cosine(10), True)
        assert abs(st10[5]["t"] - T1) < 1e-4, st10[5]["t"]
        z_ode = st10[5]["x"]
        x10_full = (st10[-1]["x"] + (1 - st10[-1]["t"]) * st10[-1]["v"])[0]
        z_lin = (1 - T1) * st10[0]["x"] + T1 * x10_full.unsqueeze(0)
        xhat_lin = one_step(z_lin, args10)
        xhat_ode = one_step(z_ode, args10)
        x10_k = [x10] + [run(seg, utt, k, cosine(10), False)[0] for k in range(1, a.K)]
        np.savez(f, P=P, t1=T1, x0hat=x0hat[:, P:], z_e=z_e[:, P:], nfe2=nfe2, x10=np.stack(x10_k),
                 z_ode=z_ode[0].numpy()[:, P:], z_lin=z_lin[0].numpy()[:, P:], eps=eps.numpy()[:, P:],
                 xhat_lin=xhat_lin[:, P:], xhat_ode=xhat_ode[:, P:], x10_full_check=x10_full.numpy()[:, P:])
        print(utt, nfe2.shape, flush=True)
    print("skipped (multi-segment):", skipped)


if __name__ == "__main__":
    main()
