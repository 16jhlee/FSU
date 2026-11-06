"""v9 §1.2 audio cross-domain: Stable Audio Open 1.0 (diffusers StableAudioPipeline, official sampler =
CosineDPMSolverMultistep: SDE-DPMSolver++ 2M, v-prediction, exponential sigma 500 -> 0.3, Brownian noise seeded per sample).
python scripts/v9_sao.py gen <grid> [seeds=0] [n_cap=all]      python scripts/v9_sao.py dump [n_cap=200]
Time map (pre-registered): repo time t = 1/(1+sigma) (rectified-flow time with the same SNR; also used for the Kynk
interval on images). Rules decided at each step's start sigma:
  baseline | R: no CFG for t < 0.1 (sigma > 9) | kynk: CFG only for sigma in (0.28, 5.42] (EDM interval, used as-is)
  zi: CFG-Zero* zero-init analogue - step 1 denoised = x/(1+sigma_0), i.e. zero rectified-flow velocity
  mambo<a>: g_k = 1+(g-1)exp(-a ||v_c-v_u||/||v_u||) on the model (v) output
g: official 7 and 1.5x (10.5). CFG as in the pipeline: v = v_u + g (v_c - v_u), uncond = zero text+duration embeddings.
Outputs out/E9/audio/sao/<cond>/s<k>/<id>.wav (44.1 kHz stereo, 10 s) + _steps.json. Seed hash(id|audio|k).
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
CK = Path(r"C:\speechFM_store\ckpt\sao")
OUT = ROOT / "out" / "E9" / "audio" / "sao"
CAPS = ROOT / "data/raw/dom/audiocaps_full/captions.tsv"
DUR = 10.0


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) % (2**31 - 1)


def grid(name):
    G = {
        "main": [(g, n, r) for g in (7.0, 10.5) for n in (2, 3, 4, 6, 8, 25) for r in ("baseline", "R")]
        + [(g, n, "kynk") for g in (7.0, 10.5) for n in (3, 4, 6, 8)]
        + [(g, n, r) for g in (7.0, 10.5) for n in (3, 4) for r in ("zi", "mambo2", "mambo4")],
        "core": [(g, n, r) for g in (7.0, 10.5) for n in (3, 4) for r in ("baseline", "R")],
        "teacher8": [(7.0, 25, "baseline")],
        "smoke": [(7.0, 4, r) for r in ("baseline", "R", "kynk", "zi", "mambo2")] + [(7.0, 100, "baseline")],
    }
    return G[name]


def decide(rule, sigma):
    if rule == "R":
        return 1.0 / (1.0 + sigma) >= 0.1
    if rule == "kynk":
        return 0.28 < sigma <= 5.42
    return True


def load():
    from diffusers import StableAudioPipeline
    pipe = StableAudioPipeline.from_pretrained(str(CK), torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def cond_embeds(pipe, caption):
    pe = pipe.encode_prompt(caption, "cuda", True, None, None, None, None, None)
    ss, se = pipe.encode_duration(0.0, DUR, "cuda", False, 1)
    tad = torch.cat([pe, ss, se], dim=1)
    ad = torch.cat([ss, se], dim=2)
    return torch.cat([torch.zeros_like(tad), tad]), torch.cat([ad, ad])


@torch.no_grad()
def sample(pipe, caption, g, n, rule, seed):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed
    tad, ad = cond_embeds(pipe, caption)
    sched = pipe.scheduler
    sched.set_timesteps(n, device="cuda")
    gen_ = torch.Generator("cuda").manual_seed(seed)
    lat = pipe.prepare_latents(1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size), tad.dtype,
                               torch.device("cuda"), gen_, None, None, 1, audio_channels=pipe.vae.config.audio_channels)
    rot = get_1d_rotary_pos_embed(pipe.rotary_embed_dim, lat.shape[2] + ad.shape[1], use_real=True, repeat_interleave_real=False)
    mask = []
    for i, t in enumerate(sched.timesteps):
        sigma = float(sched.sigmas[i])
        on = decide(rule, sigma)
        inp = sched.scale_model_input(torch.cat([lat] * 2), t)
        out = pipe.transformer(inp, t.unsqueeze(0), encoder_hidden_states=tad, global_hidden_states=ad, rotary_embedding=rot,
                               return_dict=False)[0]
        nu, nt = out.chunk(2)
        gk = g
        if rule.startswith("mambo"):
            r = float(torch.norm((nt - nu).float()) / torch.norm(nu.float()))
            gk = 1 + (g - 1) * math.exp(-float(rule[5:]) * r)
        v = nu + gk * (nt - nu) if on else nt
        if rule == "zi" and i == 0:  # denoised = c_skip x + c_out v := x/(1+sigma)
            c_skip = 1.0 / (sigma**2 + 1.0)
            c_out = -sigma / (sigma**2 + 1.0) ** 0.5
            v = ((lat.float() / (1.0 + sigma) - c_skip * lat.float()) / c_out).to(v.dtype)
        mask.append(0 if (rule == "zi" and i == 0) else int(on))
        lat = sched.step(v, t, lat, generator=gen_).prev_sample
    audio = pipe.vae.decode(lat).sample[:, :, : int(DUR * pipe.vae.config.sampling_rate)]
    return audio[0].float().cpu(), "".join(map(str, mask)), [round(1 / (1 + float(s)), 4) for s in sched.sigmas[:-1]]


def gen(grid_name, seeds, n_cap):
    pipe = load()
    caps = pd.read_csv(CAPS, sep="\t")
    caps = caps.head(n_cap) if n_cap else caps
    sr = pipe.vae.config.sampling_rate
    for g, n, rule in grid(grid_name):
        cond = f"g{g}_n{n}_{rule}"
        for k in seeds:
            od = OUT / cond / f"s{k}"
            od.mkdir(parents=True, exist_ok=True)
            t0, info = time.time(), None
            for _, c in caps.iterrows():
                p = od / f"{c.id}.wav"
                if p.exists():
                    continue
                wave, mask, tgrid = sample(pipe, c.caption, g, n, rule, h(f"{c.id}|audio|{k}"))
                sf.write(str(p), wave.T.numpy(), sr)
                info = dict(mask=mask, t=tgrid, g=g, n=n, rule=rule)
            if info:
                (od / "_steps.json").write_text(json.dumps(info))
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] sao {cond} s{k} done {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def dump(n_cap):
    """t=0 guidance size: RMS(v_c - v_u) of the model output at the first sigma (500), seed 0 noise."""
    pipe = load()
    rows = []
    for _, c in pd.read_csv(CAPS, sep="\t").head(n_cap).iterrows():
        tad, ad = cond_embeds(pipe, c.caption)
        from diffusers.models.embeddings import get_1d_rotary_pos_embed
        sched = pipe.scheduler
        sched.set_timesteps(8, device="cuda")
        gen_ = torch.Generator("cuda").manual_seed(h(f"{c.id}|audio|0"))
        lat = pipe.prepare_latents(1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size), tad.dtype,
                                   torch.device("cuda"), gen_, None, None, 1, audio_channels=pipe.vae.config.audio_channels)
        rot = get_1d_rotary_pos_embed(pipe.rotary_embed_dim, lat.shape[2] + ad.shape[1], use_real=True, repeat_interleave_real=False)
        t = sched.timesteps[0]
        out = pipe.transformer(sched.scale_model_input(torch.cat([lat] * 2), t), t.unsqueeze(0), encoder_hidden_states=tad,
                               global_hidden_states=ad, rotary_embedding=rot, return_dict=False)[0]
        nu, nt = out.float().chunk(2)
        rows.append(dict(id=c.id, rms=float((nt - nu).pow(2).mean().sqrt()), rms_u=float(nu.pow(2).mean().sqrt())))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E9_audio_g0_sao.csv", index=False)
    print(D[["rms", "rms_u"]].mean().round(4).to_dict())


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
        n_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        gen(sys.argv[2], seeds, n_cap)
    else:
        dump(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
