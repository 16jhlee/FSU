"""v9 §1.2 / §2 audio cross-domain: TangoFlux (declare-lab), AudioCaps test captions (data/raw/dom/audiocaps_full).
python scripts/v9_audio.py gen <grid> [seeds=0] [n_cap=all]      python scripts/v9_audio.py dump [n_cap=200]
Rules (t = 1 - sigma at each step's start; TangoFlux CFG: v = v_u + g (v_c - v_u), w = g - 1):
  baseline | R: no CFG for t < 0.1 | kynk: CFG only for t in [0.156, 0.781) (Kynkaenniemi EDM sigma (0.28, 5.42];
  no audio-specific interval is published, same as v8 exploration and the image runner) | zi: CFG-Zero* zero-init (velocity 0 at step 1) | mambo<a>: g_k = 1+(g-1)exp(-a r)
Schedules: default sigma = linspace(1, 1/N, N) (official). Grid 'boundary' (§2) uses explicit first-step lengths:
  NFE 2: sigma = [1, 1 - t1] with t1 in {0.1, 0.2, 0.3}; NFE 3: [1, 1-t1, 1-t2] variants.
Outputs out/E9/audio/tangoflux/<cond>/s<k>/<id>.wav (+ _steps.json). 10 s, seed hash(id|audio|k).
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
CK = Path(r"C:\speechFM_store\ckpt\TangoFlux")
OUT = ROOT / "out" / "E9" / "audio" / "tangoflux"
CAPS = ROOT / "data/raw/dom/audiocaps_full/captions.tsv"


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) % (2**31 - 1)


def grid(name):
    G = {
        "main": [(g, n, r, None) for g in (4.5, 7.0) for n in (2, 3, 4, 6, 8, 25) for r in ("baseline", "R")]
        + [(g, n, "kynk", None) for g in (4.5, 7.0) for n in (3, 4, 6, 8)]
        + [(g, n, r, None) for g in (4.5, 7.0) for n in (3, 4) for r in ("zi", "mambo2", "mambo4")],
        "core": [(g, n, r, None) for g in (4.5, 7.0) for n in (3, 4) for r in ("baseline", "R")],
        "teacher8": [(4.5, 25, "baseline", None)],
        "boundary": [(4.5, 2, r, [1.0, 1.0 - t1]) for t1 in (0.1, 0.2, 0.3) for r in ("baseline", "R")]
        + [(4.5, 3, r, s) for s in ([1.0, 0.8, 0.5], [1.0, 0.7, 0.35]) for r in ("baseline", "R")],
    }
    return G[name]


def load():
    sys.path.insert(0, str(CK))
    from diffusers import AutoencoderOobleck
    from model import TangoFlux
    from safetensors.torch import load_file
    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(str(CK / "vae.safetensors")))
    model = TangoFlux(json.loads((CK / "config.json").read_text()))
    model.load_state_dict(load_file(str(CK / "tangoflux.safetensors")), strict=False)
    return vae.to("cuda").eval(), model.to("cuda").eval()


def decide(rule, t):
    if rule == "R":
        return t >= 0.1
    if rule == "kynk":
        return 0.156 <= t < 0.781
    return True


@torch.no_grad()
def cond_inputs(model, caption):
    dh = model.encode_duration(torch.tensor([10], device="cuda")).repeat(2, 1, 1)
    ehs, mask = model.encode_text_classifier_free([caption], num_samples_per_prompt=1)
    me = mask.unsqueeze(-1).expand_as(ehs)
    pooled = torch.nanmean(torch.where(me, ehs, torch.tensor(float("nan"), device=ehs.device)), dim=1)
    return torch.cat([ehs, dh], dim=1), model.fc(pooled)


@torch.no_grad()
def sample(vae, model, caption, g, n, rule, seed, sigmas=None):
    from model import retrieve_timesteps
    ehs, pp = cond_inputs(model, caption)
    sched = model.noise_scheduler
    sig = np.array(sigmas if sigmas is not None else np.linspace(1.0, 1 / n, n))
    ts, _ = retrieve_timesteps(sched, n, "cuda", None, sig)
    torch.manual_seed(seed)
    lat = torch.randn(1, model.audio_seq_len, 64).to("cuda")
    txt_ids = torch.zeros(2, ehs.shape[1], 3, device="cuda")
    aid = torch.arange(model.audio_seq_len).unsqueeze(0).unsqueeze(-1).repeat(2, 1, 3).to("cuda")
    mask = []
    for i, t in enumerate(ts):
        tr = 1.0 - float(sched.sigmas[i])
        on = decide(rule, tr)
        out = model.transformer(hidden_states=torch.cat([lat] * 2), timestep=torch.tensor([t / 1000], device="cuda"), guidance=None,
                                pooled_projections=pp, encoder_hidden_states=ehs, txt_ids=txt_ids, img_ids=aid, return_dict=False)[0]
        nu, nt = out.chunk(2)
        gk = g
        if rule.startswith("mambo"):
            r = float(torch.norm(nt - nu) / torch.norm(nu))
            gk = 1 + (g - 1) * math.exp(-float(rule[5:]) * r)
        v = nu + gk * (nt - nu) if on else nt
        if rule == "zi" and i == 0:
            v = v * 0.0
        mask.append(0 if (rule == "zi" and i == 0) else int(on))
        lat = sched.step(v, t, lat).prev_sample
    wave = vae.decode(lat.transpose(2, 1)).sample.cpu()[0][:, : int(10 * vae.config.sampling_rate)]
    return wave, "".join(map(str, mask)), [round(1 - float(s), 4) for s in sched.sigmas[:-1]]


def gen(grid_name, seeds, n_cap):
    vae, model = load()
    caps = pd.read_csv(CAPS, sep="\t")
    caps = caps.head(n_cap) if n_cap else caps
    for g, n, rule, sig in grid(grid_name):
        cond = f"g{g}_n{n}_{rule}" + ("" if sig is None else "_t" + "-".join(f"{1 - s:.2f}" for s in sig[1:]))
        for k in seeds:
            od = OUT / cond / f"s{k}"
            od.mkdir(parents=True, exist_ok=True)
            t0, info = time.time(), None
            for _, c in caps.iterrows():
                p = od / f"{c.id}.wav"
                if p.exists():
                    continue
                wave, mask, tgrid = sample(vae, model, c.caption, g, n, rule, h(f"{c.id}|audio|{k}"), sig)
                sf.write(str(p), wave.T.float().numpy(), vae.config.sampling_rate)
                info = dict(mask=mask, t=tgrid, g=g, n=n, rule=rule)
            if info:
                (od / "_steps.json").write_text(json.dumps(info))
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] tangoflux {cond} s{k} done {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def dump(n_cap):
    _, model = load()
    rows = []
    for _, c in pd.read_csv(CAPS, sep="\t").head(n_cap).iterrows():
        ehs, pp = cond_inputs(model, c.caption)
        torch.manual_seed(h(f"{c.id}|audio|0"))
        lat = torch.randn(1, model.audio_seq_len, 64).to("cuda")
        txt_ids = torch.zeros(2, ehs.shape[1], 3, device="cuda")
        aid = torch.arange(model.audio_seq_len).unsqueeze(0).unsqueeze(-1).repeat(2, 1, 3).to("cuda")
        out = model.transformer(hidden_states=torch.cat([lat] * 2), timestep=torch.tensor([1.0], device="cuda"), guidance=None,
                                pooled_projections=pp, encoder_hidden_states=ehs, txt_ids=txt_ids, img_ids=aid, return_dict=False)[0]
        nu, nt = out.float().chunk(2)
        rows.append(dict(id=c.id, rms=float((nt - nu).pow(2).mean().sqrt()), rms_u=float(nu.pow(2).mean().sqrt())))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "results/E9_audio_g0_tangoflux.csv", index=False)
    print(D[["rms", "rms_u"]].mean().round(4).to_dict())


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0]
        n_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        gen(sys.argv[2], seeds, n_cap)
    else:
        dump(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
