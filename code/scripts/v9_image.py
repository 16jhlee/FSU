"""v9 §1.1 image cross-domain main experiment (SD3-medium, SD3.5-medium), COCO Karpathy test captions.
python scripts/v9_image.py embed <model>                         # text embeddings -> cache (text encoders, once)
python scripts/v9_image.py gen <model> <grid> [seeds=0] [n_cap=1000]
python scripts/v9_image.py dump <model> [n_cap=200]              # t=0 ||v_c - v_u|| per caption (E1 law, D3)
python scripts/v9_image.py bench <model>                         # s/step at 1024 px (sizing)
Step rules (repo time t = 1 - sigma, decided at each step's start; g = official-style guidance scale, w = g - 1):
  baseline: CFG every step | R: no CFG for t < 0.1 [R (ours)] | kynk: CFG only for t in [0.156, 0.781)
  (Kynkaenniemi EDM sigma (0.28, 5.42] -> t = 1/(1+sigma)) | zi: CFG-Zero* zero-init (velocity 0 at step 1)
  mambo<a>: MAMBO-G  g_k = 1 + (g-1) exp(-a ||v_c - v_u|| / ||v_u||)  (Diffusers PR #12862 formula, per sample)
Same caption -> same noise (seed hash(id|image|k)) for every rule / NFE / g. 1024 px, fp16, transformer + VAE resident on
GPU (text embeddings precomputed). Outputs out/E9/image/<model>/<cond>/s<k>/<id>.png (+ _steps.json).
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
CK = {"sd3": Path(r"C:\speechFM_store\ckpt\sd3-medium"), "sd35": Path(r"C:\speechFM_store\ckpt\sd35-medium")}
OUT = ROOT / "out" / "E9" / "image"
EMB = ROOT / "cache" / "e9_image"
CAPS = ROOT / "data/raw/dom/coco1k/captions.tsv"
KYNK = (0.156, 0.781)
BATCH = 2  # captions per forward (x2 for CFG); set from bench


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) % (2**31 - 1)


def captions(n):
    return pd.read_csv(CAPS, sep="\t").head(n)


def grid(name):
    G = {  # name -> list of (g, nfe, rule)
        "main": [(g, n, r) for g in (5.0, 7.0, 10.0) for n in (4, 6, 8) for r in ("baseline", "R", "kynk")]
        + [(g, n, r) for g in (5.0, 7.0, 10.0) for n in (4, 8) for r in ("zi", "mambo2", "mambo4")],
        "teacher": [(g, 28, r) for g in (5.0, 7.0, 10.0) for r in ("baseline", "R", "kynk")],
        "core": [(g, n, r) for g in (7.0, 10.0) for n in (4, 8) for r in ("baseline", "R")],
        "teacher8": [(7.0, 28, "baseline")],
        "shift": [(7.0, 4, r) for r in ("baseline", "R")],
        # v9 budget plan (0.27 s per caption-step on the 16 GB GPU; batching gives no speed-up), run in this order:
        "main1": [(g, n, r) for g in (5.0, 7.0, 10.0) for n in (4, 6, 8) for r in ("baseline", "R", "kynk")],  # 1k
        "alt7": [(7.0, n, r) for n in (4, 8) for r in ("zi", "mambo2", "mambo4")],  # 1k
        "teacher7": [(7.0, 28, r) for r in ("baseline", "R", "kynk")],  # 1k
        "teacher_g": [(g, 28, r) for g in (5.0, 10.0) for r in ("baseline", "R")],  # 300
        "alt_g": [(g, n, r) for g in (5.0, 10.0) for n in (4, 8) for r in ("zi", "mambo2", "mambo4")],  # 300
        "core7": [(7.0, n, r) for n in (4, 8) for r in ("baseline", "R")],  # diversity: 8 seeds x 200
        # E8'-10 (v9.1 §2): 2026 baselines on SD3, NFE 4/8 x g 7/10, COCO 1k
        "e8p10": [(g, n, r) for g in (7.0, 10.0) for n in (4, 8) for r in ("tfr", "tvs", "tvu")],
    }
    return G[name]


def load(model, encoders=False):
    from diffusers import StableDiffusion3Pipeline
    kw = {} if encoders else dict(text_encoder=None, text_encoder_2=None, text_encoder_3=None, tokenizer=None,
                                  tokenizer_2=None, tokenizer_3=None)
    pipe = StableDiffusion3Pipeline.from_pretrained(str(CK[model]), torch_dtype=torch.float16, **kw)
    return pipe


def embed(model, n=5000):
    pipe = load(model, encoders=True)
    pipe.enable_model_cpu_offload()
    caps = captions(n)
    (EMB / model).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for _, c in caps.iterrows():
            f = EMB / model / f"{c.id}.pt"
            if f.exists():
                continue
            pe, ne, pp, npool = pipe.encode_prompt(prompt=c.caption, prompt_2=None, prompt_3=None, device="cuda",
                                                   do_classifier_free_guidance=True)
            torch.save(dict(pe=pe.cpu(), ne=ne.cpu(), pp=pp.cpu(), npool=npool.cpu()), f)
    print("embeddings done", flush=True)


def tv_scales(sigmas, g, kind):
    """E8'-10 / v9.1 §1.1: sqt24/tvcfg guidance_schedulers.py transcribed (see run_cosy.tv_scales). sigmas = step-start
    sigmas (= scheduler timesteps / max). delta_i = sigma_i - sigma_{i+1}, final 0.0 appended (official).
    symmetric(low=1, high=2g-1, normalize) | stepup(low=g/3, high=g, tau=0.25, no normalize) |
    interval(value=g, left=0.15, right=0.78, normalize): g' inside (left, right), 1.0 outside, weighted mean = g."""
    sig = np.asarray(sigmas, dtype=float)
    n = len(sig)
    delta = np.append(sig[:-1] - sig[1:], 0.0)
    if kind == "symmetric":
        low, high, half = 1.0, 2 * g - 1.0, n // 2
        if n % 2 == 0:
            s = np.concatenate([np.linspace(low, high, half, endpoint=False), np.linspace(high, low, half, endpoint=True)])
        else:
            s = np.concatenate([np.linspace(low, high, half + 1, endpoint=True), np.linspace(high, low, half + 1, endpoint=True)[1:]])
        return s * g / np.sum(s * delta)
    if kind == "stepup":
        return np.where(np.linspace(0, 1, n) < 0.25, g / 3, g)
    if kind == "interval":
        inside = (sig > 0.15) & (sig < 0.78)
        val = (g - np.sum(delta[~inside])) / np.sum(delta[inside]) if inside.any() else g
        return np.where(inside, val, 1.0)
    raise ValueError(kind)


TV_RULES = {"tvs": "symmetric", "tvu": "stepup", "intv": "interval", "Rtv": "symmetric"}


def decide(rule, i, t):
    if rule in ("baseline", "zi", "tfr", "tvs", "tvu", "intv") or rule.startswith("mambo"):
        return True
    if rule in ("R", "Rtv"):
        return t >= 0.1
    if rule == "kynk":
        return KYNK[0] <= t < KYNK[1]
    raise ValueError(rule)


def sample(pipe, embs, g, n, rule, seeds, shift=None):
    """Batched over captions: embs / seeds are lists (B). Per-sample noise from each seed (identical to B=1)."""
    tr, sched = pipe.transformer, pipe.scheduler
    if shift is not None:
        sched = sched.__class__.from_config(sched.config, shift=shift)
    B = len(embs)
    cat = lambda key: torch.cat([e[key] for e in embs]).to("cuda")  # noqa: E731
    pe, ne, pp, npool = cat("pe"), cat("ne"), cat("pp"), cat("npool")
    lat = torch.cat([torch.randn((1, tr.config.in_channels, 128, 128), generator=torch.Generator("cpu").manual_seed(s))
                     for s in seeds]).to("cuda", torch.float16)
    sched.set_timesteps(n, device="cuda")
    tvs = tv_scales([float(s) for s in sched.sigmas[:n]], g, TV_RULES[rule]) if rule in TV_RULES else None
    mask = []
    for i, t in enumerate(sched.timesteps):
        trp = 1.0 - float(sched.sigmas[i])
        on = decide(rule, i, trp)
        out = tr(hidden_states=torch.cat([lat] * 2), timestep=t.expand(2 * B), encoder_hidden_states=torch.cat([ne, pe]),
                 pooled_projections=torch.cat([npool, pp]), return_dict=False)[0]
        nu, nt = out.chunk(2)
        if rule.startswith("mambo"):
            a = float(rule[5:])
            r = (torch.norm((nt - nu).float().flatten(1), dim=1) / torch.norm(nu.float().flatten(1), dim=1))
            gk = (1 + (g - 1) * torch.exp(-a * r)).view(-1, 1, 1, 1).to(nu.dtype)
        else:
            gk = float(tvs[i]) if tvs is not None else g
        npred = nu + gk * (nt - nu) if on else nt
        if rule == "tfr":
            # E8'-10 TFR (2607.07665 Eq.15) on the straight path x = (1-s) x0 + s eps (t = 1 - s): D = x - s v,
            # x_{n+1} = a x + (1-a) D_c + t1 (r^g - r)(D_u - D_c), a = s1/s0, r = a t0/t1 (0 at s0 = 1 or s1 = 0); w = g - 1
            s0, s1 = float(sched.sigmas[i]), float(sched.sigmas[i + 1])
            t0, t1 = 1 - s0, 1 - s1
            a = s1 / s0
            r = a * t0 / t1 if t1 > 0 else 0.0
            lf = lat.float()
            Dc, Du = lf - s0 * nt.float(), lf - s0 * nu.float()
            x_next = a * lf + (1 - a) * Dc + t1 * (r ** g - r) * (Du - Dc)
            npred = ((x_next - lf) / (s1 - s0)).to(nt.dtype)  # FlowMatchEuler step: x + (s1 - s0) v
        if rule == "zi" and i == 0:
            npred = npred * 0.0
        mask.append(int(on) if rule != "zi" or i else 0)
        lat = sched.step(npred, t, lat, return_dict=False)[0]
    vae = pipe.vae
    imgs = []
    for b in range(B):
        x = (lat[b:b + 1] / vae.config.scaling_factor) + vae.config.shift_factor
        imgs.append(pipe.image_processor.postprocess(vae.decode(x, return_dict=False)[0], output_type="pil")[0])
    return imgs, "".join(map(str, mask))


def gen(model, grid_name, seeds, n_cap, shift=None):
    pipe = load(model)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    caps = captions(n_cap)
    for g, n, rule in grid(grid_name):
        cond = f"g{g}_n{n}_{rule}" + (f"_shift{shift}" if shift is not None else "")
        for k in seeds:
            od = OUT / model / cond / f"s{k}"
            od.mkdir(parents=True, exist_ok=True)
            mask = None
            t0 = time.time()
            todo = [c for _, c in caps.iterrows() if not (od / f"{c.id}.png").exists()]
            for j in range(0, len(todo), BATCH):
                chunk = todo[j:j + BATCH]
                embs = [torch.load(EMB / model / f"{c.id}.pt") for c in chunk]
                with torch.no_grad():
                    imgs, mask = sample(pipe, embs, g, n, rule, [h(f"{c.id}|image|{k}") for c in chunk], shift)
                for c, img in zip(chunk, imgs):
                    img.save(od / f"{c.id}.png")
            if mask:
                (od / "_steps.json").write_text(json.dumps(dict(mask=mask, g=g, n=n, rule=rule, shift=shift)))
            print(f"[{time.strftime('%m-%d %H:%M:%S')}] {model} {cond} s{k} done {time.time() - t0:.0f}s", flush=True)


def dump(model, n_cap):
    """t=0 guidance size per caption: RMS(v_c - v_u) on the 128x128x16 latent at sigma=1 (seed 0 noise)."""
    pipe = load(model)
    pipe.transformer.to("cuda")
    tr, sched = pipe.transformer, pipe.scheduler
    rows = []
    for _, c in captions(n_cap).iterrows():
        emb = torch.load(EMB / model / f"{c.id}.pt")
        pe, ne, pp, npool = (emb[k].to("cuda") for k in ("pe", "ne", "pp", "npool"))
        gen_ = torch.Generator("cpu").manual_seed(h(f"{c.id}|image|0"))
        lat = torch.randn((1, tr.config.in_channels, 128, 128), generator=gen_).to("cuda", torch.float16)
        sched.set_timesteps(28, device="cuda")
        t = sched.timesteps[0]
        with torch.no_grad():
            out = tr(hidden_states=torch.cat([lat] * 2), timestep=t.expand(2), encoder_hidden_states=torch.cat([ne, pe]),
                     pooled_projections=torch.cat([npool, pp]), return_dict=False)[0]
        nu, nt = out.float().chunk(2)
        rows.append(dict(id=c.id, rms=float((nt - nu).pow(2).mean().sqrt()), rms_u=float(nu.pow(2).mean().sqrt())))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / f"results/E9_image_g0_{model}.csv", index=False)
    print(D[["rms", "rms_u"]].mean().round(4).to_dict())


def bench(model):
    pipe = load(model)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    caps = captions(4)
    embs = [torch.load(EMB / model / f"{c.id}.pt") for _, c in caps.iterrows()]
    with torch.no_grad():
        for B in (1, 2, 4):
            sample(pipe, embs[:B], 7.0, 4, "baseline", list(range(B)))
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            sample(pipe, embs[:B], 7.0, 8, "baseline", list(range(B)))
            torch.cuda.synchronize()
            print(f"{model} B={B}: {(time.time() - t0) / 8 / B:.3f} s per caption-step, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
                  flush=True)


if __name__ == "__main__":
    cmd, model = sys.argv[1], sys.argv[2]
    if cmd == "embed":
        embed(model, int(sys.argv[3]) if len(sys.argv) > 3 else 5000)
    elif cmd == "gen":
        seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
        n_cap = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
        shift = float(sys.argv[6]) if len(sys.argv) > 6 else None
        gen(model, sys.argv[3], seeds, n_cap, shift)
    elif cmd == "dump":
        dump(model, int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif cmd == "bench":
        bench(model)
