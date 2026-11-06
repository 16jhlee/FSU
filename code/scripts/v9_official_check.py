"""Fairness check (2026-10-27): our step-wise runners with rule = baseline must reproduce the OFFICIAL samplers exactly.
python scripts/v9_official_check.py sd3|sd35|tangoflux|sao
  sd3/sd35  : diffusers StableDiffusion3Pipeline.__call__ (same prompt embeddings, same initial latents) vs v9_image.sample
  tangoflux : official TangoFlux.inference_flow (ckpt model.py, same torch.manual_seed) vs v9_audio.sample (latents compared)
  sao       : diffusers StableAudioPipeline.__call__ (same generator seed) vs v9_sao.sample
Writes results/E9_official_check_<model>.csv (max / mean absolute difference per case). Pass = outputs identical up to
floating-point noise; otherwise the affected baselines are regenerated with the official sampler.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def check_image(model):
    import v9_image as V
    pipe = V.load(model)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rows = []
    for _, c in V.captions(3).iterrows():
        e = torch.load(V.EMB / model / f"{c.id}.pt")
        seed = V.h(f"{c.id}|image|0")
        for g, n in [(7.0, 4), (7.0, 28), (10.0, 4)]:
            with torch.no_grad():
                ours = np.asarray(V.sample(pipe, [e], g, n, "baseline", [seed])[0][0]).astype(np.int16)
                lat = torch.randn((1, pipe.transformer.config.in_channels, 128, 128),
                                  generator=torch.Generator("cpu").manual_seed(seed)).to("cuda", torch.float16)
                off = pipe(prompt_embeds=e["pe"].to("cuda"), negative_prompt_embeds=e["ne"].to("cuda"),
                           pooled_prompt_embeds=e["pp"].to("cuda"), negative_pooled_prompt_embeds=e["npool"].to("cuda"),
                           guidance_scale=g, num_inference_steps=n, latents=lat, output_type="pil").images[0]
            d = np.abs(ours - np.asarray(off).astype(np.int16))
            rows.append(dict(model=model, id=c.id, g=g, nfe=n, max_abs_px=int(d.max()), mean_abs_px=float(d.mean()),
                             frac_px_diff=float((d > 0).mean())))
            print(rows[-1], flush=True)
    return rows


def check_tangoflux():
    import v9_audio as A
    vae, model = A.load()
    caps = pd.read_csv(A.CAPS, sep="\t").head(3)
    rows = []
    for _, c in caps.iterrows():
        seed = A.h(f"{c.id}|audio|0")
        for g, n in [(4.5, 4), (4.5, 25), (7.0, 3)]:
            with torch.no_grad():
                ours, _, _ = A.sample(vae, model, c.caption, g, n, "baseline", seed)
                torch.manual_seed(seed)
                lat = model.inference_flow(c.caption, num_inference_steps=n, guidance_scale=g, duration=10, disable_progress=True)
                off = vae.decode(lat.transpose(2, 1)).sample.cpu()[0][:, : int(10 * vae.config.sampling_rate)]
            d = (ours.float() - off.float()).abs()
            rows.append(dict(model="tangoflux", id=c.id, g=g, nfe=n, max_abs=float(d.max()), mean_abs=float(d.mean()),
                             ref_rms=float(off.float().pow(2).mean().sqrt())))
            print(rows[-1], flush=True)
    return rows


def check_sao():
    import v9_sao as S
    pipe = S.load()
    caps = pd.read_csv(S.CAPS, sep="\t").head(3)
    rows = []
    for _, c in caps.iterrows():
        seed = S.h(f"{c.id}|audio|0")
        for g, n in [(7.0, 4), (7.0, 25), (10.5, 3)]:
            with torch.no_grad():
                ours, _, _ = S.sample(pipe, c.caption, g, n, "baseline", seed)
                off = pipe(c.caption, num_inference_steps=n, guidance_scale=g, audio_end_in_s=S.DUR,
                           generator=torch.Generator("cuda").manual_seed(seed), output_type="pt").audios[0].float().cpu()
            d = (ours.float() - off).abs()
            rows.append(dict(model="sao", id=c.id, g=g, nfe=n, max_abs=float(d.max()), mean_abs=float(d.mean()),
                             ref_rms=float(off.pow(2).mean().sqrt())))
            print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    m = sys.argv[1]
    rows = check_image(m) if m in ("sd3", "sd35") else check_tangoflux() if m == "tangoflux" else check_sao()
    pd.DataFrame(rows).to_csv(ROOT / f"results/E9_official_check_{m}.csv", index=False)
