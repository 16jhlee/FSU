"""E14-6: first-step guidance size per unconditional branch for the text-conditioned decoders (F5-TTS v0, ZipVoice),
E10-7 procedure (scripts/e10_rho.py): en_100, noise seeds k = 0, 1, generated region, RMS = sqrt(mean a^2),
  xbar_k = NFE 1, w = 0 output;  g_k(branch) = [NFE 1, w = 1 output with that branch] - xbar_k = v_c - v_branch at t = 0.
Branches are built only by zeroing / dropping the condition inputs the models already drop for CFG (no architecture change):
  allnull  : text and prompt audio dropped (F5 official CFG)
  textnull : text dropped, prompt audio kept (ZipVoice official branch at t <= 0.5)
  audionull: prompt audio dropped, text kept (speaker-null counterpart)
F5: transformer(drop_audio_cond, drop_text) flags; ZipVoice: DiffusionModel.forward with the branch's zeroed condition and
plain (1+w)c - w u (no x2 rule: pure ||v_c - v_u||; E10-7's 0.85 was [w 0.5] x2 = the same pure text-null quantity).
python scripts/e14_branches.py f5|zip   (env_f5dmo | env_zip)  -> results/E14_textcond_branches_<model>.csv
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e10_rho import h, rms  # noqa: E402

BRANCHES = {"allnull": (True, True), "textnull": (True, False), "audionull": (False, True)}  # (drop text, drop audio)
UTTS = [l.split("|")[0] for l in (ROOT / "data/lists/en_100.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()]
BR = {"b": "allnull"}


def rows_for(utt, out):
    X = [out(k, 0.0) for k in range(2)]
    rows, G = [], {}
    for b in BRANCHES:
        BR["b"] = b
        for k in range(2):
            G[b, k] = out(k, 1.0) - X[k]
            rows.append(dict(utt=utt, k=k, branch=b, g_rms=rms(G[b, k])))
    for k in range(2):  # cosine between branch guidance fields
        for b in ("textnull", "audionull"):
            a, c = G["allnull", k].ravel(), G[b, k].ravel()
            rows.append(dict(utt=utt, k=k, branch=f"cos(allnull,{b})", g_rms=float(a @ c / np.linalg.norm(a) / np.linalg.norm(c))))
    return rows


def f5():
    from importlib.resources import files
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.eval.utils_eval import get_inference_prompt, get_seedtts_testset_metainfo
    from f5_tts.infer.utils_infer import load_checkpoint
    from f5_tts.model import CFM
    from f5_tts.model.utils import get_tokenizer
    device = "cuda:0"
    cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml")))
    model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")
    ms = cfg.model.mel_spec
    vocab_char_map, vocab_size = get_tokenizer(cfg.datasets.name, cfg.model.tokenizer)
    model = CFM(transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=ms.n_mel_channels),
                mel_spec_kwargs=dict(n_fft=ms.n_fft, hop_length=ms.hop_length, win_length=ms.win_length,
                                     n_mel_channels=ms.n_mel_channels, target_sample_rate=ms.target_sample_rate,
                                     mel_spec_type=ms.mel_spec_type),
                odeint_kwargs=dict(method="euler"), vocab_char_map=vocab_char_map).to(device)
    model = load_checkpoint(model, str(ROOT / "ckpt/F5-TTS/F5TTS_Base/model_1200000.safetensors"), device, dtype=None,
                            use_ema=True)
    model.cfg_interval = None
    tr = model.transformer
    orig = tr.forward

    def fwd(x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False, cfg_infer=False, cache=False, **kw):
        if not cfg_infer:
            return orig(x=x, cond=cond, text=text, time=time, mask=mask, drop_audio_cond=drop_audio_cond,
                        drop_text=drop_text, cache=False, **kw)
        dt, da = BRANCHES[BR["b"]]
        c = orig(x=x, cond=cond, text=text, time=time, mask=mask, drop_audio_cond=False, drop_text=False, cache=False, **kw)
        u = orig(x=x, cond=cond, text=text, time=time, mask=mask, drop_audio_cond=da, drop_text=dt, cache=False, **kw)
        return torch.cat([c, u], 0)
    tr.forward = fwd
    meta = get_seedtts_testset_metainfo(str(ROOT / "data/lists/en_100.meta.lst"))
    prompts = get_inference_prompt(meta, speed=1.0, tokenizer=cfg.model.tokenizer, target_sample_rate=ms.target_sample_rate,
                                   n_mel_channels=ms.n_mel_channels, hop_length=ms.hop_length, mel_spec_type=ms.mel_spec_type,
                                   target_rms=0.1, use_truth_duration=False, infer_batch_size=1)
    rows = []
    for us, _, ref_mels, ref_lens, tot_lens, texts in prompts:
        utt = us[0]

        def out(k, w):
            with torch.inference_mode():
                gen, _ = model.sample(cond=ref_mels.to(device), text=texts,
                                      duration=torch.tensor(tot_lens, dtype=torch.long, device=device),
                                      lens=torch.tensor(ref_lens, dtype=torch.long, device=device), steps=1,
                                      cfg_strength=w, sway_sampling_coef=-1.0, seed=h(f"{utt}|noise|{k}"), use_epss=False)
            return gen[0, ref_lens[0]:tot_lens[0], :].double().cpu().numpy().T
        rows += rows_for(utt, out)
        print(utt, flush=True)
    return rows


def zip_():
    import zipvoice.bin.infer_zipvoice as Z
    from lhotse.utils import fix_random_seed
    from zipvoice.models.modules import solver as zsolver
    captured = {}

    class Capture:
        def decode(self, feats):
            captured["f"] = feats.detach().float().cpu().numpy()[0]
            return torch.zeros(1, 1, feats.shape[-1] * 256)
    for k_, v_ in {"mode": "euler", "t_span": None, "restart": None, "cfg_interval": None, "pci": False}.items():
        zsolver.E5[k_] = v_

    def fwd(self, t, x, text_condition, speech_condition, padding_mask=None, guidance_scale=0.0, **kw):
        g = guidance_scale if torch.is_tensor(guidance_scale) else torch.tensor(guidance_scale, dtype=t.dtype, device=t.device)
        if (g == 0.0).all():
            return self.model_func(t=t, xt=x, text_condition=text_condition, speech_condition=speech_condition,
                                   padding_mask=padding_mask, **kw)
        dt, da = BRANCHES[BR["b"]]
        tu = torch.zeros_like(text_condition) if dt else text_condition
        su = torch.zeros_like(speech_condition) if da else speech_condition
        u, c = self.model_func(t=t, xt=torch.cat([x] * 2, 0), text_condition=torch.cat([tu, text_condition], 0),
                               speech_condition=torch.cat([su, speech_condition], 0),
                               padding_mask=torch.cat([padding_mask] * 2, 0), **kw).chunk(2, dim=0)
        return (1 + g) * c - g * u
    zsolver.DiffusionModel.forward = fwd
    md = ROOT / "ckpt/ZipVoice/zipvoice"
    tokenizer = Z.EmiliaTokenizer(token_file=md / "tokens.txt")
    cfgj = json.loads((md / "model.json").read_text())
    model = Z.ZipVoice(**cfgj["model"], vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id)
    Z.load_checkpoint(filename=md / "model.pt", model=model, strict=True)
    device = torch.device("cuda", 0)
    model = model.to(device).eval()
    fe = Z.VocosFbank()
    meta = {l.split("|")[0]: l.split("|") for l in (ROOT / "data/lists/en_100.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()}
    os.makedirs(ROOT / "out/smoke/E14_branch_zip", exist_ok=True)
    rows = []
    for utt in UTTS:
        _, ptext, pwav, text = meta[utt][:4]

        def out(k, w):
            fix_random_seed(h(f"{utt}|noise|{k}"))
            with torch.inference_mode():
                Z.generate_sentence_raw_evaluation(save_path=str(ROOT / "out/smoke/E14_branch_zip/tmp.wav"), prompt_text=ptext,
                                                   prompt_wav=pwav, text=text, model=model, vocoder=Capture(), tokenizer=tokenizer,
                                                   feature_extractor=fe, device=device, num_step=1, guidance_scale=w, t_shift=0.5)
            return captured["f"].astype(np.float64)
        rows += rows_for(utt, out)
        print(utt, flush=True)
    return rows


def main():
    key = sys.argv[1]
    rows = {"f5": f5, "zip": zip_}[key]()
    with open(ROOT / f"results/E14_textcond_branches_{key}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for b in sorted({r["branch"] for r in rows}):
        print(b, round(float(np.mean([r["g_rms"] for r in rows if r["branch"] == b])), 4))


if __name__ == "__main__":
    main()
