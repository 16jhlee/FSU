"""F5-TTS runner (family C teacher), following f5_tts/eval/eval_infer_batch.py.

Differences from the official script (all sampler/IO level):
  * per-utterance noise seed: seed = hash(utt_id|noise|K) (official: one global seed or None)
  * nfe / cfg_strength / sway / use_epss exposed (official: nfe 32, cfg 2.0, sway -1, epss default)
  * duration = official byte-ratio rule (independent of NFE), logged to _lengths.tsv
Batch size is 1, so the seed is per utterance.
"""
import argparse
import hashlib
import sys
from importlib.resources import files
from pathlib import Path

import torch
import torchaudio
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.eval.utils_eval import get_inference_prompt, get_seedtts_testset_metainfo
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def h(s, mod=2**31 - 1):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16) % mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="F5TTS_v1_Base")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--sway", type=float, default=-1.0)
    ap.add_argument("--use_epss", type=int, default=1, help="official default 1 (no effect at nfe 32)")
    ap.add_argument("--noise", default="utt", help="utt | utt:K")
    ap.add_argument("--cfg_interval", default=None, help="E8': 'tmin,tmax' CFG only for t in range (patch F5TTS_E8p)")
    a = ap.parse_args()
    device = "cuda:0"
    target_rms = 0.1

    cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{a.exp}.yaml")))
    model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")
    ms = cfg.model.mel_spec
    vocab_char_map, vocab_size = get_tokenizer(cfg.datasets.name, cfg.model.tokenizer)
    model = CFM(
        transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=ms.n_mel_channels),
        mel_spec_kwargs=dict(n_fft=ms.n_fft, hop_length=ms.hop_length, win_length=ms.win_length,
                             n_mel_channels=ms.n_mel_channels, target_sample_rate=ms.target_sample_rate,
                             mel_spec_type=ms.mel_spec_type),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_char_map,
    ).to(device)
    model = load_checkpoint(model, a.ckpt, device, dtype=None, use_ema=True)
    model.cfg_interval = [float(x) for x in a.cfg_interval.split(",")] if a.cfg_interval else None
    vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=str(ROOT / "ckpt" / "vocos-mel-24khz"))

    meta = get_seedtts_testset_metainfo(a.meta)
    prompts = get_inference_prompt(meta, speed=1.0, tokenizer=cfg.model.tokenizer,
                                   target_sample_rate=ms.target_sample_rate, n_mel_channels=ms.n_mel_channels,
                                   hop_length=ms.hop_length, mel_spec_type=ms.mel_spec_type,
                                   target_rms=target_rms, use_truth_duration=False, infer_batch_size=1)
    k = a.noise.split(":")[1] if ":" in a.noise else "0"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "_lengths.tsv", "a", encoding="utf-8")
    for utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, texts in prompts:
        assert len(utts) == 1
        utt = utts[0]
        if (out / f"{utt}.wav").exists():
            continue
        with torch.inference_mode():
            gen, _ = model.sample(cond=ref_mels.to(device), text=texts,
                                  duration=torch.tensor(total_mel_lens, dtype=torch.long, device=device),
                                  lens=torch.tensor(ref_mel_lens, dtype=torch.long, device=device),
                                  steps=a.nfe, cfg_strength=a.cfg, sway_sampling_coef=a.sway,
                                  seed=h(f"{utt}|noise|{k}"), use_epss=bool(a.use_epss))
            g = gen[0, ref_mel_lens[0]:total_mel_lens[0], :].unsqueeze(0).permute(0, 2, 1).float()
            wav = vocoder.decode(g).cpu()
        if ref_rms_list[0] < target_rms:
            wav = wav * ref_rms_list[0] / target_rms
        torchaudio.save(str(out / f"{utt}.wav"), wav, ms.target_sample_rate)
        log.write(f"{utt}\t{a.nfe}\t{a.cfg}\t{a.noise}\t{total_mel_lens[0] - ref_mel_lens[0]}\t{wav.shape[-1]}\n")
        log.flush()
    log.close()


if __name__ == "__main__":
    sys.exit(main())
