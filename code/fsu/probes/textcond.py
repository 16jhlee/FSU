"""Probes of the text-conditioned decoders F5-TTS (environment env_f5dmo) and ZipVoice (environment env_zip, with
repos/ZipVoice on PYTHONPATH). Output features (mel / fbank, D x T) of the generated region.

    python -m fsu.probes.textcond rho --decoder f5tts
    python -m fsu.probes.textcond branches --decoder zipvoice

F5-TTS: official F5TTS_Base (v0) model as the generation runs, Euler, sway -1, EPSS off, noise seed
seed_of(f"{utt}|noise|{k}"); teacher 32 steps, cfg 2. ZipVoice: official model, raw evaluation, t-shift 0.5, the
patched solver (patches/zipvoice.diff) reset to plain Euler, fix_random_seed(seed_of(f"{utt}|noise|{k}")) before each
utterance (the per-utterance seed rule of the patched command line); teacher 16 steps, guidance 1.
Branches of the first-step guidance are built only by dropping / zeroing the condition inputs the models already drop
for CFG: allnull (text and prompt audio), textnull (text only; ZipVoice's official branch at t <= 0.5), audionull
(prompt audio only).
"""
import json
import sys

import numpy as np
import torch

from fsu.common import CKPT, OUT, seed_of
from fsu.datasets import get as get_set
from fsu.probes import TEST_SET, cloud_stats, main_for, rms, utterance_rows

BRANCHES = {"allnull": (True, True), "textnull": (True, False), "audionull": (False, True)}  # (drop text, drop audio)


def branch_rows(probe, utt, out):
    """g_k(branch) = [1-step w = 1 output with that branch] - [1-step w = 0 output], k = 0, 1, plus the cosines between
    the all-null and the other branch fields. `out(k, w)` runs one step; probe.branch selects the branch."""
    X = [out(k, 0.0) for k in range(2)]
    rows, G = [], {}
    for b in BRANCHES:
        probe.branch = b
        for k in range(2):
            G[b, k] = out(k, 1.0) - X[k]
            rows.append(dict(utt=utt, k=k, branch=b, g_rms=rms(G[b, k])))
    for k in range(2):
        for b in ("textnull", "audionull"):
            a, c = G["allnull", k].ravel(), G[b, k].ravel()
            rows.append(dict(utt=utt, k=k, branch=f"cos(allnull,{b})", g_rms=float(a @ c / np.linalg.norm(a) / np.linalg.norm(c))))
    return rows


class F5TTSProbe:
    """F5-TTS v0 CFM sampled directly; one method per measurement."""
    exp = "F5TTS_Base"
    ckpt = CKPT / "F5-TTS" / "F5TTS_Base" / "model_1200000.safetensors"
    device = "cuda:0"

    def __init__(self, decoder="f5tts"):
        from importlib.resources import files
        from hydra.utils import get_class
        from omegaconf import OmegaConf
        from f5_tts.infer.utils_infer import load_checkpoint
        from f5_tts.model import CFM
        from f5_tts.model.utils import get_tokenizer
        self.cfg = cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{self.exp}.yaml")))
        model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")
        self.ms = ms = cfg.model.mel_spec
        vocab_char_map, vocab_size = get_tokenizer(cfg.datasets.name, cfg.model.tokenizer)
        model = CFM(transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=ms.n_mel_channels),
                    mel_spec_kwargs=dict(n_fft=ms.n_fft, hop_length=ms.hop_length, win_length=ms.win_length,
                                         n_mel_channels=ms.n_mel_channels, target_sample_rate=ms.target_sample_rate,
                                         mel_spec_type=ms.mel_spec_type),
                    odeint_kwargs=dict(method="euler"), vocab_char_map=vocab_char_map).to(self.device)
        self.model = load_checkpoint(model, str(self.ckpt), self.device, dtype=None, use_ema=True)
        self.model.cfg_interval = None
        self.branch = "allnull"

    def prompts(self):
        from f5_tts.eval.utils_eval import get_inference_prompt, get_seedtts_testset_metainfo
        cfg, ms = self.cfg, self.ms
        meta = get_seedtts_testset_metainfo(str(get_set(TEST_SET).path))
        return get_inference_prompt(meta, speed=1.0, tokenizer=cfg.model.tokenizer, target_sample_rate=ms.target_sample_rate,
                                    n_mel_channels=ms.n_mel_channels, hop_length=ms.hop_length,
                                    mel_spec_type=ms.mel_spec_type, target_rms=0.1, use_truth_duration=False,
                                    infer_batch_size=1)

    def sample(self, prompt, k, n, w):
        """Generated-region mel (D x T, float64) with n steps and cfg strength w from noise draw k."""
        us, _, ref_mels, ref_lens, tot_lens, texts = prompt
        utt, device = us[0], self.device
        with torch.inference_mode():
            gen, _ = self.model.sample(cond=ref_mels.to(device), text=texts,
                                       duration=torch.tensor(tot_lens, dtype=torch.long, device=device),
                                       lens=torch.tensor(ref_lens, dtype=torch.long, device=device), steps=n,
                                       cfg_strength=w, sway_sampling_coef=-1.0, seed=seed_of(f"{utt}|noise|{k}"),
                                       use_epss=False)
        return gen[0, ref_lens[0]:tot_lens[0], :].double().cpu().numpy().T

    def rho(self, n_utt=100, teachers=8):
        """Teacher (32 steps, cfg 2) seeds 0..teachers-1; X_k = 1-step w = 0 output, G_k = [1-step w = 1] - X_k."""
        want = {r[0] for r in utterance_rows(n_utt)}
        rows = []
        for p in self.prompts():
            utt = p[0][0]
            if utt not in want:
                continue
            T = [self.sample(p, k, 32, 2.0) for k in range(teachers)]
            X = [self.sample(p, k, 1, 0.0) for k in range(2)]
            G = [self.sample(p, k, 1, 1.0) - X[k] for k in range(2)]
            rows += cloud_stats(utt, T, X, G)
            print(utt, flush=True)
        return rows

    def branches(self, n_utt=None):
        """First-step guidance per branch, through the transformer's drop_text / drop_audio_cond flags."""
        tr = self.model.transformer
        orig = tr.forward
        probe = self

        def fwd(x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False, cfg_infer=False, cache=False, **kw):
            if not cfg_infer:
                return orig(x=x, cond=cond, text=text, time=time, mask=mask, drop_audio_cond=drop_audio_cond,
                            drop_text=drop_text, cache=False, **kw)
            dt, da = BRANCHES[probe.branch]
            c = orig(x=x, cond=cond, text=text, time=time, mask=mask, drop_audio_cond=False, drop_text=False, cache=False, **kw)
            u = orig(x=x, cond=cond, text=text, time=time, mask=mask, drop_audio_cond=da, drop_text=dt, cache=False, **kw)
            return torch.cat([c, u], 0)
        tr.forward = fwd
        want = {r[0] for r in utterance_rows(n_utt)}
        rows = []
        for p in self.prompts():
            utt = p[0][0]
            if utt not in want:
                continue
            rows += branch_rows(self, utt, lambda k, w: self.sample(p, k, 1, w))
            print(utt, flush=True)
        return rows


class ZipVoiceProbe:
    """ZipVoice sampled through its raw-evaluation generation function with the vocoder replaced by a feature capture."""
    model_dir = CKPT / "ZipVoice" / "zipvoice"

    def __init__(self, decoder="zipvoice"):
        import zipvoice.bin.infer_zipvoice as Z
        from lhotse.utils import fix_random_seed
        from zipvoice.models.modules import solver as zsolver
        self.Z, self.fix_random_seed, self.zsolver = Z, fix_random_seed, zsolver
        self.captured = {}
        for k_, v_ in {"mode": "euler", "t_span": None, "restart": None, "cfg_interval": None, "pci": False}.items():
            zsolver.SAMPLER[k_] = v_
        md = self.model_dir
        self.tokenizer = Z.EmiliaTokenizer(token_file=md / "tokens.txt")
        cfgj = json.loads((md / "model.json").read_text())
        model = Z.ZipVoice(**cfgj["model"], vocab_size=self.tokenizer.vocab_size, pad_id=self.tokenizer.pad_id)
        Z.load_checkpoint(filename=md / "model.pt", model=model, strict=True)
        self.device = torch.device("cuda", 0)
        self.model = model.to(self.device).eval()
        self.fe = Z.VocosFbank()
        self.tmp = OUT / "probes" / "zipvoice_tmp"  # the zero waveform of the capture vocoder is written here
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.branch = "allnull"

    def capture(self):
        captured = self.captured

        class Capture:  # stands in for the vocoder: keeps the predicted features, returns silence of the right length
            def decode(self, feats):
                captured["f"] = feats.detach().float().cpu().numpy()[0]
                return torch.zeros(1, 1, feats.shape[-1] * 256)
        return Capture()

    def sample(self, row, k, n, w):
        """Predicted features (float64) with n steps and guidance scale w from noise draw k."""
        utt, ptext, pwav, text = row
        self.fix_random_seed(seed_of(f"{utt}|noise|{k}"))
        with torch.inference_mode():
            self.Z.generate_sentence_raw_evaluation(save_path=str(self.tmp / "tmp.wav"), prompt_text=ptext,
                                                    prompt_wav=pwav, text=text, model=self.model, vocoder=self.capture(),
                                                    tokenizer=self.tokenizer, feature_extractor=self.fe,
                                                    device=self.device, num_step=n, guidance_scale=w, t_shift=0.5)
        return self.captured["f"].astype(np.float64)

    def rho(self, n_utt=100, teachers=8):
        """Teacher (16 steps, guidance 1) seeds 0..teachers-1; X_k = 1-step w = 0 output; G_k = [1-step w = 0.5] - X_k:
        the official guidance at t <= 0.5 uses the text-null branch with the scale doubled, so this is
        v_c - v_u(text-null)."""
        rows = []
        for row in utterance_rows(n_utt):
            utt = row[0]
            T = [self.sample(row, k, 16, 1.0) for k in range(teachers)]
            X = [self.sample(row, k, 1, 0.0) for k in range(2)]
            G = [self.sample(row, k, 1, 0.5) - X[k] for k in range(2)]
            rows += cloud_stats(utt, T, X, G)
            print(utt, flush=True)
        return rows

    def branches(self, n_utt=None):
        """First-step guidance per branch: DiffusionModel.forward with the branch's zeroed condition and plain
        (1 + w) c - w u (no doubling rule), i.e. the pure ||v_c - v_u||."""
        probe = self

        def fwd(self, t, x, text_condition, speech_condition, padding_mask=None, guidance_scale=0.0, **kw):
            g = guidance_scale if torch.is_tensor(guidance_scale) else torch.tensor(guidance_scale, dtype=t.dtype, device=t.device)
            if (g == 0.0).all():
                return self.model_func(t=t, xt=x, text_condition=text_condition, speech_condition=speech_condition,
                                       padding_mask=padding_mask, **kw)
            dt, da = BRANCHES[probe.branch]
            tu = torch.zeros_like(text_condition) if dt else text_condition
            su = torch.zeros_like(speech_condition) if da else speech_condition
            u, c = self.model_func(t=t, xt=torch.cat([x] * 2, 0), text_condition=torch.cat([tu, text_condition], 0),
                                   speech_condition=torch.cat([su, speech_condition], 0),
                                   padding_mask=torch.cat([padding_mask] * 2, 0), **kw).chunk(2, dim=0)
            return (1 + g) * c - g * u
        self.zsolver.DiffusionModel.forward = fwd
        rows = []
        for row in utterance_rows(n_utt):
            rows += branch_rows(self, row[0], lambda k, w: self.sample(row, k, 1, w))
            print(row[0], flush=True)
        return rows


PROBES = {"f5tts": F5TTSProbe, "zipvoice": ZipVoiceProbe}


def main(argv=None):
    main_for("textcond", lambda decoder: PROBES[decoder](decoder), argv)


if __name__ == "__main__":
    sys.exit(main())
