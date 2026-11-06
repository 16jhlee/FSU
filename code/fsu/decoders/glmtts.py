"""GLM-TTS runner (environment env_glm; upstream repos/GLM-TTS + patches/glmtts.diff, official pipeline of
glmtts_inference.py).

prepare (--stage tokens): official text frontend, speech tokenizer, CAM++ embedding and LLM (official generate_long,
seeded per utterance with seed_of(utt) instead of the official constant 0) -> cached segments
{token, prompt_token, prompt_feat, embedding}. The flow decoder is not run in this stage.
decode: official Token2Wav.token2wav_with_cache (flow DiT + HiFT, 24 kHz) per segment, segments concatenated as in
generate_long. Defaults are official: 10 steps, cosine grid, Euler, w = 0.7. --cfg_branch all is the released
configuration (speech_token_cfg False, i.e. the speaker-null branch); allnull also removes the speech tokens.
"""
import os
import sys

import torch
import torchaudio

from fsu.common import REPOS, noise_index, seed_of
from fsu.decoders.base import Decoder, tv_scales

if not hasattr(torch, "npu"):  # the official frontend calls torch.npu.is_available() (Ascend branch) unconditionally
    from types import SimpleNamespace
    torch.npu = SimpleNamespace(is_available=lambda: False)

REPO = REPOS / "GLM-TTS"


class GLMTTS(Decoder):
    """GLM-TTS flow decoder runner."""
    defaults = dict(nfe=10, cfg=0.7, noise="utt:0")
    branches = ("all", "allnull", "speaker", "token")

    @classmethod
    def add_arguments(cls, ap):
        ap.add_argument("--pci", type=int, default=0, help="Prompt-consistency inference")
        ap.add_argument("--text_frontend", type=int, default=1, help="0 = raw text, no normalisation (ko/ja/de)")

    def __init__(self, args):
        super().__init__(args)
        os.chdir(REPO)
        sys.path.insert(0, str(REPO))

    def prepare(self, rows, cache_dir, out_dir=None):
        import glmtts_inference as G
        frontend, text_frontend, _, llm, flow = G.load_models(use_phoneme=False, sample_rate=24000)
        if not self.args.text_frontend:  # ko/ja/de: raw text (the zh/en normalisers misfire on these languages)
            text_frontend.text_normalize = lambda t: t
        cache_dir.mkdir(parents=True, exist_ok=True)
        for utt, ptext, pwav, text in rows:
            out = cache_dir / f"{utt}.pt"
            if out.exists():
                continue
            segs = []

            def capture(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
                segs.append(dict(token=torch.tensor(token_list, dtype=torch.long)[None], prompt_token=prompt_speech_tokens.cpu(),
                                 prompt_feat=speech_feat.cpu(), embedding=embedding.cpu()))
                return torch.zeros(1, 1), None
            # as jsonl_generate()
            prompt_text = text_frontend.text_normalize(ptext)
            synth_text = text_frontend.text_normalize(text)
            prompt_text_token = frontend._extract_text_token(prompt_text + " ")
            prompt_speech_token = frontend._extract_speech_token([pwav])
            speech_feat = frontend._extract_speech_feat(pwav, sample_rate=24000)
            embedding = frontend._extract_spk_embedding(pwav)
            cache_speech_token = [prompt_speech_token.squeeze().tolist()]
            flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(G.DEVICE)
            cache = {"cache_text": [prompt_text], "cache_text_token": [prompt_text_token],
                     "cache_speech_token": cache_speech_token, "use_cache": True}
            G.generate_long(frontend=frontend, text_frontend=text_frontend, llm=llm, flow=flow, text_info=[utt, synth_text],
                            cache=cache, embedding=embedding, seed=seed_of(utt), flow_prompt_token=flow_prompt_token,
                            speech_feat=speech_feat, device=G.DEVICE, use_phoneme=False, local_flow_forward=capture)
            torch.save(segs, out)
            print(utt, len(segs), [s["token"].shape[1] for s in segs], flush=True)

    def configure(self, flow):
        a = self.args
        flow.inference_cfg_rate = a.cfg
        flow.cfg_interval = self.floats(a.cfg_interval)
        flow.cfg_steps = self.floats(a.cfg_steps)
        flow.cfg_branch = a.cfg_branch
        flow.solver = a.solver
        flow.t_span_override = self.floats(a.t_span)
        nfe = len(flow.t_span_override) - 1 if a.t_span else a.nfe
        flow.cfg_branch_switch_t = a.branch_switch_t
        flow.mambo_alpha = a.mambo_alpha
        flow.cfg_zero_star = bool(a.cfg_zero_star)
        flow.zero_init_steps = a.zero_init
        flow.tfr = bool(a.tfr)
        flow.pci = bool(a.pci)
        flow.jrr = tuple(self.floats(a.jrr)) if a.jrr else None
        ts = flow.t_span_override or (1 - torch.cos(torch.linspace(0, 1, nfe + 1) * 0.5 * torch.pi)).tolist()  # cosine grid
        flow.tv_scales = tv_scales(ts, a.cfg, a.tv) if a.tv else None
        flow.external_rules = any([a.branch_switch_t is not None, a.mambo_alpha is not None, a.cfg_zero_star, a.zero_init,
                                   a.tfr, a.pci, a.jrr, a.tv])
        if flow.tv_scales:
            print("TV-CFG scales g_k:", [round(x, 3) for x in flow.tv_scales], flush=True)
        return nfe

    def decode(self, rows, cache_dir, out_dir):
        from utils import seed_util, tts_model_util, yaml_util
        a, dev = self.args, "cuda"
        flow = yaml_util.load_flow_model(os.path.join("ckpt", "flow", "flow.pt"), os.path.join("ckpt", "flow", "config.yaml"), dev)
        t2w = tts_model_util.Token2Wav(flow, sample_rate=24000, device=dev)
        nfe = self.configure(flow)
        k = noise_index(a.noise)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "_lengths.tsv", "a", encoding="utf-8")
        for utt, *_ in rows:
            wp = out_dir / f"{utt}.wav"
            if wp.exists():
                continue
            segs = torch.load(cache_dir / f"{utt}.pt")
            g = torch.Generator().manual_seed(seed_of(f"{utt}|noise|{k}"))
            flow.noise_override = torch.randn([1, 6000, flow.mel_dim], generator=g)
            wavs = []
            for i, s in enumerate(segs):
                flow.prompt_frames = int(s["prompt_feat"].shape[1])  # PCI: prompt block = first prompt frames
                seed_util.set_seed(seed_of(f"{utt}|hift|{i}"))
                with torch.inference_mode():
                    wav, _ = t2w.token2wav_with_cache(s["token"], n_timesteps=nfe, prompt_token=s["prompt_token"],
                                                      prompt_feat=s["prompt_feat"], embedding=s["embedding"])
                wavs.append(wav.detach().cpu())
            wav = torch.cat(wavs, dim=1)
            torchaudio.save(str(wp), wav, 24000)
            log.write(f"{utt}\t{nfe}\t{a.cfg}\t{a.noise}\t{sum(s['token'].shape[1] for s in segs)}\t{wav.shape[1]}\n")
            log.flush()
        log.close()


if __name__ == "__main__":
    GLMTTS.main()
