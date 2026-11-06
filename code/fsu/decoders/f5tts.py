"""F5-TTS and DMOSpeech 2 runners (environment env_f5dmo).

F5-TTS (upstream repos/F5-TTS + patches/f5tts.diff) follows f5_tts/eval/eval_infer_batch.py with batch size 1, a
per-utterance noise seed seed_of(f"{utt}|noise|{k}"), and the official byte-ratio duration rule (independent of the
step count):
    python -m fsu.decoders.f5tts f5 --exp F5TTS_v1_Base --ckpt <ckpt> --meta <list> --out_dir <dir> --nfe 16
DMOSpeech 2 (upstream repos/DMOSpeech2 + patches/dmospeech2.diff) wraps DMOInference.generate:
    official  teacher-guided student (teacher_steps 16, teacher_stopping_time 0.07, student_start_step 1)
    student   student only on the time grid --grid (before sway)
    teacher   the internal teacher alone: full Euler ODE with CFG, sway -1
    python -m fsu.decoders.f5tts dmo --meta <list> --out_dir <dir> --mode student --grid 0,0.5
Neither model has a separate token stage, so these runners only decode.
"""
import argparse
import sys
from pathlib import Path

from fsu.common import CKPT, REPOS, ROOT, noise_index, read_list, resolve, seed_of


class F5TTS:
    """F5-TTS sampler runner."""

    @staticmethod
    def parser(ap):
        ap.add_argument("--exp", default="F5TTS_v1_Base")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--meta", required=True)
        ap.add_argument("--out_dir", required=True)
        ap.add_argument("--nfe", type=int, default=32)
        ap.add_argument("--cfg", type=float, default=2.0)
        ap.add_argument("--sway", type=float, default=-1.0)
        ap.add_argument("--use_epss", type=int, default=1, help="official default 1 (no effect at 32 steps)")
        ap.add_argument("--noise", default="utt", help="utt | utt:K")
        ap.add_argument("--cfg_interval", default=None, help="'tmin,tmax': guidance only for t in this range")

    @staticmethod
    def run(a):
        import torch
        import torchaudio
        from importlib.resources import files
        from hydra.utils import get_class
        from omegaconf import OmegaConf
        from f5_tts.eval.utils_eval import get_inference_prompt, get_seedtts_testset_metainfo
        from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
        from f5_tts.model import CFM
        from f5_tts.model.utils import get_tokenizer

        device, target_rms = "cuda:0", 0.1
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
        model = load_checkpoint(model, str(resolve(a.ckpt)), device, dtype=None, use_ema=True)
        model.cfg_interval = [float(x) for x in a.cfg_interval.split(",")] if a.cfg_interval else None
        vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=str(CKPT / "vocos-mel-24khz"))

        meta = get_seedtts_testset_metainfo(str(resolve(a.meta)))
        prompts = get_inference_prompt(meta, speed=1.0, tokenizer=cfg.model.tokenizer,
                                       target_sample_rate=ms.target_sample_rate, n_mel_channels=ms.n_mel_channels,
                                       hop_length=ms.hop_length, mel_spec_type=ms.mel_spec_type,
                                       target_rms=target_rms, use_truth_duration=False, infer_batch_size=1)
        k = noise_index(a.noise)
        out = resolve(a.out_dir)
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
                                      seed=seed_of(f"{utt}|noise|{k}"), use_epss=bool(a.use_epss))
                g = gen[0, ref_mel_lens[0]:total_mel_lens[0], :].unsqueeze(0).permute(0, 2, 1).float()
                wav = vocoder.decode(g).cpu()
            if ref_rms_list[0] < target_rms:
                wav = wav * ref_rms_list[0] / target_rms
            torchaudio.save(str(out / f"{utt}.wav"), wav, ms.target_sample_rate)
            log.write(f"{utt}\t{a.nfe}\t{a.cfg}\t{a.noise}\t{total_mel_lens[0] - ref_mel_lens[0]}\t{wav.shape[-1]}\n")
            log.flush()
        log.close()


class DMOSpeech2:
    """DMOSpeech 2 runner. Before each utterance torch.manual_seed(seed_of(f"{utt}|noise|{k}")) seeds the duration
    predictor, the teacher noise and every student re-noising draw, so all step settings start from the same z."""

    @staticmethod
    def parser(ap):
        ap.add_argument("--meta", required=True)
        ap.add_argument("--out_dir", required=True)
        ap.add_argument("--mode", choices=["official", "student", "teacher"], required=True)
        ap.add_argument("--grid", default="0,0.25,0.5,0.75", help="student time grid before sway")
        ap.add_argument("--teacher_steps", type=int, default=32, help="teacher mode: Euler steps")
        ap.add_argument("--cfg", type=float, default=2.0, help="teacher mode: cfg_strength")
        ap.add_argument("--eta", type=float, default=0.0, help="teacher mode: 0 = deterministic ODE")
        ap.add_argument("--noise", default="utt")

    @staticmethod
    def run(a):
        import os
        meta, out = resolve(a.meta), resolve(a.out_dir)
        src = REPOS / "DMOSpeech2" / "src"
        sys.path.insert(0, str(src))  # the vendored f5_tts must shadow the pip F5-TTS package
        os.chdir(src)
        import torch
        import torchaudio
        import f5_tts
        f5_tts.__path__ = [str(src / "f5_tts")]  # namespace package: restrict it to the vendored copy
        from infer import DMOInference
        import f5_tts.model
        assert str(src) in f5_tts.model.__file__, f5_tts.model.__file__

        ck = CKPT / "DMOSpeech2"
        dmo = DMOInference(student_checkpoint_path=str(ck / "model_85000.pt"),
                           duration_predictor_path=str(ck / "model_1500.pt"), device="cuda")
        if a.mode == "student":
            dmo.student_steps_override = [float(x) for x in a.grid.split(",")]
        k = noise_index(a.noise)
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "_lengths.tsv", "a", encoding="utf-8")
        for utt, ptext, pwav, text in read_list(meta):
            if (out / f"{utt}.wav").exists():
                continue
            torch.manual_seed(seed_of(f"{utt}|noise|{k}"))
            dur = dmo.predict_duration(pwav, text, ptext)
            torch.manual_seed(seed_of(f"{utt}|noise|{k}"))
            if a.mode == "official":
                wav = dmo.generate(text, pwav, prompt_text=ptext, duration=dur)
            elif a.mode == "teacher":  # student_start_step = len(grid) -> no student steps
                wav = dmo.generate(text, pwav, prompt_text=ptext, duration=dur, teacher_steps=a.teacher_steps,
                                   teacher_stopping_time=1.0, student_start_step=4, eta=a.eta, cfg_strength=a.cfg)
            else:
                wav = dmo.generate(text, pwav, prompt_text=ptext, duration=dur, teacher_steps=0, student_start_step=0)
            wav = torch.from_numpy(wav).float().unsqueeze(0)
            torchaudio.save(str(out / f"{utt}.wav"), wav, 24000)
            tag = f"T{a.teacher_steps}/cfg{a.cfg}/eta{a.eta}" if a.mode == "teacher" else a.grid
            log.write(f"{utt}\t{a.mode}\t{tag}\t{a.noise}\t{dur}\t{wav.shape[-1]}\n")
            log.flush()
        log.close()


RUNNERS = {"f5": F5TTS, "dmo": DMOSpeech2}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="model", required=True)
    for name, cls in RUNNERS.items():
        cls.parser(sub.add_parser(name, help=cls.__doc__.splitlines()[0]))
    a = ap.parse_args(argv)
    RUNNERS[a.model].run(a)


if __name__ == "__main__":
    sys.exit(main())
