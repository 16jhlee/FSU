"""Per-utterance evaluation of a directory of generated wavs (environment env_eval).

Scores (written to <wav_dir>/_metrics/metrics.csv, embeddings to emb_<embedder>.npz):
  wer       Seed-TTS-eval protocol: en Whisper-large-v3 (fp32, scipy resampling to 16 kHz), zh Paraformer-zh (FunASR) +
            zhconv, normalisation copied from seed-tts-eval/run_wer.py; ko/ja CER and de WER with Whisper-large-v3 and a
            forced language
  werwh     zh: Whisper-large-v3 CER next to Paraformer-zh
  cer2      ko/ja: second ASR, SenseVoice-Small
  wavlm     Seed-TTS-eval SIM model (WavLM-large + ECAPA-TDNN, wavlm_large_finetune.pth)
  ecapa     speechbrain/spkrec-ecapa-voxceleb
  eres2net  3D-Speaker ERes2Net, VoxCeleb2 (English), ModelScope iic/speech_eres2net_sv_en_voxceleb_16k v1.0.3
  campp     3D-Speaker CAM++ (zh-cn), ModelScope iic/speech_campplus_sv_zh-cn_16k-common v1.0.0
  utmos     UTMOS22 strong (tarepan/SpeechMOS v1.2.0)
SIM = cos(generated, prompt). Each SIM is computed on the full wav and on a 6 s centre crop (sim_<emb>_c6). Prompt
embeddings are cached in data/cache/prompt_<emb>.npz. Existing columns are kept, so a run can be resumed or extended.

Prosody and spectral features (--acoustic, CPU only; <wav_dir>/_metrics/acoustic.csv):
  f0_mean_st, f0_sd_st  WORLD dio + stonemask F0 (16 kHz, 5 ms), voiced frames, semitones re 100 Hz
  rate_active           syllables per second of active speech (20 ms frames with RMS > max RMS - 40 dB);
                        syllables: en/de vowel groups per word, zh CJK characters, ko Hangul blocks, ja kana + kanji
  hf_ratio              energy 4-8 kHz / 0-8 kHz over active frames (24 kHz, |STFT| 1024/256)
  flux_var              variance of the frame-to-frame L2 difference of the 80-bin log-mel over active frames

    python -m fsu.metrics --meta data/lists/seedtts_en.lst --wav_dir out/... --lang en --metrics wer,wavlm,ecapa,eres2net,utmos
    python -m fsu.metrics --acoustic --meta data/lists/seedtts_en.lst --wav_dir out/... --lang en
"""
import argparse
import os
import re
import string
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from fsu.common import CKPT, DATA, REPOS, resolve

SV_DIR = REPOS / "seed-tts-eval" / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification"
WAVLM_CKPT = DATA / "ckpt" / "wavlm_large_finetune.pth"
DEV = "cuda:0"
CROP_S = 6.0
WHISPER_LANG = {"en": "english", "ko": "korean", "ja": "japanese", "de": "german"}


def read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            f = line.split("|")
            rows.append(dict(utt=f[0], prompt_text=f[1], prompt_wav=f[2], text=f[3]))
    return rows


def load16(path):
    import librosa
    import torch
    from torchaudio.transforms import Resample
    wav, sr = librosa.load(path, sr=None, mono=False)
    if wav.ndim == 2:
        wav = wav[0]
    w = torch.from_numpy(wav).unsqueeze(0).float()
    if sr != 16000:
        w = Resample(orig_freq=sr, new_freq=16000)(w)
    return w  # [1, T] at 16 kHz


def centre_crop(w, sec=CROP_S, sr=16000):
    n = int(sec * sr)
    if w.shape[-1] <= n:
        return w
    s = (w.shape[-1] - n) // 2
    return w[..., s:s + n]


# ---------------------------------------------------------------- speaker embedders
class Embedder:
    name = ""

    def __call__(self, w):
        raise NotImplementedError


class WavLMSV(Embedder):
    name = "wavlm"

    def __init__(self):
        import torch
        sys.path.insert(0, str(SV_DIR))
        from models.ecapa_tdnn import ECAPA_TDNN_SMALL  # official Seed-TTS-eval code
        sys.path.pop(0)
        m = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        m.load_state_dict(torch.load(WAVLM_CKPT, map_location="cpu")["model"], strict=False)
        self.m = m.to(DEV).eval()

    def __call__(self, w):
        import torch
        with torch.no_grad():
            return self.m(w.to(DEV))[0].float().cpu().numpy()


class EcapaSB(Embedder):
    name = "ecapa"

    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                savedir=str(CKPT / "spkrec-ecapa-voxceleb"),
                                                run_opts={"device": DEV}, local_strategy=LocalStrategy.COPY)

    def __call__(self, w):
        import torch
        with torch.no_grad():
            return self.m.encode_batch(w.to(DEV))[0, 0].float().cpu().numpy()


class CamPP(Embedder):
    name = "campp"
    model, revision = "iic/speech_campplus_sv_zh-cn_16k-common", "v1.0.0"

    def __init__(self):
        from modelscope.pipelines import pipeline
        self.p = pipeline(task="speaker-verification", model=self.model, model_revision=self.revision)

    def __call__(self, w):
        import torch
        with torch.no_grad():
            out = self.p([w[0].numpy(), w[0].numpy()], output_emb=True)
        return np.asarray(out["embs"][0], dtype=np.float32)


class ERes2NetEn(CamPP):
    """ERes2Net trained on VoxCeleb2 dev (English, 5,994 speakers), VoxCeleb1-O EER 0.83 %."""
    name = "eres2net"
    model, revision = "iic/speech_eres2net_sv_en_voxceleb_16k", "v1.0.3"


EMBEDDERS = {c.name: c for c in (WavLMSV, EcapaSB, CamPP, ERes2NetEn)}


def cos(a, b):
    import torch
    import torch.nn.functional as F
    return float(F.cosine_similarity(torch.from_numpy(a)[None], torch.from_numpy(b)[None])[0])


# ---------------------------------------------------------------- ASR and error rates
def wer_seedtts(hypo, truth, lang):
    """Seed-TTS-eval run_wer.py::process_one (en / zh)."""
    from jiwer import compute_measures
    from zhon.hanzi import punctuation
    for x in punctuation + string.punctuation:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")
    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")
    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    else:
        truth = truth.lower()
        hypo = hypo.lower()
    return compute_measures(truth, hypo)["wer"]


def wer_multilingual(hypo, truth, lang):
    """ko / ja / de (not part of Seed-TTS-eval): drop Unicode punctuation and symbols, NFKC, lower case;
    ko / ja -> CER over characters without whitespace, de -> WER over words."""
    from jiwer import compute_measures

    def norm(s):
        s = unicodedata.normalize("NFKC", s).lower()
        s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
        return " ".join(s.split())
    truth, hypo = norm(truth), norm(hypo)
    if lang in ("ko", "ja"):
        truth = " ".join(truth.replace(" ", ""))
        hypo = " ".join(hypo.replace(" ", ""))
    return compute_measures(truth, hypo)["wer"] if hypo else 1.0


class ASR:
    """Whisper-large-v3 (en, ko, ja, de, and zh when forced) or Paraformer-zh (zh)."""

    def __init__(self, lang, whisper_for_zh=False):
        self.lang = lang
        if lang in WHISPER_LANG or whisper_for_zh:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(DEV)
            self.whisper = True
        else:
            from funasr import AutoModel
            self.m = AutoModel(model="paraformer-zh", disable_update=True)
            self.whisper = False

    def __call__(self, path):
        import torch
        if self.whisper:
            import scipy.signal
            import soundfile as sf
            wav, sr = sf.read(path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            feats = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
            ids = self.proc.get_decoder_prompt_ids(language=WHISPER_LANG.get(self.lang, "chinese"), task="transcribe")
            with torch.no_grad():
                pred = self.m.generate(feats, forced_decoder_ids=ids)
            text = self.proc.batch_decode(pred, skip_special_tokens=True)[0]
            if self.lang == "zh":
                import zhconv
                text = zhconv.convert(text, "zh-cn")
            return text
        import zhconv
        with torch.no_grad():
            res = self.m.generate(input=path, batch_size_s=300)
        return zhconv.convert(res[0]["text"], "zh-cn")


class SenseVoice:
    """Second ASR for ko / ja (FunASR iic/SenseVoiceSmall, language forced, ITN on)."""

    def __init__(self, lang):
        from funasr import AutoModel
        self.lang = lang
        self.m = AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device=DEV)

    def __call__(self, path):
        import torch
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        with torch.no_grad():
            res = self.m.generate(input=path, language=self.lang, use_itn=True, batch_size_s=300)
        return rich_transcription_postprocess(res[0]["text"])


# ---------------------------------------------------------------- scorer
class Scorer:
    """Adds the requested metric columns to <wav_dir>/_metrics/metrics.csv."""

    def __init__(self, meta, wav_dir, lang, out_dir=None, cache_dir=None, wav_name="{utt}.wav"):
        import soundfile as sf
        self.wav_dir, self.lang, self.wav_name = Path(wav_dir), lang, wav_name
        self.out_dir = Path(out_dir) if out_dir else self.wav_dir / "_metrics"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Path(cache_dir) if cache_dir else DATA / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.rows = [r for r in read_rows(meta) if self.wav(r).exists()]
        print(f"{len(self.rows)} wavs found in {self.wav_dir}")
        self.csv = self.out_dir / "metrics.csv"
        df = pd.DataFrame({"utt_id": [r["utt"] for r in self.rows]})
        if self.csv.exists():  # resume: keep already computed columns if the utterance set matches
            old = pd.read_csv(self.csv, encoding="utf-8")
            if list(old["utt_id"]) == list(df["utt_id"]):
                df = old
        df["dur_gen_s"] = [sf.info(str(self.wav(r))).duration for r in self.rows]
        df["dur_prompt_s"] = [sf.info(r["prompt_wav"]).duration for r in self.rows]
        self.df = df

    def wav(self, r):
        return self.wav_dir / self.wav_name.format(utt=r["utt"])

    def save(self):
        self.df.to_csv(self.csv, index=False, encoding="utf-8")

    @staticmethod
    def release(model):
        import torch
        del model
        torch.cuda.empty_cache()

    def add_wer(self):
        asr = ASR(self.lang)
        hyps = [asr(str(self.wav(r))) for r in self.rows]
        score = wer_seedtts if self.lang in ("en", "zh") else wer_multilingual
        self.df["wer"], self.df["hyp"] = [score(h, r["text"], self.lang) for h, r in zip(hyps, self.rows)], hyps
        self.save()
        self.release(asr)

    def add_wer_whisper_zh(self):
        asr = ASR("zh", whisper_for_zh=True)
        hyps = [asr(str(self.wav(r))) for r in self.rows]
        self.df["wer_wh"], self.df["hyp_wh"] = [wer_seedtts(h, r["text"], "zh") for h, r in zip(hyps, self.rows)], hyps
        self.save()
        self.release(asr)

    def add_cer_sensevoice(self):
        asr = SenseVoice(self.lang)
        hyps = [asr(str(self.wav(r))) for r in self.rows]
        self.df["wer_sv"], self.df["hyp_sv"] = [wer_multilingual(h, r["text"], self.lang) for h, r in zip(hyps, self.rows)], hyps
        self.save()
        self.release(asr)

    def add_similarity(self, name):
        emb = EMBEDDERS[name]()
        pc_path = self.cache / f"prompt_{name}.npz"
        pc = dict(np.load(pc_path)) if pc_path.exists() else {}
        new_prompt = False
        store, s_full, s_c6 = {}, [], []
        for r in self.rows:
            pk = os.path.normcase(os.path.abspath(r["prompt_wav"])).replace("\\", "/")
            if pk not in pc:
                pc[pk] = emb(load16(r["prompt_wav"]))
                new_prompt = True
            w = load16(str(self.wav(r)))
            g, g6 = emb(w), emb(centre_crop(w))
            store[f"gen/{r['utt']}"], store[f"gen_c6/{r['utt']}"] = g, g6
            s_full.append(cos(g, pc[pk]))
            s_c6.append(cos(g6, pc[pk]))
        self.df[f"sim_{name}"], self.df[f"sim_{name}_c6"] = s_full, s_c6
        np.savez(self.out_dir / f"emb_{name}.npz", **store)
        if new_prompt:
            np.savez(pc_path, **pc)
        self.save()
        self.release(emb)

    def add_utmos(self):
        import librosa
        import torch
        um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
        sc = []
        for r in self.rows:
            wav, _ = librosa.load(str(self.wav(r)), sr=16000)
            with torch.no_grad():
                sc.append(float(um(torch.from_numpy(wav)[None].to(DEV), 16000)[0]))
        self.df["utmos"] = sc

    def run(self, metrics):
        df = self.df
        if "wer" in metrics and "wer" not in df:
            self.add_wer()
        if "werwh" in metrics and self.lang == "zh" and "wer_wh" not in df:
            self.add_wer_whisper_zh()
        if "cer2" in metrics and self.lang in ("ko", "ja") and "wer_sv" not in df:
            self.add_cer_sensevoice()
        for e in [m for m in metrics if m in EMBEDDERS and f"sim_{m}" not in self.df]:
            self.add_similarity(e)
        if "utmos" in metrics and "utmos" not in self.df:
            self.add_utmos()
        self.save()
        summ = {c: self.df[c].mean() for c in self.df.columns if c.startswith(("wer", "sim_", "utmos"))}
        print("N =", len(self.df), {k: round(v, 4) for k, v in summ.items()})


# ---------------------------------------------------------------- prosody / spectral features
def syllables(text, lang):
    if lang == "zh":
        return max(1, len(re.findall(r"[\u4e00-\u9fff]", text)))
    if lang == "ko":  # Hangul syllable blocks
        return max(1, len(re.findall(r"[\uac00-\ud7a3]", text)))
    if lang == "ja":  # kana + kanji (rough mora proxy)
        return max(1, len(re.findall(r"[\u3041-\u309f\u30a0-\u30ff\u4e00-\u9fff]", text)))
    if lang == "de":
        return max(1, sum(max(1, len(re.findall(r"[aeiouyäöü]+", w.lower())))
                          for w in re.findall(r"[A-Za-zÄÖÜäöüß']+", text)))
    n = 0
    for w in re.findall(r"[A-Za-z']+", text):
        n += max(1, len(re.findall(r"[aeiouy]+", w.lower())))
    return max(1, n)


def acoustic_features(path, text, lang):
    import librosa
    import pyworld
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    w24 = librosa.resample(w, orig_sr=sr, target_sr=24000) if sr != 24000 else w
    hop = 480  # 20 ms frames for the energy VAD
    fr = librosa.util.frame(np.pad(w24, (0, hop)), frame_length=hop, hop_length=hop)
    rms = np.sqrt((fr ** 2).mean(0) + 1e-12)
    active = rms > rms.max() * 10 ** (-40 / 20)
    act_sec = active.sum() * hop / 24000
    w16 = librosa.resample(w24, orig_sr=24000, target_sr=16000).astype(np.float64)
    f0, t = pyworld.dio(w16, 16000, frame_period=5.0)
    f0 = pyworld.stonemask(w16, f0, t, 16000)
    v = f0[f0 > 0]
    st = 12 * np.log2(v / 100.0) if len(v) else np.array([np.nan])
    S = np.abs(librosa.stft(w24, n_fft=1024, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=24000, n_fft=1024)
    act_s = np.repeat(active, 480 // 256 + 1)[: S.shape[1]] if S.shape[1] else active
    act_s = np.resize(act_s, S.shape[1]).astype(bool)
    Sa = S[:, act_s] if act_s.any() else S
    hf = Sa[(freqs >= 4000) & (freqs < 8000)].sum() / (Sa[freqs < 8000].sum() + 1e-12)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=Sa, sr=24000, n_mels=80, fmax=8000) + 1e-10)
    flux = np.linalg.norm(np.diff(mel, axis=1), axis=0)
    return dict(f0_mean_st=float(np.mean(st)), f0_sd_st=float(np.std(st)), voiced_frac=float(len(v) / max(1, len(f0))),
                active_sec=act_sec, rate_active=syllables(text, lang) / max(act_sec, 1e-3),
                hf_ratio=float(hf), flux_var=float(np.var(flux)) if len(flux) else np.nan)


def write_acoustic(meta, wav_dir, lang):
    wav_dir = Path(wav_dir)
    out = wav_dir / "_metrics" / "acoustic.csv"
    if out.exists():
        return
    rows = [dict(utt_id=r["utt"], **acoustic_features(wav_dir / f"{r['utt']}.wav", r["text"], lang))
            for r in read_rows(meta) if (wav_dir / f"{r['utt']}.wav").exists()]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(wav_dir, len(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "zh", "ko", "ja", "de"])
    ap.add_argument("--metrics", default="wer,wavlm,ecapa,campp,utmos")
    ap.add_argument("--acoustic", action="store_true", help="prosody / spectral features instead of the metrics")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--wav_name", default="{utt}.wav", help="file name pattern inside wav_dir")
    a = ap.parse_args(argv)
    if a.acoustic:
        write_acoustic(resolve(a.meta), resolve(a.wav_dir), a.lang)
    else:
        Scorer(resolve(a.meta), resolve(a.wav_dir), a.lang, a.out_dir, a.cache_dir, a.wav_name).run(a.metrics.split(","))


if __name__ == "__main__":
    main()
