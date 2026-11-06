"""Per-utterance metrics for a directory of generated wavs.

Metrics
  wer      : seed-tts-eval protocol (run_wer.py): en Whisper-large-v3 (HF transformers, fp32,
             scipy resample to 16k), zh Paraformer-zh (FunASR) + zhconv. Normalisation copied
             verbatim from seed-tts-eval/run_wer.py::process_one.
  wavlm    : seed-tts-eval SIM model, imported from seed-tts-eval/thirdparty/UniSpeech
             (ECAPA_TDNN_SMALL + s3prl wavlm_large, wavlm_large_finetune.pth). librosa load,
             torchaudio Resample to 16k, as in verification.py.
  ecapa    : speechbrain/spkrec-ecapa-voxceleb
  campp    : 3D-Speaker CAM++ (modelscope iic/speech_campplus_sv_zh-cn_16k-common)
  ko/ja/de : (E8'-5, MiniMax) Whisper-large-v3 with forced language; ko/ja CER, de WER (process_multi);
             cer2 = second ASR SenseVoice-Small for ko/ja -> column wer_sv
  utmos    : UTMOS22 strong (tarepan/SpeechMOS v1.2.0 torch.hub port of the official ckpt)

SIM-o = cos(gen, prompt_wav). Generated wavs from all three repos contain only the target
segment (prompt already removed), which is asserted by checking duration < prompt+target.
Each SIM is computed on the full wav and on a 6 s centre crop (whole wav if shorter).

Outputs (in --out_dir):
  metrics.csv            utt_id, wer, sim_<emb>, sim_<emb>_c6, utmos, dur_gen_s, dur_prompt_s, ...
  emb_<emb>.npz          gen/<utt>, gen_c6/<utt> embeddings (float32)
Prompt embeddings are cached in --cache_dir/prompt_<emb>.npz.
"""
import argparse
import os
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SEEDTTS = ROOT / "repos" / "seed-tts-eval"
SV_DIR = SEEDTTS / "thirdparty" / "UniSpeech" / "downstreams" / "speaker_verification"
WAVLM_CKPT = ROOT / "data" / "ckpt" / "wavlm_large_finetune.pth"
DEV = "cuda:0"
CROP_S = 6.0


def read_meta(path):
    rows = []
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        f = l.split("|")
        rows.append(dict(utt=f[0], prompt_text=f[1], prompt_wav=f[2], text=f[3]))
    return rows


def load16(path):
    import librosa
    from torchaudio.transforms import Resample
    wav, sr = librosa.load(path, sr=None, mono=False)
    if wav.ndim == 2:
        wav = wav[0]
    w = torch.from_numpy(wav).unsqueeze(0).float()
    if sr != 16000:
        w = Resample(orig_freq=sr, new_freq=16000)(w)
    return w  # [1, T] 16 kHz


def centre_crop(w, sec=CROP_S, sr=16000):
    n = int(sec * sr)
    if w.shape[-1] <= n:
        return w
    s = (w.shape[-1] - n) // 2
    return w[..., s:s + n]


# ---------------- embedders ----------------
class WavLMSV:
    name = "wavlm"

    def __init__(self):
        sys.path.insert(0, str(SV_DIR))
        from models.ecapa_tdnn import ECAPA_TDNN_SMALL  # official seed-tts-eval code
        sys.path.pop(0)
        m = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        sd = torch.load(WAVLM_CKPT, map_location="cpu")
        m.load_state_dict(sd["model"], strict=False)
        self.m = m.to(DEV).eval()

    @torch.no_grad()
    def __call__(self, w):
        return self.m(w.to(DEV))[0].float().cpu().numpy()


class EcapaSB:
    name = "ecapa"

    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self.m = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ROOT / "ckpt" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": DEV}, local_strategy=LocalStrategy.COPY)

    @torch.no_grad()
    def __call__(self, w):
        return self.m.encode_batch(w.to(DEV))[0, 0].float().cpu().numpy()


class CamPP:
    name = "campp"

    def __init__(self):
        from modelscope.pipelines import pipeline
        self.p = pipeline(task="speaker-verification",
                          model="iic/speech_campplus_sv_zh-cn_16k-common",
                          model_revision="v1.0.0")

    @torch.no_grad()
    def __call__(self, w):
        out = self.p([w[0].numpy(), w[0].numpy()], output_emb=True)
        return np.asarray(out["embs"][0], dtype=np.float32)


class ERes2NetEn(CamPP):
    """3D-Speaker ERes2Net trained on VoxCeleb2 dev (English, 5,994 spk), VoxCeleb1-O EER 0.83%."""
    name = "eres2net"

    def __init__(self):
        from modelscope.pipelines import pipeline
        self.p = pipeline(task="speaker-verification", model="iic/speech_eres2net_sv_en_voxceleb_16k",
                          model_revision="v1.0.3")


EMBEDDERS = {"wavlm": WavLMSV, "ecapa": EcapaSB, "campp": CamPP, "eres2net": ERes2NetEn}


def cos(a, b):
    return float(F.cosine_similarity(torch.from_numpy(a)[None], torch.from_numpy(b)[None])[0])


# ---------------- WER (seed-tts-eval) ----------------
def process_one(hypo, truth, lang):
    from jiwer import compute_measures
    from zhon.hanzi import punctuation
    punctuation_all = punctuation + string.punctuation
    for x in punctuation_all:
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
    m = compute_measures(truth, hypo)
    return m["wer"]


def process_multi(hypo, truth, lang):
    """E8' -5 (MiniMax ko/ja/de; not part of seed-tts-eval): drop Unicode punctuation/symbols, NFKC, lower-case.
    ko/ja -> CER over characters with whitespace removed; de -> WER over words."""
    import unicodedata
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


WHISPER_LANG = {"en": "english", "ko": "korean", "ja": "japanese", "de": "german"}


class SenseVoice:
    """Second ASR for ko/ja (FunASR iic/SenseVoiceSmall, language forced, ITN on)."""

    def __init__(self, lang):
        from funasr import AutoModel
        self.lang = lang
        self.m = AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device=DEV)

    @torch.no_grad()
    def __call__(self, path):
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        res = self.m.generate(input=path, language=self.lang, use_itn=True, batch_size_s=300)
        return rich_transcription_postprocess(res[0]["text"])


class ASR:
    def __init__(self, lang):
        self.lang = lang
        if lang in WHISPER_LANG:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(DEV)
        else:
            from funasr import AutoModel
            self.m = AutoModel(model="paraformer-zh", disable_update=True)

    @torch.no_grad()
    def __call__(self, path):
        if self.lang in WHISPER_LANG:
            import scipy.signal
            import soundfile as sf
            wav, sr = sf.read(path)
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            feats = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
            ids = self.proc.get_decoder_prompt_ids(language=WHISPER_LANG[self.lang], task="transcribe")
            pred = self.m.generate(feats, forced_decoder_ids=ids)
            return self.proc.batch_decode(pred, skip_special_tokens=True)[0]
        import zhconv
        res = self.m.generate(input=path, batch_size_s=300)
        return zhconv.convert(res[0]["text"], "zh-cn")


def utmos_model():
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--lang", required=True, choices=["en", "zh", "ko", "ja", "de"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--metrics", default="wer,wavlm,ecapa,campp,utmos")
    ap.add_argument("--cache_dir", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--wav_name", default="{utt}.wav", help="filename pattern inside wav_dir")
    a = ap.parse_args()

    metrics = a.metrics.split(",")
    wav_dir = Path(a.wav_dir)
    out_dir = Path(a.out_dir) if a.out_dir else wav_dir / "_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    rows = [r for r in read_meta(a.meta) if (wav_dir / a.wav_name.format(utt=r["utt"])).exists()]
    print(f"{len(rows)} wavs found in {wav_dir}")
    import soundfile as sf
    csv_path = out_dir / "metrics.csv"
    df = pd.DataFrame({"utt_id": [r["utt"] for r in rows]})
    if csv_path.exists():  # resume: keep already-computed metric columns if utt set matches
        old = pd.read_csv(csv_path, encoding="utf-8")
        if list(old["utt_id"]) == list(df["utt_id"]):
            df = old
    df["dur_gen_s"] = [sf.info(str(wav_dir / a.wav_name.format(utt=r["utt"]))).duration for r in rows]
    df["dur_prompt_s"] = [sf.info(r["prompt_wav"]).duration for r in rows]
    save = lambda: df.to_csv(csv_path, index=False, encoding="utf-8")  # noqa: E731

    if "wer" in metrics and "wer" not in df:
        asr = ASR(a.lang)
        hyps, wers = [], []
        for r in rows:
            h = asr(str(wav_dir / a.wav_name.format(utt=r["utt"])))
            hyps.append(h)
            wers.append(process_one(h, r["text"], a.lang) if a.lang in ("en", "zh")
                        else process_multi(h, r["text"], a.lang))
        df["wer"], df["hyp"] = wers, hyps
        save()
        del asr
        torch.cuda.empty_cache()

    if "werwh" in metrics and a.lang == "zh" and "wer_wh" not in df:
        # E10-2: Whisper-large-v3 CER for zh next to the official Paraformer-zh (en: Whisper-large-v3 is the official ASR)
        asr = ASR("en")
        asr.lang = "zh"
        hyps = []
        for r in rows:
            import scipy.signal
            import soundfile as sf_
            import zhconv
            wav, sr = sf_.read(str(wav_dir / a.wav_name.format(utt=r["utt"])))
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            feats = asr.proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEV)
            ids = asr.proc.get_decoder_prompt_ids(language="chinese", task="transcribe")
            pred = asr.m.generate(feats, forced_decoder_ids=ids)
            hyps.append(zhconv.convert(asr.proc.batch_decode(pred, skip_special_tokens=True)[0], "zh-cn"))
        df["wer_wh"], df["hyp_wh"] = [process_one(hh, r["text"], "zh") for hh, r in zip(hyps, rows)], hyps
        save()
        del asr
        torch.cuda.empty_cache()

    if "cer2" in metrics and a.lang in ("ko", "ja") and "wer_sv" not in df:
        asr = SenseVoice(a.lang)
        hyps = [asr(str(wav_dir / a.wav_name.format(utt=r["utt"]))) for r in rows]
        df["wer_sv"], df["hyp_sv"] = [process_multi(h, r["text"], a.lang) for h, r in zip(hyps, rows)], hyps
        save()
        del asr
        torch.cuda.empty_cache()

    for e in [m for m in metrics if m in EMBEDDERS and f"sim_{m}" not in df]:
        emb = EMBEDDERS[e]()
        pc_path = cache / f"prompt_{e}.npz"
        pc = dict(np.load(pc_path)) if pc_path.exists() else {}
        new_prompt = False
        store, s_full, s_c6 = {}, [], []
        for r in rows:
            key = os.path.normcase(os.path.abspath(r["prompt_wav"]))
            pk = key.replace("\\", "/")
            if pk not in pc:
                pc[pk] = emb(load16(r["prompt_wav"]))
                new_prompt = True
            w = load16(str(wav_dir / a.wav_name.format(utt=r["utt"])))
            g, g6 = emb(w), emb(centre_crop(w))
            store[f"gen/{r['utt']}"], store[f"gen_c6/{r['utt']}"] = g, g6
            s_full.append(cos(g, pc[pk]))
            s_c6.append(cos(g6, pc[pk]))
        df[f"sim_{e}"], df[f"sim_{e}_c6"] = s_full, s_c6
        np.savez(out_dir / f"emb_{e}.npz", **store)
        if new_prompt:
            np.savez(pc_path, **pc)
        save()
        del emb
        torch.cuda.empty_cache()

    if "utmos" in metrics and "utmos" not in df:
        um = utmos_model()
        sc = []
        for r in rows:
            import librosa
            wav, sr = librosa.load(str(wav_dir / a.wav_name.format(utt=r["utt"])), sr=16000)
            with torch.no_grad():
                sc.append(float(um(torch.from_numpy(wav)[None].to(DEV), 16000)[0]))
        df["utmos"] = sc

    save()
    summ = {c: df[c].mean() for c in df.columns if c.startswith(("wer", "sim_", "utmos"))}
    print("N =", len(df), {k: round(v, 4) for k, v in summ.items()})


if __name__ == "__main__":
    main()
