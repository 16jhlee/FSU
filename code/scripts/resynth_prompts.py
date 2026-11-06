"""Prompt -> mel -> vocoder resynthesis, for SIM-r (and later E6 vocoder upper bound).

--vocoder hift  : CosyVoice2 frontend mel (24 kHz, 80 bins, fmax 8000) -> CosyVoice2 HiFT  (env_cosy)
--vocoder vocos : Vocos mel (24 kHz, 100 bins, hop 256)  -> charactr/vocos-mel-24khz       (env_zip / env_f5dmo)
                  This is the feature used by ZipVoice (VocosFbank), F5-TTS and DMOSpeech2 (MelSpec 'vocos').
Input: one or more meta.lst files; every distinct prompt wav is resynthesised once.
Output: data/cache/prompt_resynth/<vocoder>/<sha1(prompt path)>.wav  + index.tsv (prompt_path \t out)
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]


def prompts(metas):
    seen = []
    for m in metas:
        for l in Path(m).read_text(encoding="utf-8").splitlines():
            if l.strip():
                p = l.split("|")[2]
                if p not in seen:
                    seen.append(p)
    return seen


def key(p):
    return hashlib.sha1(str(Path(p).resolve()).lower().encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocoder", choices=["hift", "vocos"], required=True)
    ap.add_argument("--meta", nargs="+", required=True)
    a = ap.parse_args()
    out = ROOT / "data" / "cache" / "prompt_resynth" / a.vocoder
    out.mkdir(parents=True, exist_ok=True)
    plist = prompts(a.meta)
    if a.vocoder == "hift":
        repo = ROOT / "repos" / "CosyVoice"
        sys.path[:0] = [str(repo), str(repo / "third_party" / "Matcha-TTS")]
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_cosy  # noqa: F401  (pins wetext resources, imports CosyVoice)
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.common import set_all_random_seed
        cv = AutoModel(model_dir=str(ROOT / "ckpt" / "CosyVoice2-0.5B"))
        sr = cv.sample_rate

        def synth(p):
            feat, _ = cv.frontend._extract_speech_feat(p)  # [1, T, 80]
            set_all_random_seed(int(key(p), 16) % (2**31 - 1))  # CV2 HiFT is stochastic
            with torch.inference_mode():
                wav, _ = cv.model.hift.inference(speech_feat=feat.transpose(1, 2).to(cv.model.device))
            return wav.cpu()
    else:
        from vocos import Vocos
        voc = Vocos.from_hparams(str(ROOT / "ckpt" / "vocos-mel-24khz" / "config.yaml"))
        voc.load_state_dict(torch.load(ROOT / "ckpt" / "vocos-mel-24khz" / "pytorch_model.bin", map_location="cpu"))
        voc = voc.eval().cuda()
        sr = 24000

        def synth(p):
            w, s = torchaudio.load(p)
            w = w[:1]
            if s != sr:
                w = torchaudio.functional.resample(w, s, sr)
            with torch.inference_mode():
                return voc.decode(voc.feature_extractor(w.cuda())).cpu()

    idx = open(out / "index.tsv", "a", encoding="utf-8")
    for p in plist:
        o = out / f"{key(p)}.wav"
        if o.exists():
            continue
        torchaudio.save(str(o), synth(p).float().clamp(-1, 1), sr)
        idx.write(f"{Path(p).resolve()}\t{o.name}\n")
        idx.flush()
    print(a.vocoder, len(plist), "prompts")


if __name__ == "__main__":
    main()
