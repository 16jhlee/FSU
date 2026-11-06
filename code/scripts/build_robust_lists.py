"""v8 §2.3 prompt robustness lists (CosyVoice2, en 100).
Seed-TTS-eval prompts are 3-6 s, so prompts come from LibriSpeech test-clean (public, 40 speakers): 100 utterances with
>= 10.5 s of speech after trimming leading/trailing silence (speakers cycled), cropped from the speech start to
1 / 3 / 10 s (clean), plus the 3 s crop mixed with MUSAN noise at SNR 20 / 10 dB (noise file and offset chosen with a
fixed seed per utterance, SNR on the speech RMS). Prompt texts = Whisper-large-v3 transcripts of each clean crop (the
noisy 3 s prompts reuse the clean 3 s transcript). Target texts = Seed-TTS en_all rows 0..99.
SIM reference for every condition = the clean 10 s crop of the same speaker utterance (data/lists/rb_<cond>_ref.meta.lst),
so conditions are compared against one reference.
python scripts/build_robust_lists.py crops   (env_eval, GPU: Whisper)   ->  rb_len1 / rb_len3 / rb_len10
python scripts/build_robust_lists.py noise   (after MUSAN is extracted)  ->  rb_snr20 / rb_snr10
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LS = ROOT / "data/raw/librispeech/LibriSpeech/test-clean"
MUSAN = ROOT / "data/raw/musan/musan/noise"
OUT = ROOT / "data/raw/robust_prompts"
LISTS = ROOT / "data/lists"
SR = 16000


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) % (2**31 - 1)


def texts():
    rows = [l.split("|") for l in (LISTS / "en_all.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r[3] for r in rows[:100]]


def write_list(name, items, ref):
    TX = texts()
    gen = [f"{name}_{i:03d}|{it['ptext']}|{it['wav']}|{TX[i]}" for i, it in enumerate(items)]
    rf = [f"{name}_{i:03d}|{it['ptext']}|{r['wav']}|{TX[i]}" for i, (it, r) in enumerate(zip(items, ref))]
    (LISTS / f"{name}.meta.lst").write_text("\n".join(gen) + "\n", encoding="utf-8")
    (LISTS / f"{name}_ref.meta.lst").write_text("\n".join(rf) + "\n", encoding="utf-8")


def crops():
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    sys.path.insert(0, str(Path(__file__).parent))
    import gpu_lock
    spk = sorted(d for d in LS.iterdir() if d.is_dir())
    cand = {s.name: [] for s in spk}
    for s in spk:
        for f in sorted(s.rglob("*.flac")):
            if sf.info(str(f)).duration >= 11.5:
                cand[s.name].append(f)
    picks, i = [], 0
    while len(picks) < 100:
        s = spk[i % len(spk)].name
        j = i // len(spk)
        if j < len(cand[s]):
            picks.append(cand[s][j])
        i += 1
        assert i < 10000
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for n, f in enumerate(picks):
        y, _ = librosa.load(str(f), sr=SR)
        y, _ = librosa.effects.trim(y, top_db=30)
        if len(y) < 10.5 * SR:
            continue
        meta[n] = dict(src=str(f.relative_to(ROOT)), crops={})
        for L in (1, 3, 10):
            p = OUT / f"len{L}" / f"rb_{n:03d}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), y[: L * SR], SR, subtype="PCM_16")
            meta[n]["crops"][L] = str(p)
    keep = sorted(meta)[:100]
    assert len(keep) == 100, len(keep)
    with gpu_lock.Lock("eval_job"):
        proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")
        ids = proc.get_decoder_prompt_ids(language="english", task="transcribe")
        for n in keep:
            for L in (1, 3, 10):
                y, _ = sf.read(meta[n]["crops"][L])
                feats = proc(y, sampling_rate=SR, return_tensors="pt").input_features.to("cuda")
                with torch.no_grad():
                    txt = proc.batch_decode(model.generate(feats, forced_decoder_ids=ids), skip_special_tokens=True)[0].strip()
                meta[n].setdefault("text", {})[L] = txt
    (OUT / "meta.json").write_text(json.dumps({k: meta[k] for k in keep}, ensure_ascii=False, indent=1), encoding="utf-8")
    ref = [dict(wav=meta[n]["crops"][10]) for n in keep]
    for L in (1, 3, 10):
        write_list(f"rb_len{L}", [dict(wav=meta[n]["crops"][L], ptext=meta[n]["text"][L]) for n in keep], ref)
    print("lists written: rb_len1/3/10")


def noise():
    import librosa
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    nfiles = sorted(MUSAN.rglob("*.wav"))
    assert nfiles, MUSAN
    keep = sorted(meta, key=int)
    ref = [dict(wav=meta[n]["crops"]["10"]) for n in keep]
    for snr in (20, 10):
        items = []
        for n in keep:
            y, _ = sf.read(meta[n]["crops"]["3"])
            rng = np.random.default_rng(h(f"rb|noise|{n}"))
            nz, _ = librosa.load(str(nfiles[rng.integers(len(nfiles))]), sr=SR)
            while len(nz) < len(y):
                nz = np.concatenate([nz, nz])
            off = rng.integers(0, len(nz) - len(y) + 1)
            nz = nz[off:off + len(y)]
            g = np.sqrt(np.mean(y ** 2) / (np.mean(nz ** 2) + 1e-12) / 10 ** (snr / 10))
            x = y + g * nz
            x = x / max(1.0, np.abs(x).max() / 0.99)
            p = OUT / f"snr{snr}" / f"rb_{int(n):03d}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), x, SR, subtype="PCM_16")
            items.append(dict(wav=str(p), ptext=meta[n]["text"]["3"]))
        write_list(f"rb_snr{snr}", items, ref)
    print("lists written: rb_snr20/10")


if __name__ == "__main__":
    {"crops": crops, "noise": noise}[sys.argv[1]]()
