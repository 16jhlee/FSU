"""Test sets and the builders of their lists.

A test list has one `utt|prompt_text|prompt_wav|text` line per utterance (Seed-TTS-eval format, absolute prompt paths),
data/lists/<name>.meta.lst, plus <name>.zipvoice.tsv for ZipVoice. Build all lists after downloading the data:
    python -m fsu.datasets build            # every list that needs no GPU
    python -m fsu.datasets build-robust     # prompt-length / noisy-prompt conditions (Whisper transcripts, GPU)

Sets
  seedtts_{en,zh,hard}_subset   analysis subsets: 300 / 300 / 100 utterances drawn with seed 20261017
  seedtts_{en,zh}_subset100     first 100 utterances of the subsets (multi-seed runs)
  seedtts_{en,zh,hard}_full     complete Seed-TTS-eval test sets (1,088 / 2,020 / 400)
  librispeech_pc                LibriSpeech-PC test-clean cross-sentence pairs of the F5-TTS repository (1,127)
  long_{en,zh}                  three subset sentences joined into one long text, prompt of the first (100 each)
  minimax_{de,ja,ko}            MiniMax multilingual test set (100 each)
  prompt_{1s,3s,10s}            LibriSpeech test-clean prompts cropped to 1 / 3 / 10 s, Seed-TTS en texts (100 each)
  prompt_snr{20,10}dB           the 3 s prompts mixed with MUSAN noise
                                (SIM of all prompt_* sets is measured against the clean 10 s crop: <name>_ref lists)
  crossling_<a>2<b>             prompt language a, text language b (100 each)
"""
import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fsu.common import DATA, REPOS, read_list, seed_of, write_list

LISTS = DATA / "lists"
RAW = DATA / "raw"
SEEDTTS = RAW / "seedtts" / "seedtts_testset"
SUBSET_SEED = 20261017


@dataclass(frozen=True)
class TestSet:
    name: str
    lang: str
    reference: str = None  # list used for SIM when the prompts differ from the reference speaker audio
    group: str = None  # sets drawn from the same source share one token cache (tokens are seeded per utterance)
    text_frontend: bool = True  # False: raw text, the zh/en text normalisers misfire on ko/ja/de

    @property
    def token_group(self):
        return self.group or self.name

    @property
    def path(self):
        return LISTS / f"{self.name}.meta.lst"

    @property
    def eval_path(self):
        return LISTS / f"{self.reference or self.name}.meta.lst"

    @property
    def zipvoice_path(self):
        return LISTS / f"{self.name}.zipvoice.tsv"

    def rows(self):
        return read_list(self.path)

    def __len__(self):
        return len(self.rows())


SETS = {s.name: s for s in [
    *[TestSet(f"seedtts_{p}_{n}", "zh" if p == "hard" else p, group=f"seedtts_{p}")
      for p, n in [("en", "subset"), ("zh", "subset"), ("hard", "subset"), ("en", "subset100"), ("zh", "subset100"),
                   ("en", "subset20"), ("en", "subset3"), ("hard", "full3"),
                   ("en", "full"), ("zh", "full"), ("hard", "full")]],
    TestSet("librispeech_pc", "en"),
    TestSet("long_en", "en"), TestSet("long_zh", "zh"),
    *[TestSet(f"minimax_{c}", c, text_frontend=False) for c in ("de", "ja", "ko")],
    *[TestSet(f"prompt_{c}", "en", reference=f"prompt_{c}_ref") for c in ("1s", "3s", "10s", "snr20dB", "snr10dB")],
    TestSet("crossling_zh2en", "en"), TestSet("crossling_en2zh", "zh"), TestSet("crossling_en2en", "en"),
    TestSet("crossling_zh2zh", "zh"), TestSet("crossling_en2ko", "ko", text_frontend=False),
    TestSet("crossling_ko2en", "en", text_frontend=False), TestSet("crossling_ko2ko", "ko", text_frontend=False),
]}


def get(name):
    return SETS[name]


# ---------------------------------------------------------------- Seed-TTS-eval
def make_subset(out=DATA / "subset_ids.txt"):
    """Fixed-seed subsets of Seed-TTS-eval: test-en 300, test-zh 300, hard 100."""
    def ids(path):
        return [line.split("|")[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sets = {"en": (ids(SEEDTTS / "en" / "meta.lst"), 300), "zh": (ids(SEEDTTS / "zh" / "meta.lst"), 300),
            "hard": (ids(SEEDTTS / "zh" / "hardcase.lst"), 100)}
    lines = [f"# seed={SUBSET_SEED}; format: <set>\t<utt_id>"]
    for name, (all_ids, n) in sets.items():
        pick = random.Random(f"{SUBSET_SEED}-{name}").sample(sorted(all_ids), n)
        lines += [f"{name}\t{u}" for u in pick]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def seedtts_list(name, part, n=None, subset=DATA / "subset_ids.txt"):
    """Seed-TTS-eval rows of `part` (en / zh / hard): the subset in subset_ids.txt order, or the whole set in meta order
    (subset=None); the first n rows if n is given. Writes the meta list and the ZipVoice tsv."""
    meta = SEEDTTS / ("zh/hardcase.lst" if part == "hard" else f"{part}/meta.lst")
    rows = {}
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.strip():
            f = line.split("|")
            rows[f[0]] = f
    if subset is None:
        ids = list(rows)
    else:
        ids = [line.split("\t")[1] for line in Path(subset).read_text(encoding="utf-8").splitlines()
               if line and not line.startswith("#") and line.split("\t")[0] == part]
    ids = ids[:n] if n else ids
    LISTS.mkdir(parents=True, exist_ok=True)
    with open(LISTS / f"{name}.meta.lst", "w", encoding="utf-8", newline="\n") as fm, \
            open(LISTS / f"{name}.zipvoice.tsv", "w", encoding="utf-8", newline="\n") as fz:
        for u in ids:
            f = rows[u]
            pw = str((meta.parent / f[2]).resolve())
            fm.write("|".join([u, f[1], pw, f[3]] + f[4:]) + "\n")
            fz.write("\t".join([u, f[1], pw, f[3]]) + "\n")


def long_lists():
    """Each long text joins three subset sentences of the same language (rows i, i+100, i+200) with the prompt of row i."""
    for lang, sep in [("en", " "), ("zh", "")]:
        R = get(f"seedtts_{lang}_subset").rows()
        write_list(LISTS / f"long_{lang}.meta.lst",
                   [(f"long_{lang}_{i:03d}", R[i][1], R[i][2], sep.join(R[j][3] for j in (i, i + 100, i + 200)))
                    for i in range(100)])


def minimax_lists():
    """MiniMax multilingual test set (ko / ja / de): utt = mm_<lang>_<line:03d>_<female|male>. The mp3 prompts (Common
    Voice) are converted to mono wav at their native rate, because the frontends and evaluators read wav."""
    import librosa
    import soundfile as sf
    mm, out_wav = RAW / "minimax", RAW / "minimax_wav"
    langs = {"ko": "korean", "ja": "japanese", "de": "german"}
    ptext = {}
    for line in (mm / "speaker" / "prompt_text.txt").read_text(encoding="utf-8").splitlines():
        if "|" in line:
            f, t = line.split("|", 1)
            ptext[f.strip()] = t.strip()
    for code, lang in langs.items():
        spk = {}
        for d in (mm / "speaker" / lang).iterdir():
            mp3 = next(d.glob("*.mp3"))
            wav = out_wav / code / (mp3.stem + ".wav")
            if not wav.exists():
                wav.parent.mkdir(parents=True, exist_ok=True)
                y, sr = librosa.load(str(mp3), sr=None, mono=True)
                sf.write(str(wav), y, sr, subtype="PCM_16")
            spk[d.name] = (ptext[mp3.name], wav)
        rows = []
        for i, line in enumerate((mm / "text" / f"{lang}.txt").read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            s, text = line.split("|", 1)
            pt, wav = spk[s.strip()]
            rows.append((f"mm_{code}_{i:03d}_{s.strip().split('_')[-1]}", pt, str(wav), text.strip()))
        write_list(LISTS / f"minimax_{code}.meta.lst", rows)


def librispeech_pc_list(librispeech_dir=RAW / "librispeech"):
    """LibriSpeech-PC test-clean cross-sentence pairs (F5-TTS repository list); prompts converted from flac to wav."""
    import soundfile as sf
    out = RAW / "librispeech_pc" / "prompt"
    out.mkdir(parents=True, exist_ok=True)
    pairs = [line.split("\t") for line in (REPOS / "F5-TTS" / "data" / "librispeech_pc_test_clean_cross_sentence.lst")
             .read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for pid, _, ptext, tid, _, ttext in pairs:
        spk, ch, _ = pid.split("-")
        dst = out / f"{pid}.wav"
        if not dst.exists():
            y, sr = sf.read(str(librispeech_dir / "LibriSpeech" / "test-clean" / spk / ch / f"{pid}.flac"))
            sf.write(str(dst), y, sr)
        rows.append((tid, ptext, str(dst), ttext))
    write_list(LISTS / "librispeech_pc.meta.lst", rows)


def crossling_lists():
    """Cross-lingual prompts: rows 0..99 of the en / zh subsets and of minimax_ko, prompt from one, text from another;
    same-language references with the identical prompts and texts."""
    en, zh, ko = (get(n).rows()[:100] for n in ("seedtts_en_subset", "seedtts_zh_subset", "minimax_ko"))
    for name, (P, T) in {"zh2en": (zh, en), "en2zh": (en, zh), "en2ko": (en, ko), "ko2en": (ko, en)}.items():
        write_list(LISTS / f"crossling_{name}.meta.lst",
                   [(f"xl_{name}_{i:03d}", p[1], p[2], t[3]) for i, (p, t) in enumerate(zip(P, T))])
    for src, name in [(en, "en2en"), (zh, "zh2zh"), (ko, "ko2ko")]:
        write_list(LISTS / f"crossling_{name}.meta.lst", [(f"xl_{name}_{i:03d}", r[1], r[2], r[3]) for i, r in enumerate(src)])


# ---------------------------------------------------------------- prompt robustness
class RobustPrompts:
    """Prompt-length and noisy-prompt conditions. Seed-TTS-eval prompts are 3-6 s, so prompts come from LibriSpeech
    test-clean (40 speakers): 100 utterances with >= 10.5 s of speech after trimming leading/trailing silence (speakers
    cycled), cropped from the speech start to 1 / 3 / 10 s, plus the 3 s crop mixed with MUSAN noise at SNR 20 / 10 dB
    (noise file and offset drawn with a fixed seed per utterance, SNR on the speech RMS). Prompt texts are Whisper-large-v3
    transcripts of each clean crop (the noisy prompts reuse the 3 s transcript); target texts are rows 0..99 of
    seedtts_en_subset. The SIM reference of every condition is the clean 10 s crop of the same utterance."""
    SR = 16000

    def __init__(self, librispeech_dir=RAW / "librispeech", musan_dir=RAW / "musan" / "musan" / "noise"):
        self.ls = Path(librispeech_dir) / "LibriSpeech" / "test-clean"
        self.musan = Path(musan_dir)
        self.out = RAW / "robust_prompts"

    def write(self, name, uid, items, ref):
        """<name>.meta.lst (generation) and <name>_ref.meta.lst (SIM reference), utterance ids <uid>_000 ..."""
        tx = [r[3] for r in get("seedtts_en_subset").rows()[:100]]
        write_list(LISTS / f"{name}.meta.lst", [(f"{uid}_{i:03d}", it["ptext"], it["wav"], tx[i]) for i, it in enumerate(items)])
        write_list(LISTS / f"{name}_ref.meta.lst",
                   [(f"{uid}_{i:03d}", it["ptext"], r["wav"], tx[i]) for i, (it, r) in enumerate(zip(items, ref))])

    def crops(self):
        import librosa
        import soundfile as sf
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from fsu.common import GpuLock
        sr = self.SR
        spk = sorted(d for d in self.ls.iterdir() if d.is_dir())
        cand = {s.name: [f for f in sorted(s.rglob("*.flac")) if sf.info(str(f)).duration >= 11.5] for s in spk}
        picks, i = [], 0
        while len(picks) < 100:
            s, j = spk[i % len(spk)].name, i // len(spk)
            if j < len(cand[s]):
                picks.append(cand[s][j])
            i += 1
            assert i < 10000
        meta = {}
        for n, f in enumerate(picks):
            y, _ = librosa.load(str(f), sr=sr)
            y, _ = librosa.effects.trim(y, top_db=30)
            if len(y) < 10.5 * sr:
                continue
            meta[n] = dict(src=str(f), crops={})
            for L in (1, 3, 10):
                p = self.out / f"len{L}" / f"rb_{n:03d}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), y[: L * sr], sr, subtype="PCM_16")
                meta[n]["crops"][L] = str(p)
        keep = sorted(meta)[:100]
        assert len(keep) == 100, len(keep)
        with GpuLock("eval_job"):
            proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to("cuda")
            ids = proc.get_decoder_prompt_ids(language="english", task="transcribe")
            for n in keep:
                for L in (1, 3, 10):
                    y, _ = sf.read(meta[n]["crops"][L])
                    feats = proc(y, sampling_rate=sr, return_tensors="pt").input_features.to("cuda")
                    with torch.no_grad():
                        txt = proc.batch_decode(model.generate(feats, forced_decoder_ids=ids), skip_special_tokens=True)[0].strip()
                    meta[n].setdefault("text", {})[L] = txt
        (self.out / "meta.json").write_text(json.dumps({k: meta[k] for k in keep}, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        ref = [dict(wav=meta[n]["crops"][10]) for n in keep]
        for L in (1, 3, 10):
            self.write(f"prompt_{L}s", f"rb_len{L}", [dict(wav=meta[n]["crops"][L], ptext=meta[n]["text"][L]) for n in keep], ref)

    def noisy(self):
        import librosa
        import soundfile as sf
        sr = self.SR
        meta = json.loads((self.out / "meta.json").read_text(encoding="utf-8"))
        nfiles = sorted(self.musan.rglob("*.wav"))
        assert nfiles, self.musan
        keep = sorted(meta, key=int)
        ref = [dict(wav=meta[n]["crops"]["10"]) for n in keep]
        for snr in (20, 10):
            items = []
            for n in keep:
                y, _ = sf.read(meta[n]["crops"]["3"])
                rng = np.random.default_rng(seed_of(f"rb|noise|{n}"))
                nz, _ = librosa.load(str(nfiles[rng.integers(len(nfiles))]), sr=sr)
                while len(nz) < len(y):
                    nz = np.concatenate([nz, nz])
                off = rng.integers(0, len(nz) - len(y) + 1)
                nz = nz[off:off + len(y)]
                g = np.sqrt(np.mean(y ** 2) / (np.mean(nz ** 2) + 1e-12) / 10 ** (snr / 10))
                x = y + g * nz
                x = x / max(1.0, np.abs(x).max() / 0.99)
                p = self.out / f"snr{snr}" / f"rb_{int(n):03d}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), x, sr, subtype="PCM_16")
                items.append(dict(wav=str(p), ptext=meta[n]["text"]["3"]))
            self.write(f"prompt_snr{snr}dB", f"rb_snr{snr}", items, ref)


def build_all():
    if not (DATA / "subset_ids.txt").exists():
        make_subset()
    for part in ("en", "zh", "hard"):
        seedtts_list(f"seedtts_{part}_subset", part)
        seedtts_list(f"seedtts_{part}_full", part, subset=None)
    seedtts_list("seedtts_en_subset100", "en", 100)
    seedtts_list("seedtts_zh_subset100", "zh", 100)
    seedtts_list("seedtts_en_subset20", "en", 20)
    seedtts_list("seedtts_en_subset3", "en", 3)
    seedtts_list("seedtts_hard_full3", "hard", 3, subset=None)
    long_lists()
    minimax_lists()
    librispeech_pc_list()
    crossling_lists()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "build-robust"])
    a = ap.parse_args(argv)
    if a.command == "build":
        build_all()
    else:
        rp = RobustPrompts()
        rp.crops()
        rp.noisy()
    print("lists in", LISTS)


if __name__ == "__main__":
    sys.exit(main())
