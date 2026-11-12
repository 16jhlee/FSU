# Guidance at Pure Noise Erodes Speaker Similarity in Few-Step Token-Conditioned Flow-Matching TTS

[![arXiv](https://img.shields.io/badge/arXiv-2611.11179-b31b1b.svg)](https://arxiv.org/abs/2611.11179)

Jaeho Lee, Han-Lim Choi

Code and listening samples for the paper [arXiv:2611.11179](https://arxiv.org/abs/2611.11179)
([PDF](https://arxiv.org/pdf/2611.11179)), submitted to IEEE Transactions on Audio, Speech and Language Processing.

`code/` runs every experiment of the paper and writes per-utterance results; `audio_samples/` holds listening samples
(open `audio_samples/index.html`).

## Summary
Token-conditioned flow-matching TTS decoders run with two Euler steps keep their word error rate but lose one to six
points of speaker similarity. Most of that loss comes from classifier-free guidance on the first step, taken at pure
noise. Two training-free samplers remove it, each with three network evaluations (one fewer than the official two-step
sampler):

- **FSU** (first-step unguided): the first step uses the conditional velocity only, the second step keeps the official
  guidance. Recovers 84-91% of the two-step similarity loss on CosyVoice 2/3.
- **FSU-Mid**: one midpoint step with the same guidance interval. Recovers 77-85% on IndexTTS-2 and matches the
  ten-step teacher on CosyVoice 2 and GLM-TTS.

## Code layout
```
code/
├─ run.py                     python run.py list | show | generate | evaluate <experiment>
├─ configs/
│  ├─ methods.yaml            named sampling methods -> decoder-runner arguments
│  └─ experiments/*.yaml      the runs of each experiment (decoder x test set x method x seed)
├─ fsu/
│  ├─ decoders/               runners of CosyVoice 2/3, IndexTTS-2, GLM-TTS, F5-TTS / DMOSpeech 2 (one Decoder interface)
│  ├─ experiment.py           decoders, methods, runs, generation and evaluation queues
│  ├─ datasets.py             test sets and the builders of their lists
│  ├─ metrics.py              WER / CER, speaker similarity (WavLM, ECAPA, ERes2Net), UTMOS, prosody features
│  ├─ quality.py              UTMOSv2, DNSMOS, NISQA
│  ├─ probes/                 measurements on the networks themselves (closed-form checks, guidance magnitude and
│  │                          decay, first-step deviation, branches, one-step outputs, streaming latency)
│  ├─ assets.py               checkpoints and test data at pinned revisions
│  └─ common.py               paths, seeds, list I/O, conda environments, GPU lock
├─ patches/<repo>.diff        sampler-level patches to the upstream repositories (weights, losses, architectures unchanged)
├─ setup/                     setup_windows.ps1, setup_linux.sh
└─ env/                       pins.txt (upstream commits, model revisions), lock_<env>.txt, env_*.yml
```

## Experiments
| File | Content |
|---|---|
| `nfe_sweep` | step-count sweeps and the localisation of the few-step loss (Sec. III-A, Fig. 3, Table X, App. A) |
| `first_step_weight` | first-step weight w1 sweep (Fig. 4) and the guidance weight on both steps |
| `controls` | two-step controls over eight noise seeds (Fig. 5) |
| `step_schedules` | first-step length, guidance masks, threshold t*, schedules and solvers (Sec. V, App. E) |
| `unconditional_branch` | all-null / speaker-null / token-null branches (Sec. V-E, Table XIII) |
| `rule_comparison` | published guidance rules vs FSU / FSU-Mid on the analysis subsets (Tables I, IX) |
| `full_test_sets` | full Seed-TTS-eval sets and LibriSpeech-PC (Tables II, V) |
| `robustness` | languages, long text, streaming, short / noisy and cross-lingual prompts (Table VI) |
| `seeds` | additional noise seeds (Table IV) |
| `applicability` | F5-TTS and ZipVoice (Sec. VII) |

## Methods
The samplers are options of the decoder runners (repository time convention: t = 0 is noise).

| Method | Runner arguments (two steps unless noted) |
|---|---|
| Official CFG | `--nfe 2` |
| FSU (first-step unguided) | `--nfe 2 --cfg_interval 0.1,1` |
| FSU-Mid | `--nfe 1 --solver midpoint --cfg_interval 0.1,1` (3 network calls) |
| Teacher | `--nfe 10` (IndexTTS-2: `--nfe 25`) |
| TV-CFG (corrected) | `--nfe 2 --tv symmetric_fix` |
| Selective-CFG (input-text) | `--nfe 2 --cfg_branch speaker` |
| MAMBO-G | `--nfe 2 --mambo_alpha 4` |
| GLM-TTS all-null branch | `--cfg_branch allnull` |
| CosyVoice 2 streaming | add `--stream 1` |

Noise is seeded per utterance (`--noise utt:<k>`), and the LLM speech tokens (CosyVoice 2/3, GLM-TTS) and IndexTTS-2
GPT conditions are generated once per utterance and cached, so every sampler setting decodes the same tokens.

## Running
1. `powershell -ExecutionPolicy Bypass -File setup\setup_windows.ps1` (or `bash setup/setup_linux.sh`) from `code/`:
   upstream repositories at the pinned commits with the patches, conda environments, checkpoints, test data, test lists.
2. In two terminals (`tools` environment):
   ```
   python run.py generate full_test_sets
   python run.py evaluate full_test_sets
   ```
   Both are resumable. `python run.py show full_test_sets` lists the runs and their state.
3. Per-utterance results: `out/<decoder>/<set>/<method>/seed<k>/_metrics/metrics.csv` (WER / CER, SIM per embedder,
   UTMOS, durations), `acoustic.csv` (F0, speaking rate, spectral features), `objq.csv` (UTMOSv2, DNSMOS, NISQA) and
   `emb_<embedder>.npz` (speaker embeddings). Pair runs by utterance to compare methods.
4. Measurements on the decoder networks (they need the token caches of step 2, not generated audio):
   `python -m fsu.probes list` shows each measurement and its command, `python -m fsu.probes run` runs them all
   (timing measurements: `python -m fsu.probes run streaming_`). Outputs: `results/probes/<name>.csv`.

The results were produced on Windows 11 with one RTX 4080 SUPER (16 GB). CosyVoice 3, GLM-TTS, ZipVoice and DMOSpeech 2
outputs were bit-exact on rerun with the same GPU and software; CosyVoice 2 shows run-to-run GPU nondeterminism of up to
~6e-4 in the waveform on some utterances.

## Data
All data and checkpoints are public and downloaded by the setup: Seed-TTS-eval
(https://github.com/BytedanceSpeech/seed-tts-eval), the MiniMax multilingual test set (CC-BY-SA-4.0), LibriSpeech
test-clean / LibriSpeech-PC (CC-BY-4.0) and MUSAN. Seed-TTS-eval audio is not redistributed in the samples.

## Citation
```bibtex
@article{lee2026guidance,
  title   = {Guidance at Pure Noise Erodes Speaker Similarity in Few-Step Token-Conditioned Flow-Matching {TTS}},
  author  = {Lee, Jaeho and Choi, Han-Lim},
  journal = {arXiv preprint arXiv:2611.11179},
  year    = {2026}
}
```
