# Guidance at Pure Noise Erodes Speaker Similarity in Few-Step Token-Conditioned Flow-Matching TTS — code and listening samples

Supplementary material for the IEEE/ACM TASLP submission.

| Folder | Content |
|---|---|
| `code/scripts/` | runners for each decoder (`run_cosy.py`, `run_index.py`, `run_glm.py`, `run_f5.py`, `run_dmo.py`), generation/evaluation queues, metrics (`eval_metrics.py`, `objq_metrics.py`), one job file per experiment block (`e<k>_jobs.py`), analysis scripts, test-list builders |
| `code/patches/` | sampler-level patches to the upstream repositories (time grid, CFG schedule, unconditional branch, solver, seeds); model weights, losses and architectures are not changed |
| `code/setup/` | `setup_windows.ps1`, `setup_linux.sh`: clone the upstream repositories at the pinned commits, apply the patches, create the conda environments, download checkpoints and data |
| `code/env/` | `pins.txt` (upstream commits and model revisions), `lock_<env>.txt` (pip freeze), `env_*.yml` (conda export) |
| `audio_samples/` | listening samples; open `audio_samples/index.html` |

## Methods in the code
The proposed samplers are options of the existing runners (repository time convention: t = 0 is noise).

| Method | Runner arguments (two steps unless noted) |
|---|---|
| Official CFG | `--nfe 2` |
| FSU (first-step unguided) | `--nfe 2 --cfg_interval 0.1,1` |
| FSU-Mid | `--nfe 1 --solver midpoint --cfg_interval 0.1,1` (3 network calls) |
| Teacher | `--nfe 10` (IndexTTS-2: `--nfe 25`) |
| TV-CFG (corrected) | `--nfe 2 --tv symmetric_fix` |
| Selective-CFG (input-text) | `--nfe 2 --cfg_branch speaker` |
| MAMBO-G | `--nfe 2 --mambo_alpha 4` |
| GLM-TTS all-null branch | `--cfg_branch allnull` (`run_glm.py`) |
| CosyVoice2 streaming | add `--stream 1` |

Noise is seeded per utterance (`--noise utt:<k>`, k = 0 unless stated), and the CosyVoice2/3 and GLM-TTS LLM speech tokens
are generated once and cached (`--stage tokens`), so every decoder configuration sees the same tokens. The exact command
lines of every configuration are in the job files (`e<k>_jobs.py`).

## Reproducing
1. Run `setup/setup_windows.ps1` (or `setup/setup_linux.sh`): upstream repositories, environments, checkpoints and the
   public test sets.
2. Build the test lists: `scripts/make_subset.py` (analysis subset, seed 20261017) and `scripts/relocate.py`
   (Seed-TTS-eval lists), then `build_long_lists.py`, `build_robust_lists.py`, `build_minimax_lists.py`,
   `build_lspc_list.py`, `build_crossling_lists.py`.
3. Generate and evaluate a block with its job file through the queues, for example
   `python scripts/queue_gen.py e15_jobs <config>` and `python scripts/queue_eval.py e15_jobs <config>`
   (the `chain_*.ps1` files show the order used). Runs are resumable (completion markers, skip-existing).
4. The analysis scripts (`e<k>_analyze.py`) compute the paired statistics (bootstrap 10k, Wilcoxon, Holm).

Results were produced on Windows 11 with one RTX 4080 SUPER (16 GB). ZipVoice, DMOSpeech2, CosyVoice3 and GLM-TTS outputs were
bit-exact on rerun with the same GPU and software; CosyVoice2 shows run-to-run GPU nondeterminism of up to ~6e-4 in the waveform
on some utterances. A few orchestration scripts (`chain_*.ps1`, `v9_*`, `v8_dom_run.py`, `build_lspc_list.py`) contain
absolute paths of the original machine and need editing before use.

## Data
All data and checkpoints are public. Seed-TTS-eval (https://github.com/BytedanceSpeech/seed-tts-eval), the MiniMax
multilingual test set (CC-BY-SA-4.0), LibriSpeech test-clean / LibriSpeech-PC (CC-BY-4.0) and MUSAN noise are downloaded by
the setup. Seed-TTS-eval audio is not redistributed in the samples.
