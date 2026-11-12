# Listening samples

Samples for [arXiv:2611.11179](https://arxiv.org/abs/2611.11179). Open `index.html` for a page with all players. Layout:
`<section>/<decoder>/<condition>/<utterance>/<method>.wav`, plus `text.txt` (prompt text and target text) in each
utterance folder. Every file is the unprocessed decoder output used in the paper (float WAV).

| Section | Content | Methods |
|---|---|---|
| `1_guidance_rules_analysis_subset` | two Euler steps, analysis subset (test-en / test-zh), seed 0 | CosyVoice2/3: Official CFG, FSU, FSU-Mid, teacher, Selective-CFG (input-text, default), CFG-Zero* (scale, zero-init), MAMBO-G. IndexTTS-2 and GLM-TTS (all-null): additionally the guidance interval, TV-CFG (released, corrected, step-up), terminal-fitted repair, JRR-CFG and prompt-consistency (GLM-TTS) |
| `2_full_test_sets` | full Seed-TTS-eval test-en / test-zh / test-hard and LibriSpeech-PC test-clean, two steps | Official CFG, FSU, FSU-Mid, teacher, TV-CFG (corrected), Selective-CFG (input-text), MAMBO-G |
| `3_robustness` | long-form text (offline and streaming), de / ja / ko, 1 s / 3 s / 10 s prompts, prompts at 20 / 10 dB SNR | as in section 2 |
| `4_first_step_weight_sweep` | first-step weight w1 with the second step at the official w = 0.7 | w1 = 0 is FSU, w1 = 0.7 the official CFG |

Notes
- Utterances: sections 2 and 3 use the first utterances of each list (the robustness and LibriSpeech-PC sets added last use
  the first, middle and last). The CosyVoice2/3 utterances of section 1 were picked by the per-utterance FSU gain (utterances
  near the median gain, from the top 10 %, with no gain, and with a WER change), so they are not a random draw.
- `GLM-TTS_speaker-null` is the released unconditional branch, and `GLM-TTS_all-null` the all-null branch used in the rule
  comparison. In `GLM-TTS_all-null` the teacher is the speaker-null ten-step teacher. On `GLM-TTS_all-null/de`, Selective-CFG
  (input-text) is the same run as the speaker-null official CFG. In `GLM-TTS_all-null/librispeech-pc`, the `speaker-null_*`
  files are the speaker-null runs, included for reference.
- Teacher: ten steps; IndexTTS-2 25 steps.
- Seed-TTS-eval prompt audio is not redistributed (its license does not allow it); the prompt file name is given in
  `text.txt`. The generated outputs for those prompts are included.
