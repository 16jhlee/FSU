"""v8 §2.1 cross-lingual prompt lists (prompt language != text language), 100 each, seed-tts-eval meta format.
  xl_zh2en: zh prompt (zh_all rows 0..99) + en text (en_all rows 0..99)
  xl_en2zh: en prompt + zh text
  xl_en2ko: en prompt + MiniMax ko text (mm_ko 0..99)
  xl_ko2en: MiniMax ko prompts (2 speakers, as listed in mm_ko) + en text
Same-language references: en_all / zh_all (first 100 = same prompts) and mm_ko.  Run in any env.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    return [l.split("|") for l in (ROOT / f"data/lists/{name}.meta.lst").read_text(encoding="utf-8").splitlines() if l.strip()]


EN, ZH, KO = rows("en_all")[:100], rows("zh_all")[:100], rows("mm_ko")[:100]
SPEC = {"xl_zh2en": (ZH, EN), "xl_en2zh": (EN, ZH), "xl_en2ko": (EN, KO), "xl_ko2en": (KO, EN)}
for name, (P, T) in SPEC.items():
    out = [f"{name}_{i:03d}|{p[1]}|{p[2]}|{t[3]}" for i, (p, t) in enumerate(zip(P, T))]
    (ROOT / f"data/lists/{name}.meta.lst").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(name, len(out))
# same-language 100-utterance references with identical prompts/texts as the cross lists use
for src, name in [(EN, "xl_en2en"), (ZH, "xl_zh2zh"), (KO, "xl_ko2ko")]:
    out = [f"{name}_{i:03d}|{r[1]}|{r[2]}|{r[3]}" for i, r in enumerate(src)]
    (ROOT / f"data/lists/{name}.meta.lst").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(name, len(out))
