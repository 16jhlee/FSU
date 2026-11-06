"""E3 main experiment (v2 §2.3) configuration.

Configs (reference + comparisons), en 300 / zh 300, noise seeds k = 0..7.
k = 0 is the E1 output (identical seed `hash(utt|noise|0)`, verified bit-exact in the pilot);
k = 1..7 are generated under out/E3/main/....
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = [("en", "data/lists/en_all", "en"), ("zh", "data/lists/zh_all", "zh")]
K = list(range(8))
ZIPDIR = ROOT / "repos" / "ZipVoice"

# name: (family, env, e1_dir (k=0), e3 pattern, cmd builder(meta, k, out))
CONFIGS = {}


def _cosy(nfe):
    return lambda s, m, k, o: ["scripts/run_cosy.py", "--model_dir", "ckpt/CosyVoice2-0.5B", "--meta", m + ".meta.lst",
                               "--stage", "decode", "--cache_dir", f"cache/cosy2_tokens/{s}", "--nfe", str(nfe),
                               "--cfg", "0.7", "--noise", f"utt:{k}", "--out_dir", o]


def _zip(model, nfe, w):
    return lambda s, m, k, o: ["-m", "zipvoice.bin.infer_zipvoice", "--model-name", model,
                               "--model-dir", str(ROOT / "ckpt" / "ZipVoice" / model),
                               "--vocoder-path", str(ROOT / "ckpt" / "vocos-mel-24khz"),
                               "--test-list", str(ROOT / (m + ".zipvoice.tsv")), "--raw-evaluation", "True",
                               "--num-step", str(nfe), "--guidance-scale", w, "--seed-per-utt", "True",
                               "--noise-key", str(k), "--skip-existing", "True", "--res-dir", str(ROOT / o)]


def _dmo(mode, extra):
    return lambda s, m, k, o: ["scripts/run_dmo.py", "--meta", str(ROOT / (m + ".meta.lst")), "--mode", mode,
                               *extra, "--noise", f"utt:{k}", "--out_dir", str(ROOT / o)]


def add(name, fam, env, e1, cmd):
    CONFIGS[name] = dict(fam=fam, env=env, e1=e1, e3=e1.replace("out/E1/", "out/E3/main/").replace("/utt0", "/utt{k}"),
                         cmd=cmd)


add("cv2_nfe10", "A", "env_cosy", "out/E1/A/cosyvoice2/nfe10/cfg0.7/utt0", _cosy(10))
add("cv2_nfe3", "A", "env_cosy", "out/E1/A/cosyvoice2/nfe3/cfg0.7/utt0", _cosy(3))
add("cv2_nfe2", "A", "env_cosy", "out/E1/A/cosyvoice2/nfe2/cfg0.7/utt0", _cosy(2))
add("zip_nfe16", "B", "env_zip", "out/E1/B/zipvoice/nfe16/cfg1.0/utt0", _zip("zipvoice", 16, "1.0"))
add("zip_nfe4", "B", "env_zip", "out/E1/B/zipvoice/nfe4/cfg1.0/utt0", _zip("zipvoice", 4, "1.0"))
add("zipd_nfe4", "B", "env_zip", "out/E1/B/zipvoice_distill/nfe4/cfg3.0/utt0", _zip("zipvoice_distill", 4, "3.0"))
add("zipd_nfe2", "B", "env_zip", "out/E1/B/zipvoice_distill/nfe2/cfg3.0/utt0", _zip("zipvoice_distill", 2, "3.0"))
add("dmoT_nfe32", "C", "env_f5dmo", "out/E1/C/dmo_teacher/nfe32/cfg2.0/utt0",
    _dmo("teacher", ["--teacher_steps", "32"]))
add("dmoS_nfe4", "C", "env_f5dmo", "out/E1/C/dmo_student/nfe4/cfgfused/utt0",
    _dmo("student", ["--grid", "0,0.25,0.5,0.75"]))
add("dmoS_nfe2", "C", "env_f5dmo", "out/E1/C/dmo_student/nfe2/cfgfused/utt0", _dmo("student", ["--grid", "0,0.5"]))

# comparisons (few, reference) per v2 §2.3 table
COMPARISONS = [("cv2_nfe2", "cv2_nfe10"), ("cv2_nfe3", "cv2_nfe10"),
               ("zip_nfe4", "zip_nfe16"),
               ("zipd_nfe2", "zip_nfe16"), ("zipd_nfe4", "zip_nfe16"),
               ("dmoS_nfe2", "dmoT_nfe32"), ("dmoS_nfe4", "dmoT_nfe32")]


def wav_dir(name, k, s):
    c = CONFIGS[name]
    return ROOT / (c["e1"] if k == 0 else c["e3"].format(k=k)) / s


def n_lines(meta):
    return sum(1 for l in (ROOT / (meta + ".meta.lst")).read_text(encoding="utf-8").splitlines() if l.strip())
