"""Command-line entry point. Run with the `tools` environment from the repository root.

    python run.py list                                     experiments and their size
    python run.py show rule_comparison                     runs of one experiment and their state
    python run.py generate rule_comparison                 generate the runs (resumable)
    python run.py evaluate rule_comparison                 score them (run next to `generate`; waits for new runs)

`generate` and `evaluate` are meant to run at the same time in two terminals: generation uses one GPU job at a time and
evaluation another, and the two never overlap on the decoders that need the whole GPU (IndexTTS-2, GLM-TTS).
Per-utterance results are written to out/<decoder>/<set>/<method>/seed<k>/_metrics/ (metrics.csv, acoustic.csv,
objq.csv, emb_<embedder>.npz).
"""
import argparse
import sys

from fsu import experiment


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "show", "generate", "evaluate"])
    ap.add_argument("experiment", nargs="?")
    ap.add_argument("--only", nargs="*", help="restrict to runs whose method starts with, or decoder equals, one of these")
    ap.add_argument("--no-wait", action="store_true", help="evaluate: return when nothing is ready instead of waiting")
    a = ap.parse_args(argv)
    if a.command == "list":
        for p in sorted((experiment.CONFIGS / "experiments").glob("*.yaml")):
            e = experiment.Experiment(p)
            print(f"{e.name:24s} {len(e.runs):4d} runs   {e.description}")
        return
    if not a.experiment:
        ap.error("an experiment name is required")
    exp = experiment.Experiment(a.experiment)
    if a.command == "show":
        for r in exp.runs:
            state = "evaluated" if (r.dir / "_evaluated").exists() else "done" if (r.dir / "_done").exists() else \
                f"{r.n_done()}/{r.n_expected()}"
            print(f"{r.decoder:16s} {r.test_set:22s} {r.method:28s} seed{r.seed}  {state}")
    elif a.command == "generate":
        experiment.generate(exp, a.only)
    else:
        experiment.evaluate(exp, wait=not a.no_wait, only=a.only)


if __name__ == "__main__":
    sys.exit(main())
