"""``python -m rag_eval_kit.eval``: run the evaluation harness and write ``eval/RESULTS.md``."""

import sys

from rag_eval_kit.eval.run import main

raise SystemExit(main(sys.argv[1:]))
