#!/bin/bash
# One command for the rest of Stage 5 (USER-RUN): bash scripts/run_stage5.sh
# Steps, resume logic, pauses and stop rules: scripts/run_stage5.py and results/figures/evidence/STAGE5_ONE_COMMAND.md.
cd "$HOME/Downloads/tp_project" || exit 1
source env/mlx/bin/activate || exit 1
exec python scripts/run_stage5.py "$@"
