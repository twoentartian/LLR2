#!/usr/bin/env bash
# anti_ratio + train_pct sweeps for x+y mod 597, 3 seeds each.
# Batch size and epochs follow the paper's Appendix A rule for n=597 (Table 2):
#   batch = min(len(D), 65536) = 65536, E = max(2000, ceil(150000 / 3^1.5)) = 28868 epochs,
# used for every run. Runs stop early once train and val accuracy both reach 0.95.
# Options (groups, DRY_RUN, JOBS, OUT_ROOT, ...): see tool/run_grokking_sweep.sh
MOD=597 BS=${BS:-65536} EPOCHS=${EPOCHS:-28868} EARLY_STOP_VAL=${EARLY_STOP_VAL:-0.95} \
    exec bash "$(dirname "${BASH_SOURCE[0]}")/run_grokking_sweep.sh" "$@"
