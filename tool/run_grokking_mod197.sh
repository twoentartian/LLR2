#!/usr/bin/env bash
# anti_ratio + train_pct sweeps for x+y mod 197, 3 seeds each.
# Options (groups, DRY_RUN, JOBS, OUT_ROOT, ...): see tool/run_grokking_sweep.sh
MOD=197 exec bash "$(dirname "${BASH_SOURCE[0]}")/run_grokking_sweep.sh" "$@"
