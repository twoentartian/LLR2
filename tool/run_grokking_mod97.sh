#!/usr/bin/env bash
# anti_ratio + train_pct sweeps for x+y mod 97, 3 seeds each.
# Repeats of the existing (unseeded, single-run) mod-97 results on the share.
# Options (groups, DRY_RUN, JOBS, OUT_ROOT, ...): see tool/run_grokking_sweep.sh
MOD=97 exec bash "$(dirname "${BASH_SOURCE[0]}")/run_grokking_sweep.sh" "$@"
