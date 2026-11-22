#!/usr/bin/env bash
# Anti-grokking same-partition-ratio sweep + random-split train-percentage sweep
# for x+y mod MOD, each config repeated over 3 seeds.
#
# Normally called through tool/run_grokking_mod{97,197,597}.sh.
#
# Mirrors the mod-97 result folders on the share:
#   grokking/anti_grokking/x+y_mod_97_train50_chessboard_random_ratio_*
#   grokking/different_train_percentage/x+y_mod_97_train*
# Same model/optimizer/schedule as those runs (generate_grokking.py defaults:
# AdamW lr 1e-3, wd 0, 150k epochs, full-batch on the train partition).
#
# Usage:
#   MOD=97 bash tool/run_grokking_sweep.sh                 # both groups
#   MOD=97 bash tool/run_grokking_sweep.sh anti_ratio      # one group
#   MOD=97 DRY_RUN=1 bash tool/run_grokking_sweep.sh       # print commands only
#
# Groups:
#   anti_ratio  chessboard_random, same-partition ratio 100/90/80/70/60/50 (Fig. 4, 14)
#   train_pct   random split, train 15..50% in steps of 5               (Fig. 15)
#
# Env:
#   MOD             modulus (required)
#   PYTHON          interpreter (default: python)
#   OUT_ROOT        output root (default: tool/grokking_mod$MOD)
#   JOBS            concurrent runs on the GPU (default: 1)
#   CKPT_ALL_SEEDS  1 = save checkpoints every 500 epochs for every run;
#                   default saves them only for seed 0 (~1 GB per run at mod 97)
#   PLOT            1 = run the log / correct-position plotters at the end (default: 1)
#   EPOCHS          max training epochs (default: generate_grokking.py's 150000)
#   BS              batch size (default: full batch)
#   EARLY_STOP_VAL  stop a run once val acc >= this and train acc >= 0.95 (default: off)
#
# Re-running is safe: finished runs (0.model.pt present) are skipped; an
# existing folder without 0.model.pt is reported and skipped (delete it to rerun).

set -euo pipefail

: "${MOD:?set MOD, e.g. MOD=97}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
PYTHON=${PYTHON:-python}
EXPR="x+y_mod_${MOD}"
OUT_ROOT=${OUT_ROOT:-$SCRIPT_DIR/grokking_mod${MOD}}
OUT_ROOT=$(mkdir -p "$OUT_ROOT" && cd "$OUT_ROOT" && pwd)
JOBS=${JOBS:-1}
CKPT_ALL_SEEDS=${CKPT_ALL_SEEDS:-0}
DRY_RUN=${DRY_RUN:-0}
PLOT=${PLOT:-1}
EPOCHS=${EPOCHS:-}
BS=${BS:-}
EARLY_STOP_VAL=${EARLY_STOP_VAL:-}
TRAIN_OPTS=()
[[ -n $EPOCHS ]] && TRAIN_OPTS+=(-epoch "$EPOCHS")
[[ -n $BS ]] && TRAIN_OPTS+=(-bs "$BS")
[[ -n $EARLY_STOP_VAL ]] && TRAIN_OPTS+=(--early_stop_val_accuracy "$EARLY_STOP_VAL")

SEEDS=(0 1 2)
RATIOS=(100 90 80 70 60 50)
TRAIN_PCTS=(15 20 25 30 35 40 45 50)

ALL_GROUPS=(anti_ratio train_pct)
if (($#)); then GROUPS_SEL=("$@"); else GROUPS_SEL=("${ALL_GROUPS[@]}"); fi
for g in "${GROUPS_SEL[@]}"; do
    [[ " ${ALL_GROUPS[*]} " == *" $g "* ]] || { echo "unknown group: $g (valid: ${ALL_GROUPS[*]})" >&2; exit 1; }
done
want() { [[ " ${GROUPS_SEL[*]} " == *" $1 "* ]]; }

# Job spec: "subdir|name|seed|extra args (space separated, no spaces inside)".
# Seed-major order: one full single-seed pass finishes before the repeats.
# A shared seed gives every ratio run the same initial weights and base split.
JOB_SPECS=()
for seed in "${SEEDS[@]}"; do
    if want anti_ratio; then
        for r in "${RATIOS[@]}"; do
            JOB_SPECS+=("anti_grokking|x+y_mod_${MOD}_train50_chessboard_random_ratio_${r}_${seed}|${seed}|-st chessboard_random -tp 50 --chessboard_transpose_ratio ${r}")
        done
    fi
    if want train_pct; then
        for p in "${TRAIN_PCTS[@]}"; do
            JOB_SPECS+=("different_train_percentage|x+y_mod_${MOD}_train${p}_${seed}|${seed}|-st random -tp ${p}")
        done
    fi
done

run_one() {
    local group=$1 name=$2 seed=$3 extra=$4
    local out="$OUT_ROOT/$group/$name"
    local log="$OUT_ROOT/logs/$name.log"
    local -a args ckpt=()
    read -r -a args <<<"$extra"
    if [[ $seed == 0 || $CKPT_ALL_SEEDS == 1 ]]; then
        ckpt=(--save_format file --save_interval 500)
    fi
    local -a cmd=("$PYTHON" ./generate_grokking.py -dexp "$EXPR" --modulus "$MOD" -s "$seed"
                  -o "$out" "${args[@]}" ${TRAIN_OPTS[@]+"${TRAIN_OPTS[@]}"} ${ckpt[@]+"${ckpt[@]}"})

    if [[ -f "$out/0.model.pt" ]]; then
        echo "[skip] $name (finished)"
        return 0
    fi
    if [[ -e "$out" ]]; then
        echo "[skip] $name: $out exists without 0.model.pt (incomplete); delete it to rerun" >&2
        return 0
    fi
    if [[ $DRY_RUN == 1 ]]; then
        echo "(cd $SCRIPT_DIR && ${cmd[*]})"
        return 0
    fi
    mkdir -p "$OUT_ROOT/$group" "$OUT_ROOT/logs"
    echo "[start $(date '+%F %T')] $name"
    if (cd "$SCRIPT_DIR" && "${cmd[@]}") >"$log" 2>&1; then
        echo "[done  $(date '+%F %T')] $name"
    else
        echo "[FAIL  $(date '+%F %T')] $name (see $log)" >&2
    fi
}

echo "modulus     : $MOD"
echo "output root : $OUT_ROOT"
echo "groups      : ${GROUPS_SEL[*]}"
echo "runs        : ${#JOB_SPECS[@]}   concurrent: $JOBS"
echo "train opts  : ${TRAIN_OPTS[*]:-(defaults)}"

for spec in "${JOB_SPECS[@]}"; do
    IFS='|' read -r group name seed extra <<<"$spec"
    if [[ $DRY_RUN == 1 ]]; then
        run_one "$group" "$name" "$seed" "$extra"
        continue
    fi
    run_one "$group" "$name" "$seed" "$extra" &
    while (($(jobs -rp | wc -l) >= JOBS)); do
        wait -n || true
    done
done
wait

if [[ $DRY_RUN != 1 && $PLOT == 1 ]]; then
    echo "[plot] training curves and correct-position maps"
    (cd "$REPO_DIR/result_processing_tool/grokking" && "$PYTHON" ./plot_grokking_logs.py "$OUT_ROOT") \
        || echo "[plot] plot_grokking_logs.py failed" >&2
    (cd "$REPO_DIR/result_processing_tool/grokking" && "$PYTHON" ./plot_correct_position.py "$OUT_ROOT" --modulus "$MOD") \
        || echo "[plot] plot_correct_position.py failed" >&2
fi
