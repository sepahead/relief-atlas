#!/usr/bin/env bash
# Run full, certified batches until the scoped selection is complete, or stop
# after repeated passes without progress. Quarantine is reported as incomplete.
# Example: nohup scripts/local/run_forever.sh > state/full_run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p "${RELIEF_STATE_DIR:-state}"

STALL_LIMIT="${STALL_LIMIT:-3}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
[[ "$STALL_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "STALL_LIMIT must be positive" >&2; exit 2; }
[[ "$SLEEP_SECONDS" =~ ^[0-9]+$ ]] || { echo "SLEEP_SECONDS must be nonnegative" >&2; exit 2; }

# This driver promises a certified full deliverable. Diagnostic and partial
# runs belong in run_all.py, where their completion rules are explicit.
expect_phase=0
for arg in "$@"; do
    if [ "$expect_phase" -eq 1 ]; then
        [ "$arg" = all ] || { echo "run_forever.sh requires --phase all" >&2; exit 2; }
        expect_phase=0
    fi
    case "$arg" in
        --phase) expect_phase=1 ;;
        --phase=all) ;;
        --phase=*) echo "run_forever.sh requires --phase all" >&2; exit 2 ;;
        --skip-*-qa|--skip-preflight|--preflight-only|--dry-run)
            echo "run_forever.sh requires generation and every certification gate" >&2; exit 2 ;;
    esac
done

# Fail once on missing models/packages/GPU/disk instead of repeatedly launching
# an expensive run that cannot make progress.
python3 scripts/local/run_all.py --preflight-only "$@"

best_pending=-1
stalls=0
pass=0
while true; do
    pass=$((pass + 1))
    echo "pass $pass starting"
    if ! python3 scripts/local/run_all.py --skip-preflight "$@"; then
        echo "pass $pass: incomplete; checking whether progress warrants a retry"
    fi
    python3 scripts/local/verify_local.py
    PENDING=$(python3 scripts/local/pending.py --certified --count "$@")
    [[ "$PENDING" =~ ^[0-9]+$ ]] || { echo "Invalid pending count: $PENDING" >&2; exit 1; }
    echo "items pending certification: $PENDING"
    if [ "$PENDING" -eq 0 ]; then
        if python3 scripts/local/pending.py --certified --require-complete "$@"; then
            echo "all selected, permitted assets certified"
            exit 0
        fi
        echo "stopped with quarantined items; inspect state/qa_quarantine.json" >&2
        exit 1
    fi
    if [ "$best_pending" -lt 0 ] || [ "$PENDING" -lt "$best_pending" ]; then
        best_pending="$PENDING"
        stalls=0
    else
        stalls=$((stalls + 1))
        echo "no progress ($stalls/$STALL_LIMIT; best pending count $best_pending)"
        if [ "$stalls" -ge "$STALL_LIMIT" ]; then
            echo "stopped: $PENDING items still need certification" >&2
            python3 scripts/local/pending.py --certified "$@"
            exit 1
        fi
    fi
    sleep "$SLEEP_SECONDS"
done
