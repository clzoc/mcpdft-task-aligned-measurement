#!/usr/bin/env bash
# Retry missing variant solves (gate 0.2 and gate 0.0), then re-summarize.
# - serial execution, one solver at a time to avoid the OOM kills seen at -P4
# - waits for the main run_scales.sh driver to exit first
set -u
cd "$(dirname "$0")"
mkdir -p logs

exec 9>logs/.missing_retry.lock
flock 9

LOG=logs/missing_retry_driver.log
echo "[missing-retry] queued $(date -Is)" >>"$LOG"

TAGS=(n2_r080 n2_r090 n2_r100 n2_r110 n2_r125 n2_r145 n2_r160 n2_r180 n2_r200 n2_r220 n2_r250)

wait_for_others() {
    while pgrep -f "run_scan.py solve" >/dev/null || pgrep -f "run_scales.sh" >/dev/null; do
        sleep 60
    done
}

wait_mem() {
    while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 12 ]; do
        sleep 30
    done
}

complete_count() {
    local d="$1" tag="$2"
    ls "$d/$tag/variants"/*.json 2>/dev/null | wc -l
}

retry_scale() {
    local scale="$1"
    local dir
    case "$scale" in
        1.0) dir=results ;;
        0.2) dir=results_gate020 ;;
        0.0) dir=results_gate000 ;;
        *) echo "[missing-retry] bad scale $scale" >>"$LOG"; return 1 ;;
    esac
    for tag in "${TAGS[@]}"; do
        for attempt in 1 2 3; do
            n=$(complete_count "$dir" "$tag")
            if [ "$n" -ge 16 ]; then
                [ "$attempt" -eq 1 ] && echo "[missing-retry] scale $scale $tag complete ($n/16)" >>"$LOG"
                break
            fi
            echo "[missing-retry] scale $scale $tag attempt $attempt ($n/16) $(date -Is)" >>"$LOG"
            wait_for_others
            wait_mem
            python -u run_scan.py solve --tag "$tag" --gate-scale "$scale" \
                >>"logs/solve_g${scale}_${tag}_retry.log" 2>&1
            rc=$?
            n=$(complete_count "$dir" "$tag")
            echo "[missing-retry] scale $scale $tag attempt $attempt rc=$rc ($n/16) $(date -Is)" >>"$LOG"
            [ "$n" -ge 16 ] && break
        done
    done
}

retry_scale 0.2
retry_scale 0.0

echo "[missing-retry] summarize $(date -Is)" >>"$LOG"
python -u run_scan.py summarize --scales 1.0,0.2,0.0 >>logs/summarize_sweep_retry.log 2>&1
echo "[missing-retry] summarize rc=$? $(date -Is)" >>"$LOG"
echo "[missing-retry] done $(date -Is)" >>"$LOG"
