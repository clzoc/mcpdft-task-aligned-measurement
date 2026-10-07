#!/usr/bin/env bash
# Gate-noise scale sweep around the Wukong-calibrated model:
#   scale 1.0 (already sampled) -> add linear-REM variants
#   scale 0.2 (1/5 gate noise)  -> sample + solve
#   scale 0.0 (readout only)    -> sample + solve
set -u
cd "$(dirname "$0")"
mkdir -p logs
TAGS=(n2_r080 n2_r090 n2_r100 n2_r110 n2_r125 n2_r145 n2_r160 n2_r180 n2_r200 n2_r220 n2_r250)

solve_gate() {
    local scale="$1"
    printf '%s\n' "${TAGS[@]}" | xargs -P 4 -I{} sh -c \
        "python -u run_scan.py solve --tag {} --gate-scale ${scale} > logs/solve_g${scale}_{}.log 2>&1"
}

echo "[driver] scale 1.0: linear-REM solves $(date -Is)"
solve_gate 1.0

echo "[driver] scale 0.2 sampling $(date -Is)"
for tag in "${TAGS[@]}"; do
    python -u run_scan.py sample --tag "$tag" --gate-scale 0.2 \
        > "logs/sample_g020_${tag}.log" 2>&1 &
done
wait
echo "[driver] scale 0.2 sampling done $(date -Is)"
solve_gate 0.2

echo "[driver] scale 0.0 sampling $(date -Is)"
for tag in "${TAGS[@]}"; do
    python -u run_scan.py sample --tag "$tag" --gate-scale 0.0 \
        > "logs/sample_g000_${tag}.log" 2>&1 &
done
wait
echo "[driver] scale 0.0 sampling done $(date -Is)"
solve_gate 0.0

echo "[driver] summarize $(date -Is)"
python -u run_scan.py summarize --scales 1.0,0.2,0.0 > logs/summarize_sweep.log 2>&1
echo "[driver] done $(date -Is)"
