#!/usr/bin/env bash
# Full noisy N2 scan: sample all geometries in parallel, solve with bounded
# concurrency (DQG solves peak around 5.5 GiB each), then summarize.
set -u
cd "$(dirname "$0")"
mkdir -p logs results
TAGS=(n2_r080 n2_r090 n2_r100 n2_r110 n2_r125 n2_r145 n2_r160 n2_r180 n2_r200 n2_r220 n2_r250)

echo "[driver] sampling $(date -Is)"
for tag in "${TAGS[@]}"; do
    python -u run_scan.py sample --tag "$tag" > "logs/sample_${tag}.log" 2>&1 &
done
wait
echo "[driver] sampling done $(date -Is)"

echo "[driver] solving $(date -Is)"
printf '%s\n' "${TAGS[@]}" | xargs -P 4 -I{} sh -c \
    'python -u run_scan.py solve --tag {} > logs/solve_{}.log 2>&1'
echo "[driver] solving done $(date -Is)"

python -u run_scan.py summarize > logs/summarize.log 2>&1
echo "[driver] summarized $(date -Is)"
