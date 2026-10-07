N2 bond scan, 120K shots, guard15 (1,2) versus uniform no-pilot spin (1,0).

Runs ONLY after guard15_equal_real30 completes all 320 primary results.
Bonds (Angstrom): 0.80, 0.90, 1.00, 1.10, 1.25, 1.45, 1.60, 1.80, 2.00, 2.20, 2.50.
Each geometry: N2 cc-pVDZ CAS(10e,8o), ncore=2, streams r0-r7, same canonical
RHF/CASCI context and DQG-anchor procedure as the original N2 experiment.
Total 176 results: reuse 16 existing 1.10-A results; run 160 new solves.
The 1.10-A point is never recomputed. Source result and array hashes are saved.

Fixed pool: same 30 Haar O(8) real rotations, seed 20260716, identical alpha/beta.
Each geometry and stream recomputes guard15 selection using its own 30x500 pilot;
the frame pool and per-stream seed convention are shared across geometries.
Guard15: 15000 pilot + 105000 production = 120000 total; 7000 new shots/frame
on selected15, with 500 pilot reused per selected frame; fit total 112500.
Uniform: no pilot, 4000 shots/frame on all30, fit total 120000.
Guard merit H + 2*F_lin(geometry DQG anchor) + Tr(E); baseline H + Tr(E).
Both use spin-constrained error matrices (not full-E), DQG, MOSEK tolerance 1e-8.
Truth is used for simulated acquisition and scoring, not frame selection.

after_primary.py: live dependency watcher; automatic import and scan launch.
queue_state.json: waiting/running/completed dependency status.
run_campaign.py: resumable scheduler with at most four concurrent jobs,
8 GiB/guard and 10 GiB/uniform reservations, 36 GiB total RSS limit,
5 GiB available-memory reserve; single budget per process, solver threads=3.
runtime_policy.json can be adjusted live; memory guard retries at lower concurrency.
engine.py is a frozen copy of the primary numerical engine; scan.py supplies
the scan geometry context and reporting without modifying primary code.

Start chain: python -u guard15_equal_real30_n2_scan120k/after_primary.py
Stop watcher/scan: create guard15_equal_real30_n2_scan120k/STOP.
After completion: summary.csv/json gives separate H/F RMSE and bias, plus
mean/RMS unnormalized Frobenius 2-RDM errors. per_stream.csv/json contains
all 176 individual results. scan_errors.png/pdf plots the bond-length curves.
equilibrium_reuse.json records the direct numerical reuse of the 1.10-A point.
