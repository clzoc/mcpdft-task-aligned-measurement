# Reproducing the calculations

Run commands from the repository root on Linux or WSL. Python 3.10.20 and the versions in `requirements.txt` were used for packaging verification. The numerical environment is also recorded in `provenance/environment.json`.

## 1. Validate and plot the archived measurements

Only `requirements-analysis.txt` is needed:

```bash
python scripts/verify_data.py
python scripts/reproduce_figures.py
```

The first command checks all 496 result records, energy-component identities, traces, feasibility diagnostics, pilot digests, acquisition costs, 128 paired objective comparisons, and the 16 reused scan records. It independently reconstructs all 120 equilibrium RMSE confidence intervals and 44 scan signed-mean confidence intervals from the eight-stream records. It writes `generated/validation.json`.

The second command uses the original plotting functions and directs their output to `generated/figures/`. Supplied publication PDFs in `figures/` remain unchanged. Reproducing plot layouts does not require a MOSEK license. Minor font/rendering differences across systems do not change the plotted statistics.

## 2. Regenerate the supplementary numerical tables

```bash
python scripts/build_revision_tables.py
```

This reaggregates saved records into `source-data/`, checks their numerical consistency, and emits the supplementary LaTeX content in `generated/si.tex`. The manuscript class and cross-reference files are not part of this computational repository; that LaTeX output is intended for the manuscript source tree. The 1-RDM and molecular one-electron contractions have already been archived. To recalculate these contractions from saved fitted 1-RDMs and regenerated classical references, install the full dependencies and run:

```bash
python scripts/extract_one_electron_errors.py
```

This rebuilds two classical references but does not repeat measurements, selection or SDP fits. Record validation is intended for the pristine archive; regenerating source tables can update provenance or floating-point formatting and hence their file hashes.

## 3. Recalculate one complete acquisition and reconstruction

Install `requirements.txt`, configure a local MOSEK license, and check imports:

```bash
python scripts/check_environment.py
python scripts/run_reconstruction.py equilibrium --system n2 \
  --arm guard15_equal --budget 30000 --stream 0 --threads 1 \
  --output runs/reproduction
```

This imports the original computational kernels, copies only fixed anchors and rotations to the output tree, and samples the pilot, performs selection, samples production observations and solves the reconstruction. A later μ=0 run in the same output tree reuses that pilot and selection, matching the original paired design:

```bash
python scripts/run_reconstruction.py equilibrium --system n2 \
  --arm guard15_equal_mu0 --budget 30000 --stream 0 --threads 1 \
  --output runs/reproduction
```

Choose `co_eq` for CO, a budget in `{30000,60000,120000,240000}`, and stream `0`–`7`. The five arm names are defined in the README. Each complete result is skipped if it already exists in the chosen output tree. Use a new output directory to force a fresh run. Original scientific records under `analysis/` are never replaced by this wrapper.

The original calculations used a 20-GiB per-worker virtual-memory limit and recorded roughly 8–13 GiB memory reservations for large solves. A workstation with at least 24–32 GiB available RAM is appropriate for serial reconstruction; runtime is several minutes per fit in the recorded environment and can be longer. Start with one job rather than the historical multi-worker supervisor.

## 4. Recalculate the complete equilibrium study and scan

Run the following serial loops after the one-record check:

```bash
for system in n2 co_eq; do
  for stream in 0 1 2 3 4 5 6 7; do
    for budget in 30000 60000 120000 240000; do
      for arm in guard15_equal guard15_equal_mu0 exclusive_guard15_mu2 exclusive_guard15_mu0 uniform; do
        python scripts/run_reconstruction.py equilibrium --system "$system" \
          --arm "$arm" --budget "$budget" --stream "$stream" --threads 1 \
          --output runs/reproduction
      done
    done
  done
done

for system in n2_r080 n2_r090 n2_r100 n2_r110 n2_r125 n2_r145 n2_r160 n2_r180 n2_r200 n2_r220 n2_r250; do
  for stream in 0 1 2 3 4 5 6 7; do
    for arm in guard15_equal uniform; do
      python scripts/run_reconstruction.py scan --system "$system" \
        --arm "$arm" --budget 120000 --stream "$stream" --threads 1 \
        --output runs/reproduction
    done
  done
done
```

The 1.10-Å scan command copies the corresponding recalculated equilibrium record and updates its provenance; it does not solve again. In total these loops entail 320 equilibrium fits plus 160 new scan fits. They can require many hours. Cached DQG anchors are supplied; to recalculate an anchor, use a separate working copy and remove that anchor before calling the underlying context builder.

The `run_campaign.py` and `after_primary.py` files are retained as historical run provenance. The `run_reconstruction.py` wrapper is the documented portable entry point, without the original campaign-completion and workstation-specific scheduling assumptions.

## 5. Sensitivity and exact-mean diagnostics

The schematic perturbation calculation has its own original driver:

```bash
python analysis/outputs/n2_10e8o_dqg_ftpbe_1pct_range/run_range_tight.py
python analysis/outputs/n2_10e8o_dqg_ftpbe_1pct_range/make_figure_color.py
```

These commands rebuild the CASSCF reference and rerun the nonlinear search. They write in that diagnostic directory, so run them in a separate clone if retaining the pristine numerical archive. Its two feasible points are illustrations of energy sensitivity, not certified global extrema.

The full-pool exact-mean checks used in SI Section S4 are retained with `analysis/guard15_real30_dqg_lineality/noiseless_reference.py` and their JSON/NPZ records. The historical directory name does not imply a lineality-based selection rule in the released method. The driver skips existing records; in a separate clone, move the chosen archived result pair aside before rerunning it with `--system n2` or `--system co_eq`.

The historical four-panel schematic is assembled by `analysis/method_schematic/make_schematic.py` from supplied panel images and vector elements. `make_panel_d.py` can regenerate its measured-data panel. The schematic source keeps the original graphical design; the manuscript caption and data dictionary define its relative RDM norm, total-energy tangent and illustrative geometry.

## 6. Circuit-noise verification and reproduction

The README verification commands use archived outputs without a solver license. `verify_noise_data.py` independently checks workbook calibration values, shot accounting, selected plans and all aggregate RMSEs; it also emits three SI tables under `generated/noise-tables/`. `verify_noise_moments.py` independently reconstructs raw, ideal, linear-corrected and signed-sector-conditioned pair moments for the first frame of every histogram file. The four regenerated noise plots go to `generated/figures/noise/`. Archived files remain unchanged by these commands.

For fresh sampling and fitting, install the full environment plus MindQuantum:

```bash
python -m pip install -r requirements-noise.txt
python scripts/run_noise_reconstruction.py sample --tag n2_r080 --gate-scale 0.2
python scripts/run_noise_reconstruction.py solve --tag n2_r080 --gate-scale 0.2
```

The wrapper copies the computational sources and fixed plans to `runs/noise-reproduction/code/`, excluding archived noise histograms and fits. Subsequent calls reuse this working tree. `sample` generates both ideal and noisy histograms; `solve` performs eight processing/reconstruction variants for each of the two arms and needs a MOSEK license. The original driver uses three computational threads and a 20-GiB address-space limit during fitting. Use the eleven geometry tags listed above and gate scales 0.2 and 1.0 for the full campaign. A new `--output` directory gives an independent output tree. The archived plans remain fixed; these commands do not rerun selection with noisy pilot data.

The effective depolarizing channels follow complete Givens blocks, not individual gates of the illustrated decomposition. Each participant receives an independent one-qubit depolarizing channel. Only these gate parameters are scaled; independent symmetric readout bit flips remain fixed. The full model and calibration provenance are in `noise-study/README.txt` and `noise-study/circuits/calibration/`.

Historical `g0.2`/`g1.0` filenames and `gate_scale` keys are retained to preserve source traceability; they denote the dimensionless scale. In early full-scale JSON records a missing `gate_scale` means 1.0. The `acceptance_post` field in `rem_lin_post` originates from a different clipped processing path and is not the signed-sector normalization used by the reported estimator.

## Reproducibility levels

Archived numeric records and hashes reproduce the reported statistics exactly. Fresh chemistry and conic optimization can show small version-, platform- or tolerance-dependent differences; compare solver status, equality/PSD residuals and energy errors as well as random seeds. The archived calculations retain their original source hashes, and `provenance/portability-changes.json` identifies packaging changes. The circuit-noise extension contains simulated gate/readout-noise benchmarks; it does not contain new quantum-device executions.

