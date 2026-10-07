# Task-aligned measurements for quantum–MC-PDFT

Code and numerical data accompanying **Reducing Measurement Costs in Quantum–Classical Multiconfiguration Pair-Density Functional Theory via Task-Aligned Derandomized Shadow**, by Zhanou Liu, Yuhao Chen, Yingjin Ma, Xiao He and Yuxin Deng.

The protocol measures a 500-shot pilot in each of 30 fixed Haar-real orbital-rotation frames, selects 15 frames using a covariance model for the RDM and the two energy directions, and divides the remaining shots equally. A DQG-constrained measurement-band SDP reconstructs the RDM with an objective containing the Hamiltonian, a fixed tangent of the total ftPBE MC-PDFT energy, and a trace penalty. Final MC-PDFT energies use the full nonlinear functional.

## Contents

| Directory | Contents |
| --- | --- |
| `analysis/guard15_equal_real30/` | N₂ and CO equilibrium calculations: original drivers, protocol, rotations, anchors, pilots, selection plans, 320 JSON/NPZ result pairs and plotting tables |
| `analysis/guard15_equal_real30_n2_scan120k/` | Eleven-point N₂ scan: drivers and 176 result pairs, including 16 equilibrium records reused at 1.10 Å |
| `analysis/outputs/n2_10e8o_dqg_ftpbe_1pct_range/` | CASSCF-based sensitivity diagnostic used in the schematic, including feasible RDM perturbations |
| `analysis/method_schematic/` | Schematic source and component images |
| Other `analysis/` directories | Imported computational modules, retained in their original relative layout |
| `noise-study/` | Fixed-plan circuit-noise extension: calibration workbook, circuits, histograms, 352 fitted records, processing controls and plotting tables |
| `source-data/` | Figure statistics, complete-stream errors, 1-RDM and energy components, variance-model comparisons and numerical checks |
| `figures/` | Current main-text and supplementary figures, circuit diagrams and TOC graphic |
| `scripts/` | Independent record validation, figure regeneration, supplementary-table generation and fresh reconstruction entry point |
| `docs/` | Reproduction instructions, data dictionary and figure-to-data map |
| `provenance/` | Source hashes, environment versions and documented portability changes |

All acquisition benchmarks use cc-pVDZ, CAS(10e,8o), two doubly occupied core orbitals, and singlet CASCI reference states in canonical RHF orbitals. N₂ equilibrium is at 1.10 Å; CO is at 1.128 Å. Equilibrium budgets are 30,000, 60,000, 120,000 and 240,000 total shots, each with eight complete streams. The N₂ scan covers 0.80–2.50 Å at 120,000 shots per geometry. The CASSCF sensitivity diagnostic is a separate deterministic calculation, as described in `docs/DATA_DICTIONARY.md`.

## Quick start: check existing results

Linux or WSL, Python 3.10:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/verify_data.py
python scripts/reproduce_figures.py
```

This validates archived results and regenerates the five experimental figures plus the supplementary detail view without a quantum chemistry calculation or an SDP solve. Published PDFs in `figures/` are preserved; regenerated copies go to `generated/figures/`. Numerical verification independently reproduces RMSEs and 95% bootstrap intervals from the complete-stream records, checks acquisition costs and paired observations, and verifies the file manifest.

A fresh N₂/30,000-shot/stream-0 reconstruction has also been checked through the portable entry point: the sampled observations and frame selection reproduce the archive exactly, with Hamiltonian and MC-PDFT error differences below $3\times10^{-7}$ mHa. The recorded check is in `provenance/reconstruction-smoke-check.json`.

For chemistry, reconstruction and the sensitivity calculation:

```bash
python -m pip install -r requirements.txt
python scripts/check_environment.py
python scripts/run_reconstruction.py equilibrium --system n2 \
  --arm guard15_equal --budget 30000 --stream 0 --threads 1 \
  --output runs/reproduction
```

A valid MOSEK license must be installed locally to run SDP calculations. No license file or credential is distributed. Complete recalculation is substantially more expensive than validating saved results; use the serial examples in [Reproduction](docs/REPRODUCING.md). The archived result files are never overwritten by `run_reconstruction.py`.

## Experimental interpretation

`guard15_equal` is Select15 with `(λ, μ) = (1, 2)`; `guard15_equal_mu0` changes only μ to zero on the same observations. `exclusive_guard15_mu0` and `exclusive_guard15_mu2` use the complementary 15 frames. `uniform` uses all 30 frames equally, with `(λ, μ) = (1, 0)`. The acquisition rule uses both energy directions in either selected-subset reconstruction.

Every subset protocol pays all 15,000 pilot shots, including the 7,500 discarded pilot observations. Thus the fitted shot count is `B - 7,500`. Repetitions include a fresh pilot and production sampling; protocols and budgets are paired by shared per-frame random-number prefixes. The frame pool is fixed. The original eight-stream campaigns include multinomial shot noise and same-shot correlations, with ideal state preparation and rotations. The additional circuit-noise campaign described below uses fixed plans and one realization per geometry and condition.

Energy errors are relative to the specified active-space reference. `F` always denotes the **total** nonlinear ftPBE MC-PDFT energy; its on-top contribution is recorded separately. Measurement-plot RDM errors are unnormalized pair-basis Frobenius norms. See [Data dictionary](docs/DATA_DICTIONARY.md) for units, array conventions and the component-bar scaling.

## Circuit-noise extension (v1.1.0)

The eleven-point N₂ scan additionally uses explicit 16-qubit orbital-rotation circuits with effective block-level gate noise and readout noise derived from an Origin Wukong-180-2 calibration workbook downloaded from the Origin Quantum Cloud platform. Its export timestamp is 16 July 2026, 11:22:01 UTC; a separate device-calibration timestamp is not available in that workbook. The workbook, hashes and logical-to-physical mapping are included.

The dimensionless **scale** multiplies the effective single-qubit depolarizing parameters applied to each participant after a Givens block: 1.0 retains the calibration-derived parameters and 0.2 reduces them fivefold. Readout bit-flip probabilities remain fixed; exact CASCI state preparation is ideal. Plans are frozen from the ideal-measurement stream-0 pilot. Uniform30 fits 120,000 outcomes; Select15 fits 112,500 and charges 7,500 discarded-pilot outcomes to the same nominal budget. There is one sampling realization per geometry and condition, without noisy-pilot reselection.

REM-lin + post inverts the tensor-product readout matrix, retains negative quasi-probabilities and normalizes within the five-electron-per-spin sector before fitting the occupation moments. At scale 0.2, scan-wide MC-PDFT RMSEs are 8.71 mEh for Select15 and 18.41 mEh for Uniform30; at scale 1.0 they are 50.27 and 64.05 mEh. These pool errors across geometries and are distinct from the eight-stream RMSE estimates.

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_noise_data.py
python noise-study/tools/verify_noise_moments.py
python noise-study/plot_scan_energy_signed_error.py
```

These commands check the calibration, all 352 records, 44 histogram files, 32 aggregate groups and 176 first-frame processing blocks, and regenerate the four noisy-scan plots under `generated/`. See [Reproduction](docs/REPRODUCING.md) for fresh circuit sampling and reconstruction.

The SDP retains its implemented two-matrix notation E⁺/E⁻. For these occupation maps, a PSD E⁺ has nonnegative mapped entries; replacing it by zero widens the bands and cannot increase the nonnegative trace penalty. An optimum therefore exists with E⁺ = 0, equivalent to a single PSD band-width matrix. This clarification changes neither the solver nor the reported numerical records.

## Citation and provenance

Use `CITATION.cff` and identify the repository version or commit when reusing these data. The original computational modules and numerical records are retained; packaging changes are listed in `provenance/portability-changes.json`. Historical directory names identify shared module origins, not additional experiments claimed in this release. The two ideal-measurement campaigns, the circuit-noise extension and the listed diagnostics supply the revised manuscript results.

The band-reconstruction reference implementation is documented in `THIRD_PARTY_NOTICES.md`; its existing license is retained under `licenses/`. This release does not assign a new blanket license to materials with different ownership or provenance.
