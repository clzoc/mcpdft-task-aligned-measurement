# Data dictionary

## Molecular settings and acquisition

The equilibrium and scan `protocol.json` files record chemistry, pool seed, acquisition counts, solver settings and source hashes. `pool_real30.npz` stores rotations of shape `(30, 2, 8, 8)`: real orthogonal 8-orbital matrices, identical in the two spin sectors. Each geometry has a measurement-free DQG anchor in `contexts/`.

`pilots/<system>/r<stream>.npz` contains `samples` of shape `(30, 500, 120)` and `covariances` of shape `(30, 120, 120)`. A sample is the vector of occupation-pair indicators obtained from one shot, not 120 independent shots. Covariances include the stated shrinkage to particle-sector moments. `plans/` records the selected indices, all acquired counts, retained counts, pilot digest, measured rank and local variance ratios.

All outcomes in an orbital frame are generated together from the active-space state's joint distribution. Seeds and per-frame RNG prefixes determine production observations; the production bitstrings are not separately archived. The means actually supplied to every fit are archived in the result NPZ files.

## Result records

Each `results/<system>/<arm>/b<budget>_r<stream>.json` has a matching NPZ file.

| Field | Meaning |
| --- | --- |
| `d2` (NPZ) | Reconstructed real spin-orbital pair matrix, shape `(120,120)`, trace 45; ordered pairs `p < q` of 16 spin orbitals |
| `gamma` (NPZ) | Contracted active 1-RDM, shape `(16,16)`, trace 10; alpha orbitals followed by beta orbitals |
| `values` (NPZ) | Flattened pair-occupation means for the retained frames (1,800 entries for a subset; 3,600 for Uniform30) |
| `counts` | Acquired counts in all 30 frames, including pilots in discarded frames |
| `indices` | Zero-based retained-frame indices |
| `fit_counts`, `fit_shots` | Counts used in the reconstruction; subset total `budget - 7500` |
| `h_error_meh`, `f_error_meh` | Signed Hamiltonian and total ftPBE errors in mHa, reconstructed minus reference |
| `d2_error`, `gamma_error` | Unnormalized Frobenius errors against the exact active-space RDMs |
| `h_gamma_meh`, `h_d2_meh` | Signed effective one-electron and two-electron Hamiltonian contributions in mHa |
| `f_non_ontop_meh`, `f_on_top_meh` | Signed common and on-top MC-PDFT components in mHa |
| `minimum_dqg_eigenvalue`, `equality_residual`, `contraction_error` | Numerical feasibility diagnostics |
| `lambda_radius`, `mu_ftpbe` | Objective coefficients, λ and μ |
| `guard_proxy` | Pilot-based local variance ratios; these are not the MSE of the constrained fit |

Eight stream IDs (`0`–`7`) are the independent resampling units. For each arm/geometry/budget, energy RMSE is `sqrt(mean(error**2))`. An RDM curve is likewise the RMS Frobenius error, rather than the arithmetic mean norm. Error bars are pointwise 95% percentile bootstrap intervals from 100,000 resamples. Equilibrium seeds are 20261006 for N₂ and 20261007 for CO; scan means use 20261006. The same resampled stream IDs preserve within-molecule pairing. These are not simultaneous bands or prediction intervals.

## Source tables

`source-data/equilibrium_rmse_ci95.csv` contains point estimates and bootstrap bounds. `scan_signed_mean_ci95.csv` contains signed mean errors and confidence intervals of the mean. `scan_reference_energies.json` contains the classical CASCI Hamiltonian and ftPBE reference energies used for absolute energy curves. `scan_summary.csv` includes pointwise RMSEs.

`rdm_and_energy_components.csv` contains the component means, RMS values and cross second moments. The molecular one-electron contribution to MC-PDFT is provided separately in `molecular_one_electron_errors.csv`; it uses the bare molecular one-electron operator, rather than the effective active-space operator including core interactions. Fixed nuclear and frozen-core contributions cancel in the corresponding differences.

The ablation figure's opaque bar heights are total energy RMSEs. Its signed component means are multiplied by `RMSE / mean(total signed error)` for display. The components are therefore not component RMSEs and do not give an additive variance decomposition. Unscaled means and cross terms are retained in the source tables.

The 1.10-Å scan records reuse the equilibrium results: 16 of the 176 scan records are reused, and 160 are distinct new fits. These are not additional independent repetitions. `equilibrium_reuse.json` and source hashes record this provenance. Absolute paths in original protocol/result metadata refer to the original run locations; the distributed records are located by their repository-relative paths.

## Schematic diagnostic

Panel a uses a separate CASSCF(10e,8o)/cc-pVDZ N₂ reference at 1.10 Å with ftPBE grid level 1. The orbitals are fixed during the RDM search. Two numerically optimized DQG-feasible perturbations satisfy relative RDM error at most 1.01%, Hamiltonian shift at most 1 mHa and common-energy shift at most 5 mHa. Their total MC-PDFT shifts are −13.637 and +13.578 mHa; on-top shifts are −8.637 and +8.578 mHa. The percentage divides by the reference Frobenius norm 6.53462689, not by its trace. No measurement shots enter this diagnostic, and the numerical search does not certify global extrema.

Panels b and c contain illustrative variance and feasible-region geometries. Fifteen retained frames is a prescribed setting, not an inferred knee. The schematic's `E±` matrices correspond to the Methods' `P,N`; `F_lin` is a tangent of the total ftPBE energy. Panel d uses the archived N₂ acquisition results.

