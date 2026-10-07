Circuit-noise source data for the revised MC-PDFT measurement manuscript

Scope
N2/cc-pVDZ/CAS(10e,8o), eleven bond lengths, fixed stream-0 measurement plans,
one circuit-sampling realization per geometry/arm/condition, and effective
gate-noise scales scale=0.2 and scale=1.0. The exact CASCI input state is prepared ideally.
The 120,000-shot protocol accounting corresponds to 112,500 simulated fitted
outcomes for Select15 (plus 7,500 charged discarded-pilot outcomes) and 120,000
for Uniform30. No noisy pilot reselection is performed in this extension.

Calibration provenance
The Wukong-180-2 workbook was downloaded from the Origin Quantum Cloud platform,
as confirmed by the authors. Its export timestamp is 2026-07-16T11:22:01Z.
A separate device-calibration timestamp is not recorded in the supplied file.
Workbook SHA256:
8db26f06cdbab91f880614607741f9752a1093d66e7f895a893f4dc63a909db1
The frozen noise-model JSON and workbook are in circuits/calibration/.

Contents
analysis/ supplies per-geometry errors, aggregate RMSEs and exact energy references.
code/mindquantum_poc/results/ and results_gate020/ contain the scale=1.0 and scale=0.2
histograms, fitted moments/RDMs and scores, respectively. Original records are
preserved: missing gate_scale in early results/ JSON files means scale=1.0. The combined
CSV explicitly fills that field. Historical internal arm IDs are retained;
uniform means Uniform30, guard15_equal means Select15 with mu=2.
circuits/ contains the explicit rotation circuits and calibration inputs.
figures/ contains the four noisy-scan PDFs used in the article and SI.
code/ retains the original relative module layout for the sampled-circuit study;
unrelated hardware execution scripts, hardware job records and logs are excluded.
validation/ records independent checks of all 352 scores, 44 histogram files,
32 scan-wide RMSE groups and 176 first-frame processing blocks.

Processing
REM-lin uses the tensor-product inverse of the symmetric readout matrices without
clipping negative values. REM-lin + post then conditions the signed distribution
on five electrons per spin sector and computes pair moments. It is not a
regularized inverse. The linear correction's sector normalization is not an
accepted-shot fraction. The original acceptance_post metadata is retained only
for traceability; in rem_lin_post it was computed from a different, clipped
processing path and should not be used to interpret the linear estimator.

Verification and figure regeneration
With Python and NumPy: python tools/verify_noise_moments.py
With NumPy, SciPy and Matplotlib: python plot_scan_energy_signed_error.py
The latter regenerates the four noisy-scan PDFs from archived tables under
../generated/figures/noise/; the moment check writes to ../generated/noise-validation/.
Circuit sampling used Python 3.10 and MindQuantum 0.12.0 (mqvector); fitting used
the original CVXPY/MOSEK pipeline. The source layout and scripts are included
for inspection and reuse. The checks shipped here use archived outputs and
do not require an SDP solve or access to a quantum device.

The manuscript SI Sections S9-S10 define the effective block-level noise model,
calibration table, fixed-plan accounting and all processing variants.
