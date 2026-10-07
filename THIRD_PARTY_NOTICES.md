# Computational provenance

The inequality-band DQG reconstruction follows the constrained-shadow formulation of Avdic and Mazziotti cited in the manuscript. The original Maple reference in the working source was distributed as `ConstrainedShadowTomography` with the Apache License, Version 2.0. A copy of that existing license is retained in `licenses/ConstrainedShadowTomography-Apache-2.0.txt`.

`analysis/mcpdft_measurement_revision/guarded_mcpdft_shadow_protocol/code/vendor/constrained_shadow.py` is the local Python reconstruction utility used by these calculations. `analysis/original_bands/run.py` constructs the inequality-band variant and `reusable_band_solver.py` reuses its graph. These files are retained as used in the calculations; their source hashes are recorded in the protocol and provenance manifests. The frame screen, affine ftPBE term, campaign drivers, statistics and source data are the accompanying research code and records.

External dependencies, including PySCF, Qiskit, CVXPY and MOSEK, retain their own licenses and are installed separately. No MOSEK license, API token, third-party paper PDF or reviewer correspondence is included. No new blanket license is assigned by this repository to third-party material or the authors' code and data.


## Circuit-noise extension

Circuit simulation uses MindQuantum 0.12.0, installed separately under its upstream license. The supplied Origin Wukong-180-2 calibration workbook was downloaded from the Origin Quantum Cloud platform; its export timestamp and SHA256 are recorded in `noise-study/README.txt`. The workbook retains its original provenance. No new blanket license is assigned to third-party calibration material.
