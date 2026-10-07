"""Transparent resource accounting for direct point and frame measurements."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb


@dataclass(frozen=True)
class MeasurementResources:
    n_spatial_orbitals: int
    quartic_coefficients: int
    quadratic_coefficients: int
    frames: int
    probes_per_execution: int
    readout_bits_per_execution: int
    two_qubit_gates_per_execution: int
    two_qubit_depth_parameter_lower_bound: int
    two_qubit_depth_linear_upper_bound: int
    executions: int
    state_preparation_included: bool = False

    @property
    def two_qubit_gate_shots(self) -> int:
        return self.two_qubit_gates_per_execution * self.executions


def resource_estimate(
    n_spatial_orbitals: int,
    shots_per_setting: int,
    oversampling: float = 2.0,
) -> MeasurementResources:
    """Count ideal frame resources without claiming a hardware advantage.

    A generic real N by N orbital rotation uses N(N-1)/2 Givens rotations per
    spin sector. Alpha and beta networks can run in parallel, but both count
    toward gate-shot volume. The depth interval reports a parameter-counting
    lower bound and the conservative 2N-3 nearest-neighbor Givens schedule.
    State preparation is deliberately excluded because it is ansatz-dependent.
    """
    if n_spatial_orbitals < 1 or shots_per_setting < 1 or oversampling < 1.0:
        raise ValueError("resource inputs must be positive and oversampling >= 1")
    n = n_spatial_orbitals
    k4 = comb(n + 3, 4)
    k2 = comb(n + 1, 2)
    frames = ceil(oversampling * k4 / n)
    gates = n * (n - 1)
    per_spin_gates = n * (n - 1) // 2
    parallel_gates = max(1, n // 2)
    depth_lower = ceil(per_spin_gates / parallel_gates)
    depth_upper = max(0, 2 * n - 3)
    return MeasurementResources(
        n_spatial_orbitals=n,
        quartic_coefficients=k4,
        quadratic_coefficients=k2,
        frames=frames,
        probes_per_execution=n,
        readout_bits_per_execution=2 * n,
        two_qubit_gates_per_execution=gates,
        two_qubit_depth_parameter_lower_bound=depth_lower,
        two_qubit_depth_linear_upper_bound=depth_upper,
        executions=frames * shots_per_setting,
    )
