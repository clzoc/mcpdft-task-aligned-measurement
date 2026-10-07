#!/usr/bin/env python3
"""Compare real random prefixes with phase-complete complex-frame prefixes.

The hybrid path keeps the original finite-shot outcomes for every retained
real orbital frame and replaces selected frames with independent complex Haar
orbital unitaries. Cadence, fixed-count random placement, and independent
Bernoulli placement are supported. Complex-frame outcomes are sampled from the
same PySCF CASCI state with the same number of shots per frame. Exact RDMs are
used only to generate the validation samples and to score the frozen
reconstructions; frame selection is state independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hybrid-complex-shadow-ablation")

import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT_ROOT = ROOT.parent
CODE_ROOT = ROOT / "code"
VENDOR_ROOT = CODE_ROOT / "vendor"
for path in (VENDOR_ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    PairVectorDesign,
    ShadowData,
    _wilson_bounds,
    build_n2_reference,
    shadow_pair_vectors,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    acquire_shadow_bases,
    build_c2_reference,
    build_c2_selection_reference,
    build_n2_selection_reference,
    load_blind_shadow_npz,
    shadow_design_blocks,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from ridge_selector import weighted_prediction_rmse  # noqa: E402
from run_safe_mcpdft_derandomization import (  # noqa: E402
    RECONSTRUCTION_ENERGY,
    RECONSTRUCTION_RIDGE,
    _initial_result,
    _solve_shadow_dqg,
)


DEFAULT_REAL = {
    "n2": ROOT / "sweeps" / "n2_mosek_m1_30",
    "c2": ROOT / "sweeps" / "c2_seed19_m1_1000_shots1000_mosek",
}
SYSTEM_TAG = {"n2": 7, "c2": 19}


def _parse_counts(value: str) -> tuple[int, ...]:
    counts: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            if first > last:
                raise argparse.ArgumentTypeError("Count ranges must be increasing.")
            counts.update(range(first, last + 1))
        else:
            counts.add(int(item))
    if not counts or min(counts) < 1:
        raise argparse.ArgumentTypeError("Counts must contain positive integers.")
    return tuple(sorted(counts))


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from error
    if not grid or min(grid) < 0.0:
        raise argparse.ArgumentTypeError("Ridge values must be nonnegative.")
    return grid


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("n2", "c2"), required=True)
    parser.add_argument("--real-trajectory", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--counts", type=_parse_counts, default=(4, 8, 12, 20, 30))
    parser.add_argument("--cadence", type=int, default=4)
    parser.add_argument(
        "--complex-placement",
        choices=("cadence", "random", "bernoulli"),
        default="cadence",
    )
    parser.add_argument("--complex-count", type=int)
    parser.add_argument("--complex-probability", type=float, default=0.5)
    parser.add_argument("--placement-seed", type=int, default=161803)
    parser.add_argument("--complex-seed", type=int, default=271828)
    parser.add_argument("--sampling-seed", type=int, default=314159)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument(
        "--shadow-data", choices=("finite_shot", "exact"), default="finite_shot"
    )
    parser.add_argument("--exact-fit-rmse-cap", type=float, default=1e-7)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument(
        "--fit-mode", choices=("paired", "two_stage"), default="paired"
    )
    parser.add_argument(
        "--selector",
        choices=("energy", "cv_ridge_closest"),
        default="energy",
    )
    parser.add_argument(
        "--ridge-grid",
        type=_parse_float_grid,
        default=(0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0),
    )
    parser.add_argument("--ridge-folds", type=int, default=5)
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Write the finite-shot candidate archive and stop before reconstruction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _random_unitaries(
    n_orbitals: int, count: int, seed: int
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    rotations = []
    for _ in range(count):
        matrix = (
            rng.normal(size=(n_orbitals, n_orbitals))
            + 1j * rng.normal(size=(n_orbitals, n_orbitals))
        ) / math.sqrt(2.0)
        q_matrix, r_matrix = np.linalg.qr(matrix)
        diagonal = np.diag(r_matrix)
        phases = np.where(np.abs(diagonal) > 0.0, diagonal / np.abs(diagonal), 1.0)
        rotations.append(q_matrix @ np.diag(phases))
    return tuple(rotations)


def _statevector_to_ci(reference: Any) -> tuple[np.ndarray, np.ndarray]:
    from pyscf.fci import cistring

    n_spatial = int(reference.n_spatial_orbitals)
    alpha = np.asarray(
        cistring.make_strings(range(n_spatial), reference.n_alpha), dtype=np.uint64
    )
    beta = np.asarray(
        cistring.make_strings(range(n_spatial), reference.n_beta), dtype=np.uint64
    )
    occupations = alpha[:, None] | (beta[None, :] << np.uint64(n_spatial))
    ci = np.asarray(reference.statevector)[occupations.astype(np.int64)]
    norm = float(np.linalg.norm(ci))
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise RuntimeError(f"Reference CI sector norm is {norm:.12g}, not one.")
    return np.asarray(ci / norm), occupations


def _pair_indicators(
    occupations: np.ndarray, pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    states = np.asarray(occupations, dtype=np.uint64).reshape(-1)
    return np.column_stack(
        [
            ((states >> np.uint64(first)) & np.uint64(1))
            * ((states >> np.uint64(second)) & np.uint64(1))
            for first, second in pairs
        ]
    ).astype(np.int8)


def _prepare_references(
    system: str, real_configuration: dict[str, Any]
) -> tuple[Any, Any, Path]:
    if system == "n2":
        signs = tuple(float(value) for value in real_configuration["active_orbital_signs"])
        selection = build_n2_selection_reference(
            float(real_configuration["bond_length_angstrom"]),
            basis=str(real_configuration["basis"]),
            active_electrons=int(real_configuration["active_space"][0]),
            active_orbitals=int(real_configuration["active_space"][1]),
            active_orbital_signs=signs,
        )
        exact = build_n2_reference(
            float(real_configuration["bond_length_angstrom"]),
            basis=str(real_configuration["basis"]),
            active_electrons=int(real_configuration["active_space"][0]),
            active_orbitals=int(real_configuration["active_space"][1]),
            active_orbital_signs=signs,
        )
        archive = Path(real_configuration["archive"])
    else:
        archive = Path(real_configuration["frozen_mo_npz"])
        with np.load(archive, allow_pickle=False) as arrays:
            frozen_mo = np.array(arrays["mo_coeff"], copy=True)
        selection = build_c2_selection_reference(
            float(real_configuration["bond_length_angstrom"]),
            basis=str(real_configuration["basis"]),
            frozen_mo_coeff=frozen_mo,
        )
        exact = build_c2_reference(
            float(real_configuration["bond_length_angstrom"]),
            basis=str(real_configuration["basis"]),
            frozen_mo_coeff=frozen_mo,
        )
    with np.load(archive, allow_pickle=False) as arrays:
        archived_d2 = np.asarray(arrays["exact_d2"], dtype=float)
        archived_gamma = np.asarray(arrays["exact_gamma"], dtype=float)
    if (
        np.max(np.abs(exact.exact_d2 - archived_d2)) > 1e-9
        or np.max(np.abs(exact.exact_gamma - archived_gamma)) > 1e-9
    ):
        raise RuntimeError("Rebuilt exact RDMs do not match the source archive.")
    return selection, exact, archive


def _hybrid_shadows(
    raw: ShadowData,
    exact: Any,
    count: int,
    replacement_indices: Sequence[int],
    complex_seed: int,
    sampling_seed: int,
) -> tuple[ShadowData, np.ndarray, dict[str, Any]]:
    if count > raw.n_shadows:
        raise ValueError("The source archive does not contain enough real frames.")
    replacement_indices = tuple(sorted(int(index) for index in replacement_indices))
    if (
        len(set(replacement_indices)) != len(replacement_indices)
        or (replacement_indices and min(replacement_indices) < 0)
        or (replacement_indices and max(replacement_indices) >= count)
    ):
        raise ValueError("Complex replacement indices are invalid.")
    complex_rotations = iter(
        _random_unitaries(
            exact.n_spatial_orbitals,
            len(replacement_indices),
            complex_seed + SYSTEM_TAG["n2" if exact.n_spatial_orbitals == 6 else "c2"],
        )
    )
    rotations = []
    is_complex = np.zeros(count, dtype=bool)
    for index in range(count):
        if index in replacement_indices:
            rotations.append(next(complex_rotations))
            is_complex[index] = True
        else:
            rotations.append(np.asarray(raw.rotations[index]))
    pair_vectors = shadow_pair_vectors(rotations, exact.n_spatial_orbitals, exact.pairs)
    design = PairVectorDesign(pair_vectors)
    exact_values = design @ exact.exact_d2.ravel(order="C")

    rows_per_basis = len(exact.pairs)
    hits = np.empty(count * rows_per_basis, dtype=int)
    values = np.empty(count * rows_per_basis, dtype=float)
    real_exact_error = 0.0
    for index in range(count):
        start = index * rows_per_basis
        stop = start + rows_per_basis
        if is_complex[index]:
            continue
        hits[start:stop] = raw.hits[start:stop]
        values[start:stop] = raw.values[start:stop]
        if np.all(np.isfinite(raw.exact_values[start:stop])):
            real_exact_error = max(
                real_exact_error,
                float(
                    np.max(
                        np.abs(exact_values[start:stop] - raw.exact_values[start:stop])
                    )
                ),
            )

    ci, determinant_occupations = _statevector_to_ci(exact)
    indicators = _pair_indicators(determinant_occupations, exact.pairs)
    child_seeds = np.random.SeedSequence(
        [sampling_seed, SYSTEM_TAG["n2" if exact.n_spatial_orbitals == 6 else "c2"]]
    ).spawn(len(replacement_indices))
    maximum_complex_moment_error = 0.0
    from pyscf import fci

    for index, child_seed in zip(replacement_indices, child_seeds):
        rotation = np.asarray(rotations[index])
        rotated_ci = fci.addons.transform_ci_for_orbital_rotation(
            ci,
            exact.n_spatial_orbitals,
            (exact.n_alpha, exact.n_beta),
            rotation,
        )
        probabilities = np.abs(np.asarray(rotated_ci).reshape(-1)) ** 2
        probabilities = np.asarray(probabilities / probabilities.sum(), dtype=float)
        direct_moments = probabilities @ indicators
        start = index * rows_per_basis
        stop = start + rows_per_basis
        moment_error = float(
            np.max(np.abs(direct_moments - exact_values[start:stop]))
        )
        maximum_complex_moment_error = max(maximum_complex_moment_error, moment_error)
        if moment_error > 2e-10:
            raise RuntimeError(
                f"Complex frame {index + 1} moment mismatch: {moment_error:.3e}."
            )
        sampled = np.random.default_rng(child_seed).choice(
            len(probabilities), size=raw.shots_per_basis, p=probabilities
        )
        hits[start:stop] = np.asarray(indicators[sampled].sum(axis=0), dtype=int)
        values[start:stop] = hits[start:stop] / raw.shots_per_basis

    lower, upper = _wilson_bounds(hits, raw.shots_per_basis, 2.0)
    shadows = ShadowData(
        rotations=tuple(rotations),
        pair_vectors=pair_vectors,
        design=design,
        values=values,
        lower_bounds=lower,
        upper_bounds=upper,
        hits=hits,
        shots_per_basis=raw.shots_per_basis,
        exact_values=exact_values,
        exact_constraints=False,
        occupations=None,
    )
    diagnostics = {
        "replacement_indices_zero_based": list(replacement_indices),
        "replacement_counts_one_based": [index + 1 for index in replacement_indices],
        "complex_basis_count": len(replacement_indices),
        "maximum_real_exact_moment_error": real_exact_error,
        "maximum_complex_exact_moment_error": maximum_complex_moment_error,
        "finite_shot_pair_rmse": float(np.sqrt(np.mean((values - exact_values) ** 2))),
    }
    return shadows, is_complex, diagnostics


def _replace_with_exact_values(shadows: ShadowData) -> ShadowData:
    """Use analytic pair expectations with a near-equality weighted fit.

    Pseudo-hits only define positive numerical row scales.  Since the fit cap
    is driven to zero, the feasible set is the exact linear-shadow equality
    set and does not depend on those scales.
    """

    exact_values = np.asarray(shadows.exact_values, dtype=float)
    pseudo_hits = np.rint(exact_values * shadows.shots_per_basis).astype(int)
    pseudo_hits = np.clip(pseudo_hits, 0, shadows.shots_per_basis)
    return ShadowData(
        rotations=shadows.rotations,
        pair_vectors=shadows.pair_vectors,
        design=shadows.design,
        values=exact_values.copy(),
        lower_bounds=exact_values.copy(),
        upper_bounds=exact_values.copy(),
        hits=pseudo_hits,
        shots_per_basis=shadows.shots_per_basis,
        exact_values=exact_values.copy(),
        exact_constraints=False,
        occupations=None,
    )


def _real_checkpoint_path(system: str, root: Path, count: int) -> Path:
    if system == "n2":
        return root / "checkpoints" / f"shadows_{count:02d}.npz"
    return root / "checkpoints" / "random_prefix" / f"shadows_{count:04d}.npz"


def _load_real_checkpoint(
    system: str, root: Path, count: int
) -> tuple[np.ndarray, np.ndarray]:
    path = _real_checkpoint_path(system, root, count)
    if not path.is_file():
        raise FileNotFoundError(f"Missing real-prefix checkpoint: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return (
            np.asarray(arrays["d2"], dtype=float),
            np.asarray(arrays["gamma"], dtype=float),
        )


def _design_diagnostics(
    shadows: ShadowData, reference: Any
) -> dict[str, float | int]:
    rows, cols = _symmetric_d2_variables(reference)
    blocks = shadow_design_blocks(shadows, len(reference.pairs), rows, cols)
    matrix = np.vstack(blocks)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        singular[0] * max(matrix.shape) * np.finfo(float).eps
        if len(singular)
        else 0.0
    )
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        "design_variable_count": int(len(rows)),
        "design_rank": rank,
        "design_nullity": int(len(rows) - rank),
        "design_nonzero_condition_number": (
            float(singular[0] / singular[rank - 1]) if rank else float("inf")
        ),
    }


def _standardized_rmse(shadows: ShadowData, predictions: np.ndarray) -> float:
    probabilities = (shadows.hits.astype(float) + 0.5) / (
        shadows.shots_per_basis + 1.0
    )
    variances = np.maximum(
        probabilities * (1.0 - probabilities) / shadows.shots_per_basis,
        1e-12,
    )
    return float(
        np.sqrt(np.mean((np.asarray(predictions) - shadows.values) ** 2 / variances))
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-14:
        return None
    return float(np.clip(np.vdot(left, right).real / denominator, -1.0, 1.0))


def _score_row(
    count: int,
    is_complex: np.ndarray,
    acquired: ShadowData,
    result: Any,
    real_d2: np.ndarray,
    real_gamma: np.ndarray,
    exact: Any,
    objective: FtPBEEnergyObjective,
    exact_ftpbe: float,
    solver_seconds: float,
    solver_source: str,
    reconstruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hybrid_d2 = np.asarray(result.d2, dtype=float)
    hybrid_gamma = np.asarray(result.gamma, dtype=float)
    hybrid_eval = objective.evaluate(hybrid_d2, hybrid_gamma, gradient=False)
    real_eval = objective.evaluate(real_d2, real_gamma, gradient=False)
    exact_prediction = acquired.predict(exact.exact_d2)
    hybrid_prediction = acquired.predict(hybrid_d2)
    target = np.asarray(exact.exact_d2 - real_d2).ravel()
    update = np.asarray(hybrid_d2 - real_d2).ravel()
    target_norm = float(np.linalg.norm(target))
    update_norm = float(np.linalg.norm(update))
    dot = float(np.vdot(update, target).real)
    update_cosine = _cosine(update, target)
    hybrid_d2_error = float(np.linalg.norm(hybrid_d2 - exact.exact_d2))
    real_d2_error = float(np.linalg.norm(real_d2 - exact.exact_d2))
    hybrid_energy_error = float(abs(result.energy - exact.exact_energy))
    real_energy = float(
        np.sum(exact.one_body * real_gamma)
        + np.sum(exact.two_body * real_d2)
        + exact.nuclear_energy
    )
    real_energy_error = float(abs(real_energy - exact.exact_energy))
    row = {
        "shadows": count,
        "total_shots": count * acquired.shots_per_basis,
        "complex_basis_count": int(np.count_nonzero(is_complex[:count])),
        "complex_basis_counts": ",".join(
            str(index + 1) for index in np.flatnonzero(is_complex[:count])
        ),
        "solver_source": solver_source,
        "solver_seconds": solver_seconds,
        "status": str(result.status),
        "fit_status": result.fit_status,
        "weighted_fit_rmse": float(weighted_prediction_rmse(acquired, hybrid_d2)),
        "exact_standardized_fit_rmse": _standardized_rmse(acquired, exact_prediction),
        "exact_pair_moment_rmse": float(
            np.sqrt(np.mean((hybrid_prediction - exact_prediction) ** 2))
        ),
        "finite_shot_pair_rmse": float(
            np.sqrt(np.mean((acquired.values - exact_prediction) ** 2))
        ),
        "hybrid_d2_frobenius_error": hybrid_d2_error,
        "real_d2_frobenius_error": real_d2_error,
        "hybrid_to_real_d2_error_ratio": hybrid_d2_error / real_d2_error,
        "hybrid_gamma_frobenius_error": float(
            np.linalg.norm(hybrid_gamma - exact.exact_gamma)
        ),
        "real_gamma_frobenius_error": float(
            np.linalg.norm(real_gamma - exact.exact_gamma)
        ),
        "hybrid_hamiltonian_energy_eh": float(result.energy),
        "real_hamiltonian_energy_eh": real_energy,
        "hybrid_hamiltonian_error_eh": hybrid_energy_error,
        "real_hamiltonian_error_eh": real_energy_error,
        "hybrid_to_real_hamiltonian_error_ratio": (
            hybrid_energy_error / real_energy_error if real_energy_error > 0.0 else None
        ),
        "hybrid_ftpbe_energy_eh": float(hybrid_eval.total_energy),
        "real_ftpbe_energy_eh": float(real_eval.total_energy),
        "exact_ftpbe_energy_eh": exact_ftpbe,
        "hybrid_ftpbe_error_eh": float(
            abs(hybrid_eval.total_energy - exact_ftpbe)
        ),
        "real_ftpbe_error_eh": float(abs(real_eval.total_energy - exact_ftpbe)),
        "hybrid_to_real_ftpbe_error_ratio": (
            abs(hybrid_eval.total_energy - exact_ftpbe)
            / abs(real_eval.total_energy - exact_ftpbe)
            if abs(real_eval.total_energy - exact_ftpbe) > 0.0
            else None
        ),
        "hybrid_update_cosine_to_exact_correction": update_cosine,
        "hybrid_update_norm_to_exact_correction": (
            update_norm / target_norm if target_norm > 0.0 else None
        ),
        "hybrid_update_parallel_amplitude": (
            dot / target_norm**2 if target_norm > 0.0 else None
        ),
        "hybrid_update_oracle_alpha": (
            dot / update_norm**2 if update_norm > 1e-14 else None
        ),
        **(reconstruction or {}),
        **_design_diagnostics(acquired, exact),
    }
    return row


def _report(system: str, configuration: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    if configuration["complex_placement"] == "cadence":
        design_description = (
            f"Every {configuration['cadence']}-th real random orbital basis is "
            "replaced by a complex Haar orbital basis."
        )
    elif configuration["complex_placement"] == "random":
        design_description = (
            f"{len(configuration['complex_basis_counts'])} positions are sampled "
            "without replacement and use complex Haar orbital bases."
        )
    else:
        design_description = (
            "Each position independently samples a real or complex Haar frame "
            f"with complex probability {configuration['complex_probability']}."
        )
    lines = [
        f"# {system.upper()} real-vs-hybrid complex shadow ablation",
        "",
        (
            f"{design_description} Each basis uses the same "
            f"{configuration['shots_per_basis']} shots."
        ),
        "",
        "| m | complex | rank | D2 real | D2 hybrid | ratio | H err real/hybrid (mEh) | ftPBE err real/hybrid (mEh) | update cosine |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cosine = row["hybrid_update_cosine_to_exact_correction"]
        lines.append(
            f"| {row['shadows']} | {row['complex_basis_count']} | "
            f"{row['design_rank']}/{row['design_variable_count']} | "
            f"{row['real_d2_frobenius_error']:.6f} | "
            f"{row['hybrid_d2_frobenius_error']:.6f} | "
            f"{row['hybrid_to_real_d2_error_ratio']:.3f} | "
            f"{1000 * row['real_hamiltonian_error_eh']:.3f}/"
            f"{1000 * row['hybrid_hamiltonian_error_eh']:.3f} | "
            f"{1000 * row['real_ftpbe_error_eh']:.3f}/"
            f"{1000 * row['hybrid_ftpbe_error_eh']:.3f} | "
            f"{cosine:.3f} |" if cosine is not None else
            f"| {row['shadows']} | {row['complex_basis_count']} | "
            f"{row['design_rank']}/{row['design_variable_count']} | "
            f"{row['real_d2_frobenius_error']:.6f} | "
            f"{row['hybrid_d2_frobenius_error']:.6f} | "
            f"{row['hybrid_to_real_d2_error_ratio']:.3f} | "
            f"{1000 * row['real_hamiltonian_error_eh']:.3f}/"
            f"{1000 * row['hybrid_hamiltonian_error_eh']:.3f} | "
            f"{1000 * row['real_ftpbe_error_eh']:.3f}/"
            f"{1000 * row['hybrid_ftpbe_error_eh']:.3f} | n/a |"
        )
    lines.extend(
        [
            "",
            "The update cosine scores `D_hybrid - D_real` against "
            "`D_exact - D_real`; it is a posthoc diagnostic, not an input to the method.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    args.solver = args.solver.upper()
    real_root = (
        args.real_trajectory.resolve()
        if args.real_trajectory is not None
        else DEFAULT_REAL[args.system].resolve()
    )
    real_configuration = json.loads((real_root / "configuration.json").read_text())
    maximum_count = max(args.counts)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT
        / "sweeps"
        / f"{args.system}_hybrid_complex_every{args.cadence}_m1_30"
    )
    output.mkdir(parents=True, exist_ok=True)

    print(f"Building aligned {args.system.upper()} references...", flush=True)
    selection, exact, archive_path = _prepare_references(
        args.system, real_configuration
    )
    raw = load_blind_shadow_npz(archive_path, load_design=False)
    if maximum_count > raw.n_shadows:
        raise ValueError(
            f"The source archive contains {raw.n_shadows} frames, "
            f"but m={maximum_count} was requested."
        )
    if raw.shots_per_basis != int(real_configuration["shots_per_shadow"]):
        raise RuntimeError("Source shots per basis disagree with the real trajectory.")
    if args.complex_placement == "cadence":
        placement_description = f"cadence {args.cadence}"
    elif args.complex_placement == "random":
        placement_description = "fixed-count random placement"
    else:
        placement_description = (
            f"independent Bernoulli({args.complex_probability:g}) placement"
        )
    print(
        f"Generating {maximum_count} hybrid frames with {placement_description}...",
        flush=True,
    )
    if args.complex_placement == "cadence":
        if args.cadence < 2:
            raise ValueError("cadence must be at least two.")
        replacement_indices = tuple(
            index
            for index in range(maximum_count)
            if (index + 1) % args.cadence == 0
        )
    elif args.complex_placement == "random":
        complex_count = (
            args.complex_count
            if args.complex_count is not None
            else int(round(maximum_count / args.cadence))
        )
        if complex_count < 0 or complex_count > maximum_count:
            raise ValueError("complex-count must lie between zero and max(counts).")
        placement_rng = np.random.default_rng(
            [args.placement_seed, SYSTEM_TAG[args.system]]
        )
        replacement_indices = tuple(
            sorted(
                int(index)
                for index in placement_rng.choice(
                    maximum_count, size=complex_count, replace=False
                )
            )
        )
    else:
        if not 0.0 <= args.complex_probability <= 1.0:
            raise ValueError("complex-probability must lie between zero and one.")
        placement_rng = np.random.default_rng(
            [args.placement_seed, SYSTEM_TAG[args.system]]
        )
        replacement_indices = tuple(
            int(index)
            for index, use_complex in enumerate(
                placement_rng.random(maximum_count) < args.complex_probability
            )
            if use_complex
        )
    hybrid, is_complex, sampling = _hybrid_shadows(
        raw,
        exact,
        maximum_count,
        replacement_indices,
        args.complex_seed,
        args.sampling_seed,
    )
    if args.shadow_data == "exact":
        hybrid = _replace_with_exact_values(hybrid)
        sampling = {
            **sampling,
            "data_mode": "analytic exact pair expectations",
            "finite_shot_pair_rmse": 0.0,
            "pseudo_hits_used_only_for_row_scaling": True,
        }
    _atomic_npz(
        output / "hybrid_shadows.npz",
        rotations=np.asarray(hybrid.rotations),
        pair_vectors=np.asarray(hybrid.pair_vectors),
        hits=np.asarray(hybrid.hits),
        values=np.asarray(hybrid.values),
        lower_bounds=np.asarray(hybrid.lower_bounds),
        upper_bounds=np.asarray(hybrid.upper_bounds),
        shots_per_basis=np.asarray(hybrid.shots_per_basis),
        exact_values=np.asarray(hybrid.exact_values),
        is_complex_basis=is_complex,
    )

    if args.shadow_data == "exact":
        fit_cap = float(args.exact_fit_rmse_cap)
    else:
        fit_cap = (
            real_configuration.get("weighted_fit_rmse_cap")
            if args.fit_mode == "paired"
            else None
        )
    reconstruction_objective = (
        RECONSTRUCTION_ENERGY
        if args.selector == "energy"
        else RECONSTRUCTION_RIDGE
    )
    solver_args = SimpleNamespace(
        solver=args.solver,
        solver_tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        verbose_solver=False,
        weighted_fit_rmse_cap=fit_cap,
        reconstruction_objective=reconstruction_objective,
        ridge_grid=args.ridge_grid,
        ridge_folds=args.ridge_folds,
        fallback_ridge=1e-2,
        symmetry_blocked_psd=True,
    )
    configuration = {
        "system": args.system.upper(),
        "basis": real_configuration["basis"],
        "bond_length_angstrom": real_configuration["bond_length_angstrom"],
        "active_space": real_configuration["active_space"],
        "real_trajectory": str(real_root),
        "source_archive": str(archive_path),
        "counts": list(args.counts),
        "cadence": args.cadence,
        "complex_placement": args.complex_placement,
        "complex_probability": (
            args.complex_probability
            if args.complex_placement == "bernoulli"
            else None
        ),
        "placement_seed": (
            args.placement_seed
            if args.complex_placement in {"random", "bernoulli"}
            else None
        ),
        "complex_basis_counts": [
            int(index + 1) for index in np.flatnonzero(is_complex)
        ],
        "complex_seed": args.complex_seed,
        "sampling_seed": args.sampling_seed,
        "shots_per_basis": hybrid.shots_per_basis,
        "shadow_data": args.shadow_data,
        "exact_fit_rmse_cap": (
            args.exact_fit_rmse_cap if args.shadow_data == "exact" else None
        ),
        "same_basis_budget_as_real": True,
        "complex_frame_type": "spatial Haar unitary applied equally to alpha and beta",
        "estimator_scope": (
            "phase-complete orbital pair-density tomography for a real D2; "
            "not a full matchgate/Low fermionic classical-shadow estimator"
        ),
        "solver": args.solver,
        "solver_tolerance": args.solver_tolerance,
        "solver_threads": args.solver_threads,
        "max_iterations": args.max_iterations,
        "positivity": "DQG (2-positivity)",
        "symmetry_blocked_psd": True,
        "weighted_fit_rmse_cap": fit_cap,
        "fit_mode": args.fit_mode,
        "selector": args.selector,
        "ridge_grid": list(args.ridge_grid),
        "ridge_folds": args.ridge_folds,
        "grid_level": args.grid_level,
        "generate_only": args.generate_only,
        "sampling_diagnostics": sampling,
    }
    _atomic_json(output / "configuration.json", configuration)

    if args.generate_only:
        print(f"Candidate archive: {output / 'hybrid_shadows.npz'}", flush=True)
        return

    print(f"Building ftPBE grid level {args.grid_level}...", flush=True)
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    exact_ftpbe = objective.evaluate(
        exact.exact_d2, exact.exact_gamma, gradient=False
    ).total_energy
    rows = []
    current_d2 = None
    current_gamma = None
    for count in args.counts:
        checkpoint_npz = output / "checkpoints" / f"shadows_{count:04d}.npz"
        checkpoint_json = output / "checkpoints" / f"shadows_{count:04d}.json"
        if checkpoint_npz.is_file() and checkpoint_json.is_file() and not args.overwrite:
            with np.load(checkpoint_npz, allow_pickle=False) as arrays:
                current_d2 = np.asarray(arrays["d2"], dtype=float)
                current_gamma = np.asarray(arrays["gamma"], dtype=float)
            row = json.loads(checkpoint_json.read_text())
            rows.append(row)
            print(f"[m={count:02d}] checkpoint", flush=True)
            continue

        real_d2, real_gamma = _load_real_checkpoint(args.system, real_root, count)
        acquired = acquire_shadow_bases(hybrid, tuple(range(count)))
        reconstruction: dict[str, Any] = {}
        if (
            not np.any(is_complex[:count])
            and args.shadow_data == "finite_shot"
            and args.selector == "energy"
            and args.fit_mode == "paired"
        ):
            result = SimpleNamespace(
                d2=real_d2,
                gamma=real_gamma,
                energy=float(
                    np.sum(selection.one_body * real_gamma)
                    + np.sum(selection.two_body * real_d2)
                    + selection.nuclear_energy
                ),
                status="reused_exactly_paired_real_prefix",
                fit_status="unchanged_before_first_complex_frame",
            )
            elapsed = 0.0
            source = "paired real checkpoint"
        else:
            if current_d2 is None or current_gamma is None:
                if count > 1:
                    try:
                        current_d2, current_gamma = _load_real_checkpoint(
                            args.system, real_root, count - 1
                        )
                    except FileNotFoundError:
                        current_d2 = real_d2.copy()
                        current_gamma = real_gamma.copy()
                else:
                    initial = _initial_result(
                        solver_args, LeakGuardReference(selection)
                    )
                    current_d2 = np.asarray(initial.d2, dtype=float)
                    current_gamma = np.asarray(initial.gamma, dtype=float)
            print(
                f"[m={count:02d}] solving {int(np.count_nonzero(is_complex[:count]))} "
                "complex + real frames...",
                flush=True,
            )
            started = time.perf_counter()
            result, reconstruction = _solve_shadow_dqg(
                solver_args,
                LeakGuardReference(selection),
                acquired,
                current_d2,
                current_gamma,
            )
            elapsed = time.perf_counter() - started
            source = (
                "exact-shadow hybrid MOSEK DQG"
                if args.shadow_data == "exact"
                else "hybrid MOSEK DQG"
            )
        row = _score_row(
            count,
            is_complex,
            acquired,
            result,
            real_d2,
            real_gamma,
            exact,
            objective,
            exact_ftpbe,
            elapsed,
            source,
            reconstruction,
        )
        rows.append(row)
        current_d2 = np.asarray(result.d2, dtype=float)
        current_gamma = np.asarray(result.gamma, dtype=float)
        _atomic_npz(checkpoint_npz, d2=current_d2, gamma=current_gamma)
        _atomic_json(checkpoint_json, row)
        print(
            f"[m={count:02d}] D2 real={row['real_d2_frobenius_error']:.6f}, "
            f"hybrid={row['hybrid_d2_frobenius_error']:.6f}, "
            f"ratio={row['hybrid_to_real_d2_error_ratio']:.3f}, "
            f"rank={row['design_rank']}/{row['design_variable_count']}, "
            f"{elapsed:.1f} s",
            flush=True,
        )

    _write_csv(output / "trajectory.csv", rows)
    _atomic_json(output / "analysis.json", {"configuration": configuration, "rows": rows})
    (output / "HYBRID_COMPLEX_ABLATION.md").write_text(
        _report(args.system, configuration, rows), encoding="utf-8"
    )
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
