#!/usr/bin/env python3
"""Audit adaptive ftPBE-targeted frame selection with shot reallocation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CODE = PROJECT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    PairVectorDesign,
    ShadowData,
    _wilson_bounds,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    LeakGuardShadowData,
    build_c2_reference,
    build_c2_selection_reference,
    contraction_gamma_gradient_vector,
    guarded_target_subspace_choice,
    load_blind_shadow_npz,
    matrix_to_variable_vector,
    shadow_design_blocks,
    symmetric_matrix_gradient_vector,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from run_hybrid_complex_shadow_ablation import (  # noqa: E402
    _pair_indicators,
    _statevector_to_ci,
)
from run_safe_mcpdft_derandomization import (  # noqa: E402
    RECONSTRUCTION_ENERGY,
    _initial_result,
    _solve_shadow_dqg,
)
from run_sweep import _atomic_json, _atomic_npz  # noqa: E402


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--frozen-mo-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots-per-block", type=int, default=10_000)
    parser.add_argument("--max-blocks", type=int, default=10)
    parser.add_argument("--gradient-window", type=int, default=4)
    parser.add_argument("--gradient-decay", type=float, default=0.75)
    parser.add_argument("--design-guard-fraction", type=float, default=0.1)
    parser.add_argument("--information-ridge-fraction", type=float, default=1e-3)
    parser.add_argument("--probability-floor", type=float, default=0.01)
    parser.add_argument("--sampling-seed", type=int, default=424242)
    parser.add_argument(
        "--fixed-indices",
        help=(
            "Comma-separated zero-based candidate indices for a frozen baseline "
            "path; omit for adaptive ftPBE selection."
        ),
    )
    parser.add_argument("--target-meh", type=float, default=1.6)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _parse_fixed_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("fixed-indices must be comma-separated integers.") from error
    if not indices or min(indices) < 0:
        raise ValueError("fixed-indices must contain nonnegative integers.")
    return indices


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _total_ftpbe_gradient(
    objective: FtPBEEnergyObjective,
    reference: Any,
    d2: np.ndarray,
    gamma: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    evaluation = objective.evaluate(d2, gamma, gradient=True)
    gradient = symmetric_matrix_gradient_vector(
        evaluation.d2_gradient, rows, cols
    )
    gradient += contraction_gamma_gradient_vector(
        evaluation.gamma_gradient,
        reference.pairs,
        rows,
        cols,
        reference.n_electrons,
    )
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-14:
        raise RuntimeError("The total-contracted ftPBE gradient is zero.")
    return gradient / norm


def _gradient_modes(
    gradients: Sequence[np.ndarray], window: int, decay: float
) -> np.ndarray:
    selected = tuple(gradients[-window:])
    ages = np.arange(len(selected) - 1, -1, -1, dtype=float)
    weights = decay**ages
    weights /= np.sum(weights)
    return np.sqrt(weights)[:, None] * np.asarray(selected)


class AcquisitionOracle:
    """Generate bitstrings only after a candidate frame has been selected."""

    def __init__(
        self,
        reference: Any,
        rotations: Sequence[np.ndarray],
        pair_vectors: np.ndarray,
        shots_per_block: int,
        seed: int,
    ):
        self.reference = reference
        self.rotations = tuple(np.asarray(rotation) for rotation in rotations)
        self.pair_vectors = np.asarray(pair_vectors)
        self.shots_per_block = int(shots_per_block)
        self.seed = int(seed)
        self.ci, determinant_occupations = _statevector_to_ci(reference)
        self.determinant_occupations = np.asarray(
            determinant_occupations, dtype=np.uint64
        ).reshape(-1)
        self.indicators = _pair_indicators(
            self.determinant_occupations, reference.pairs
        )
        self._probabilities: dict[int, np.ndarray] = {}

    def _frame_probabilities(self, index: int) -> np.ndarray:
        if index not in self._probabilities:
            from pyscf import fci

            rotated_ci = fci.addons.transform_ci_for_orbital_rotation(
                self.ci,
                self.reference.n_spatial_orbitals,
                (self.reference.n_alpha, self.reference.n_beta),
                self.rotations[index],
            )
            probabilities = np.abs(np.asarray(rotated_ci).reshape(-1)) ** 2
            self._probabilities[index] = np.asarray(
                probabilities / probabilities.sum(), dtype=float
            )
        return self._probabilities[index]

    def sample(self, index: int, repeat: int) -> dict[str, np.ndarray | int]:
        probabilities = self._frame_probabilities(index)
        rng = np.random.default_rng([self.seed, int(index), int(repeat)])
        sampled = rng.choice(
            len(probabilities), size=self.shots_per_block, p=probabilities
        )
        hits = np.asarray(self.indicators[sampled].sum(axis=0), dtype=int)
        start = index * len(self.reference.pairs)
        stop = start + len(self.reference.pairs)
        return {
            "candidate_index": int(index),
            "rotation": self.rotations[index],
            "pair_vectors": self.pair_vectors[start:stop],
            "hits": hits,
            "occupations": self.determinant_occupations[sampled],
        }


def _records_to_shadows(
    records: Sequence[dict[str, np.ndarray | int]], shots_per_block: int
) -> ShadowData:
    hits = np.concatenate([np.asarray(record["hits"], dtype=int) for record in records])
    values = hits / shots_per_block
    lower, upper = _wilson_bounds(hits, shots_per_block, 2.0)
    pair_vectors = np.vstack(
        [np.asarray(record["pair_vectors"]) for record in records]
    )
    return ShadowData(
        rotations=tuple(np.asarray(record["rotation"]) for record in records),
        pair_vectors=pair_vectors,
        design=PairVectorDesign(pair_vectors),
        values=values,
        lower_bounds=lower,
        upper_bounds=upper,
        hits=hits,
        shots_per_basis=shots_per_block,
        exact_values=np.full_like(values, np.nan),
        exact_constraints=False,
        occupations=np.vstack(
            [np.asarray(record["occupations"], dtype=np.uint64) for record in records]
        ),
    )


def _first_stable_hit(
    errors_meh: Sequence[float], threshold: float, consecutive: int = 3
) -> int | None:
    if consecutive < 1:
        raise ValueError("consecutive must be positive.")
    for index in range(len(errors_meh) - consecutive + 1):
        if max(errors_meh[index : index + consecutive]) <= threshold:
            return index + 1
    return None


def _report(
    configuration: dict[str, Any], rows: Sequence[dict[str, Any]], summary: dict[str, Any]
) -> str:
    lines = [
        "# C2 ftPBE shot-allocation audit",
        "",
        "Selection uses only candidate designs, acquired bitstrings, and a rolling "
        "ensemble of total-contracted ftPBE gradients. Exact RDM data are used only "
        "by the acquisition simulator and for posthoc scoring after the path freezes.",
        "",
        f"- Shots per allocation block: {configuration['shots_per_block']}",
        f"- Selection policy: {configuration['selection_policy']}",
        f"- Target: {configuration['target_meh']:.3f} mEh",
        f"- First hit: {summary['first_hit_block']}",
        "- First stable hit (three consecutive blocks): "
        f"{summary['first_stable_hit_block']}",
        f"- Endpoint distinct frames: {summary['endpoint_distinct_frames']}",
        "",
        "| blocks | distinct | complex blocks | ftPBE error (mEh) | fit RMSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['blocks']} | {row['distinct_frames']} | "
            f"{row['complex_blocks']} | {row['ftpbe_error_meh']:.6f} | "
            f"{row['weighted_fit_rmse']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    if args.shots_per_block < 1 or args.max_blocks < 1:
        raise ValueError("Shot and block counts must be positive.")
    if args.gradient_window < 1 or not 0.0 < args.gradient_decay <= 1.0:
        raise ValueError("Invalid gradient ensemble parameters.")
    args.candidate_pool = args.candidate_pool.resolve()
    args.frozen_mo_npz = args.frozen_mo_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_indices = _parse_fixed_indices(args.fixed_indices)

    with np.load(args.frozen_mo_npz, allow_pickle=False) as archive:
        frozen_mo = np.asarray(archive["mo_coeff"], dtype=float)
    with np.load(args.candidate_pool, allow_pickle=False) as archive:
        is_complex = np.asarray(archive["is_complex_basis"], dtype=bool)

    selection_base = build_c2_selection_reference(
        1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo
    )
    selection_reference = LeakGuardReference(selection_base)
    candidate_shadows = load_blind_shadow_npz(
        args.candidate_pool, load_design=False
    )
    if fixed_indices is not None:
        if len(fixed_indices) < args.max_blocks:
            raise ValueError("fixed-indices is shorter than max-blocks.")
        if max(fixed_indices) >= candidate_shadows.n_shadows:
            raise ValueError("fixed-indices contains an out-of-range candidate.")
    objective = FtPBEEnergyObjective(
        selection_reference, grid_level=args.grid_level
    )
    variable_rows, variable_cols = _symmetric_d2_variables(selection_reference)
    blocks = shadow_design_blocks(
        LeakGuardShadowData.from_shadow_data(candidate_shadows),
        len(selection_reference.pairs),
        variable_rows,
        variable_cols,
    )

    solver_args = SimpleNamespace(
        solver=args.solver.upper(),
        solver_tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        verbose_solver=False,
        weighted_fit_rmse_cap=None,
        reconstruction_objective=RECONSTRUCTION_ENERGY,
        ridge_grid=(0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0),
        ridge_folds=5,
        fallback_ridge=1e-2,
        symmetry_blocked_psd=True,
    )
    print("Solving no-data DQG initialization...", flush=True)
    initial = _initial_result(solver_args, selection_reference)

    # The exact state acts only as the simulated quantum acquisition device.
    acquisition_reference = build_c2_reference(
        1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo
    )
    oracle = AcquisitionOracle(
        acquisition_reference,
        candidate_shadows.rotations,
        candidate_shadows.pair_vectors,
        args.shots_per_block,
        args.sampling_seed,
    )

    gradients = [
        _total_ftpbe_gradient(
            objective,
            selection_reference,
            initial.d2,
            initial.gamma,
            variable_rows,
            variable_cols,
        )
    ]
    records: list[dict[str, np.ndarray | int]] = []
    selected_indices: list[int] = []
    repeat_counts = np.zeros(candidate_shadows.n_shadows, dtype=int)
    blind_rows: list[dict[str, Any]] = []
    current_d2 = np.asarray(initial.d2)
    current_gamma = np.asarray(initial.gamma)

    if not args.overwrite:
        for block_count in range(1, args.max_blocks + 1):
            checkpoint_npz = (
                args.output_dir / "checkpoints" / f"blocks_{block_count:04d}.npz"
            )
            checkpoint_json = (
                args.output_dir / "checkpoints" / f"blocks_{block_count:04d}.json"
            )
            if not checkpoint_npz.exists() and not checkpoint_json.exists():
                break
            if not checkpoint_npz.is_file() or not checkpoint_json.is_file():
                raise RuntimeError(
                    f"Incomplete checkpoint pair for block {block_count}."
                )
            row = json.loads(checkpoint_json.read_text(encoding="utf-8"))
            if int(row["blocks"]) != block_count:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_json} has an inconsistent block count."
                )
            index = int(row["selected_index"])
            repeat = int(row["selected_repeat"])
            if repeat != int(repeat_counts[index]):
                raise RuntimeError(
                    f"Checkpoint {checkpoint_json} has an inconsistent repeat count."
                )
            repeat_counts[index] += 1
            selected_indices.append(index)
            records.append(oracle.sample(index, repeat))
            with np.load(checkpoint_npz, allow_pickle=False) as archive:
                current_d2 = np.asarray(archive["d2"])
                current_gamma = np.asarray(archive["gamma"])
            gradients.append(
                _total_ftpbe_gradient(
                    objective,
                    selection_reference,
                    current_d2,
                    current_gamma,
                    variable_rows,
                    variable_cols,
                )
            )
            blind_rows.append(row)
        if blind_rows:
            print(
                f"Resuming after {len(blind_rows)} allocation blocks from blind "
                "checkpoints...",
                flush=True,
            )

    for block_count in range(len(blind_rows) + 1, args.max_blocks + 1):
        variables = matrix_to_variable_vector(
            current_d2, variable_rows, variable_cols
        )
        if fixed_indices is None:
            modes = _gradient_modes(
                gradients, args.gradient_window, args.gradient_decay
            )
            choice = guarded_target_subspace_choice(
                blocks,
                selected_indices,
                variables,
                modes,
                args.shots_per_block,
                design_guard_fraction=args.design_guard_fraction,
                ridge_fraction=args.information_ridge_fraction,
                probability_floor=args.probability_floor,
                allow_repeats=True,
            )
            index = int(choice.selected_index)
            target_gain = float(choice.selected.mcpdft_fractional_reduction)
            design_gain_fraction = float(
                choice.selected.d_optimal_gain / choice.maximum_d_optimal_gain
            )
        else:
            index = int(fixed_indices[block_count - 1])
            target_gain = None
            design_gain_fraction = None
        repeat = int(repeat_counts[index])
        repeat_counts[index] += 1
        selected_indices.append(index)
        records.append(oracle.sample(index, repeat))
        acquired = _records_to_shadows(records, args.shots_per_block)
        print(
            f"[block={block_count:02d}] candidate={index:02d}, repeat={repeat}, "
            f"distinct={len(set(selected_indices))}, "
            f"target-gain={target_gain:.3%}"
            if target_gain is not None
            else f"[block={block_count:02d}] candidate={index:02d}, "
            f"repeat={repeat}, distinct={len(set(selected_indices))}, fixed-path",
            flush=True,
        )
        started = time.perf_counter()
        result, reconstruction = _solve_shadow_dqg(
            solver_args,
            selection_reference,
            acquired,
            current_d2,
            current_gamma,
        )
        elapsed = time.perf_counter() - started
        current_d2 = np.asarray(result.d2)
        current_gamma = np.asarray(result.gamma)
        evaluation = objective.evaluate(current_d2, current_gamma, gradient=False)
        gradients.append(
            _total_ftpbe_gradient(
                objective,
                selection_reference,
                current_d2,
                current_gamma,
                variable_rows,
                variable_cols,
            )
        )
        row = {
            "blocks": block_count,
            "total_shots": block_count * args.shots_per_block,
            "selected_index": index,
            "selected_repeat": repeat,
            "selected_indices": ",".join(map(str, selected_indices)),
            "distinct_frames": len(set(selected_indices)),
            "complex_blocks": int(np.count_nonzero(is_complex[selected_indices])),
            "distinct_complex_frames": len(
                {item for item in selected_indices if is_complex[item]}
            ),
            "blind_ftpbe_energy_eh": float(evaluation.total_energy),
            "weighted_fit_rmse": float(
                np.sqrt(np.mean((acquired.predict(current_d2) - acquired.values) ** 2))
            ),
            "selected_target_fractional_reduction": target_gain,
            "selected_design_gain_fraction": design_gain_fraction,
            "solver_seconds": elapsed,
            "fit_status": str(result.fit_status),
            **reconstruction,
        }
        blind_rows.append(row)
        _atomic_npz(
            args.output_dir / "checkpoints" / f"blocks_{block_count:04d}.npz",
            d2=current_d2,
            gamma=current_gamma,
        )
        _atomic_json(
            args.output_dir / "checkpoints" / f"blocks_{block_count:04d}.json",
            row,
        )
        print(
            f"[block={block_count:02d}] E_ftPBE={evaluation.total_energy:.9f}, "
            f"solve={elapsed:.1f}s",
            flush=True,
        )

    exact_ftpbe = objective.evaluate(
        acquisition_reference.exact_d2,
        acquisition_reference.exact_gamma,
        gradient=False,
    ).total_energy
    rows = []
    for blind in blind_rows:
        count = int(blind["blocks"])
        with np.load(
            args.output_dir / "checkpoints" / f"blocks_{count:04d}.npz",
            allow_pickle=False,
        ) as archive:
            d2 = np.asarray(archive["d2"])
            gamma = np.asarray(archive["gamma"])
        ftpbe = objective.evaluate(d2, gamma, gradient=False).total_energy
        hamiltonian = float(
            np.sum(acquisition_reference.one_body * gamma)
            + np.sum(acquisition_reference.two_body * d2)
            + acquisition_reference.nuclear_energy
        )
        rows.append(
            {
                **blind,
                "ftpbe_error_meh": 1000.0 * abs(ftpbe - exact_ftpbe),
                "d2_frobenius_error": float(
                    np.linalg.norm(d2 - acquisition_reference.exact_d2)
                ),
                "hamiltonian_error_meh": 1000.0
                * abs(hamiltonian - acquisition_reference.exact_energy),
            }
        )
    errors = [float(row["ftpbe_error_meh"]) for row in rows]
    first_hit = next(
        (index + 1 for index, value in enumerate(errors) if value <= args.target_meh),
        None,
    )
    stable_hit = _first_stable_hit(errors, args.target_meh)
    configuration = {
        "candidate_pool": str(args.candidate_pool),
        "candidate_real_frames": int(np.count_nonzero(~is_complex)),
        "candidate_complex_frames": int(np.count_nonzero(is_complex)),
        "shots_per_block": args.shots_per_block,
        "sampling_seed": args.sampling_seed,
        "max_blocks": args.max_blocks,
        "gradient_window": args.gradient_window,
        "gradient_decay": args.gradient_decay,
        "design_guard_fraction": args.design_guard_fraction,
        "selection_policy": (
            "rolling_ftPBE_gradient_A-optimal"
            if fixed_indices is None
            else "frozen_random_baseline"
        ),
        "fixed_indices": list(fixed_indices) if fixed_indices is not None else None,
        "target_meh": args.target_meh,
        "stable_hit_consecutive_blocks": 3,
        "selection_uses_exact_rdm": False,
        "exact_state_use": "chosen-frame bitstring simulation and frozen-path scoring only",
    }
    summary = {
        "exact_ftpbe_energy_eh": exact_ftpbe,
        "first_hit_block": first_hit,
        "first_stable_hit_block": stable_hit,
        "endpoint_error_meh": errors[-1],
        "endpoint_distinct_frames": rows[-1]["distinct_frames"],
        "endpoint_total_shots": rows[-1]["total_shots"],
        "endpoint_complex_blocks": rows[-1]["complex_blocks"],
    }
    _atomic_json(args.output_dir / "configuration.json", configuration)
    _atomic_json(
        args.output_dir / "analysis.json",
        {"configuration": configuration, "summary": summary, "rows": rows},
    )
    _write_csv(args.output_dir / "trajectory.csv", rows)
    (args.output_dir / "FTPBE_SHOT_REALLOCATION.md").write_text(
        _report(configuration, rows, summary), encoding="utf-8"
    )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
