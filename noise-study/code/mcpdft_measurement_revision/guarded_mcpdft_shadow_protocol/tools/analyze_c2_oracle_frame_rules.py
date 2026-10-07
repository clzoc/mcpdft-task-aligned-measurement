#!/usr/bin/env python3
"""Extract observable frame features behind the C2 oracle subsets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.linalg import solve
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CODE = PROJECT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    LeakGuardShadowData,
    build_c2_selection_reference,
    contraction_gamma_gradient_vector,
    load_blind_shadow_npz,
    matrix_to_variable_vector,
    shadow_design_blocks,
    symmetric_matrix_gradient_vector,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from run_sweep import _atomic_json  # noqa: E402


MODEL_FEATURES = (
    "is_complex",
    "standardized_residual_rms",
    "absolute_ftpbe_pull_z",
    "absolute_hamiltonian_pull_z",
    "absolute_contrast_pull_z",
    "ftpbe_variance_loss_fraction",
    "hamiltonian_variance_loss_fraction",
    "trace_variance_loss_fraction",
    "d_optimal_loss",
    "d2_pull_frobenius",
    "target_information_alignment",
)
RANK_MODEL_FEATURES = (
    "is_complex",
    "rank_standardized_residual_rms",
    "rank_absolute_ftpbe_pull_z",
    "rank_absolute_hamiltonian_pull_z",
    "rank_absolute_contrast_pull_z",
    "rank_ftpbe_variance_loss_fraction",
    "rank_hamiltonian_variance_loss_fraction",
    "rank_trace_variance_loss_fraction",
    "rank_d_optimal_loss",
    "rank_d2_pull_frobenius",
    "target_information_alignment",
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--m50-dir", type=Path, required=True)
    parser.add_argument("--shadow-archive", type=Path, required=True)
    parser.add_argument("--frozen-mo-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--information-ridge-fraction", type=float, default=1e-3)
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _gradient_vectors(
    objective: FtPBEEnergyObjective,
    reference: Any,
    d2: np.ndarray,
    gamma: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    evaluation = objective.evaluate(d2, gamma, gradient=True)
    ftpbe = symmetric_matrix_gradient_vector(evaluation.d2_gradient, rows, cols)
    ftpbe += contraction_gamma_gradient_vector(
        evaluation.gamma_gradient,
        reference.pairs,
        rows,
        cols,
        reference.n_electrons,
    )
    hamiltonian = symmetric_matrix_gradient_vector(reference.two_body, rows, cols)
    hamiltonian += contraction_gamma_gradient_vector(
        reference.one_body,
        reference.pairs,
        rows,
        cols,
        reference.n_electrons,
    )
    return ftpbe, hamiltonian


def _d2_vector_norm(
    vector: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> float:
    off_diagonal = rows != cols
    return float(
        np.sqrt(
            np.sum(np.square(vector[~off_diagonal]))
            + 2.0 * np.sum(np.square(vector[off_diagonal]))
        )
    )


def _labels(oracle_dir: Path) -> dict[int, set[int]]:
    rank2 = json.loads(
        (oracle_dir / "rank2_validation.json").read_text(encoding="utf-8")
    )["rows"]
    primary = json.loads(
        (oracle_dir / "analysis.json").read_text(encoding="utf-8")
    )
    labels = {}
    for maximum in (20, 30):
        row = next(item for item in rank2 if int(item["maximum"]) == maximum)
        labels[maximum] = {
            int(item) - 1 for item in row["subset_one_based"].split(",")
        }
    row = primary["selected_subsets"]["40"]
    labels[40] = {
        int(item) - 1 for item in row["subset_one_based"].split(",")
    }
    return labels


def _checkpoint_path(
    maximum: int, baseline_dir: Path, oracle_dir: Path, m50_dir: Path
) -> Path:
    if maximum in (20, 30):
        return baseline_dir / "checkpoints" / f"shadows_{maximum:04d}.npz"
    if maximum == 40:
        return oracle_dir / "checkpoints" / "m40_full.npz"
    if maximum == 50:
        return m50_dir / "checkpoints" / "shadows_0050.npz"
    raise ValueError(f"Unsupported prefix {maximum}.")


def _prefix_features(
    maximum: int,
    d2: np.ndarray,
    gamma: np.ndarray,
    objective: FtPBEEnergyObjective,
    reference: Any,
    shadows: Any,
    blocks: Sequence[np.ndarray],
    is_complex: np.ndarray,
    variable_rows: np.ndarray,
    variable_cols: np.ndarray,
    ridge_fraction: float,
    kept: set[int] | None,
) -> list[dict[str, Any]]:
    variables = matrix_to_variable_vector(d2, variable_rows, variable_cols)
    ftpbe_gradient, hamiltonian_gradient = _gradient_vectors(
        objective,
        reference,
        d2,
        gamma,
        variable_rows,
        variable_cols,
    )
    observations_per_frame = len(shadows.values) // shadows.n_shadows
    frame_information = []
    frame_scores = []
    residual_rows = []
    for index in range(maximum):
        start = index * observations_per_frame
        stop = start + observations_per_frame
        block = np.asarray(blocks[index], dtype=float)
        hits = np.asarray(shadows.hits[start:stop], dtype=float)
        probabilities = (hits + 0.5) / (shadows.shots_per_basis + 1.0)
        variances = np.maximum(
            probabilities * (1.0 - probabilities) / shadows.shots_per_basis,
            1e-12,
        )
        inverse_variances = 1.0 / variances
        residual = np.asarray(shadows.values[start:stop]) - block @ variables
        weighted = np.sqrt(inverse_variances)[:, None] * block
        frame_information.append(weighted.T @ weighted)
        frame_scores.append(block.T @ (inverse_variances * residual))
        residual_rows.append(
            {
                "standardized_residual_rms": float(
                    np.sqrt(np.mean(np.square(residual) * inverse_variances))
                ),
                "mean_absolute_residual": float(np.mean(np.abs(residual))),
                "mean_binomial_variance": float(np.mean(variances)),
            }
        )
    frame_information_array = np.asarray(frame_information)
    frame_scores_array = np.asarray(frame_scores)
    variable_count = len(variable_rows)
    scale = max(
        float(np.trace(frame_information_array[0]) / variable_count), 1e-12
    )
    information = ridge_fraction * scale * np.eye(variable_count)
    information += np.sum(frame_information_array, axis=0)
    inverse_information = solve(
        information, np.eye(variable_count), assume_a="pos", check_finite=False
    )
    sign, logdet = np.linalg.slogdet(information)
    if sign <= 0:
        raise np.linalg.LinAlgError("Prefix information is not positive definite.")
    current_trace = float(np.trace(inverse_information))
    current_ftpbe_variance = float(
        ftpbe_gradient @ inverse_information @ ftpbe_gradient
    )
    current_hamiltonian_variance = float(
        hamiltonian_gradient @ inverse_information @ hamiltonian_gradient
    )

    rows: list[dict[str, Any]] = []
    ftpbe_pulls = []
    hamiltonian_pulls = []
    for index in range(maximum):
        update = inverse_information @ frame_scores_array[index]
        ftpbe_pull = float(ftpbe_gradient @ update)
        hamiltonian_pull = float(hamiltonian_gradient @ update)
        ftpbe_noise_variance = float(
            ftpbe_gradient
            @ inverse_information
            @ frame_information_array[index]
            @ inverse_information
            @ ftpbe_gradient
        )
        hamiltonian_noise_variance = float(
            hamiltonian_gradient
            @ inverse_information
            @ frame_information_array[index]
            @ inverse_information
            @ hamiltonian_gradient
        )
        cross_information = float(
            ftpbe_gradient
            @ inverse_information
            @ frame_information_array[index]
            @ inverse_information
            @ hamiltonian_gradient
        )
        without = information - frame_information_array[index]
        inverse_without = solve(
            without, np.eye(variable_count), assume_a="pos", check_finite=False
        )
        without_trace = float(np.trace(inverse_without))
        without_ftpbe_variance = float(
            ftpbe_gradient @ inverse_without @ ftpbe_gradient
        )
        without_hamiltonian_variance = float(
            hamiltonian_gradient @ inverse_without @ hamiltonian_gradient
        )
        without_sign, without_logdet = np.linalg.slogdet(without)
        if without_sign <= 0:
            raise np.linalg.LinAlgError(
                "Leave-one-frame information is not positive definite."
            )
        alignment_denominator = np.sqrt(
            max(ftpbe_noise_variance * hamiltonian_noise_variance, 0.0)
        )
        row = {
            "maximum": maximum,
            "frame_zero_based": index,
            "frame_one_based": index + 1,
            "position_fraction": (index + 1) / maximum,
            "is_complex": int(bool(is_complex[index])),
            "oracle_kept": (int(index in kept) if kept is not None else None),
            **residual_rows[index],
            "ftpbe_pull_meh": 1000.0 * ftpbe_pull,
            "hamiltonian_pull_meh": 1000.0 * hamiltonian_pull,
            "absolute_ftpbe_pull_z": abs(ftpbe_pull)
            / np.sqrt(max(ftpbe_noise_variance, 1e-30)),
            "absolute_hamiltonian_pull_z": abs(hamiltonian_pull)
            / np.sqrt(max(hamiltonian_noise_variance, 1e-30)),
            "d2_pull_frobenius": _d2_vector_norm(
                update, variable_rows, variable_cols
            ),
            "ftpbe_variance_loss_fraction": (
                without_ftpbe_variance / current_ftpbe_variance - 1.0
            ),
            "hamiltonian_variance_loss_fraction": (
                without_hamiltonian_variance
                / current_hamiltonian_variance
                - 1.0
            ),
            "trace_variance_loss_fraction": without_trace / current_trace - 1.0,
            "d_optimal_loss": float(logdet - without_logdet),
            "target_information_alignment": (
                cross_information / alignment_denominator
                if alignment_denominator > 0.0
                else 0.0
            ),
        }
        rows.append(row)
        ftpbe_pulls.append(ftpbe_pull)
        hamiltonian_pulls.append(hamiltonian_pull)

    ftpbe_array = np.asarray(ftpbe_pulls)
    hamiltonian_array = np.asarray(hamiltonian_pulls)
    centered_ftpbe = ftpbe_array - np.median(ftpbe_array)
    centered_hamiltonian = hamiltonian_array - np.median(hamiltonian_array)
    denominator = float(centered_hamiltonian @ centered_hamiltonian)
    beta = (
        float(centered_hamiltonian @ centered_ftpbe / denominator)
        if denominator > 0.0
        else 0.0
    )
    contrast = centered_ftpbe - beta * centered_hamiltonian
    median_contrast = float(np.median(contrast))
    contrast_scale = max(
        1.4826 * float(np.median(np.abs(contrast - median_contrast))), 1e-12
    )
    for row, value in zip(rows, contrast):
        row["ftpbe_hamiltonian_pull_beta"] = beta
        row["absolute_contrast_pull_z"] = abs(value - median_contrast) / contrast_scale
        row["low_information_conflict_score"] = (
            row["absolute_ftpbe_pull_z"]
            / np.sqrt(max(row["ftpbe_variance_loss_fraction"], 1e-12))
        )
    rank_features = (
        "standardized_residual_rms",
        "absolute_ftpbe_pull_z",
        "absolute_hamiltonian_pull_z",
        "absolute_contrast_pull_z",
        "ftpbe_variance_loss_fraction",
        "hamiltonian_variance_loss_fraction",
        "trace_variance_loss_fraction",
        "d_optimal_loss",
        "d2_pull_frobenius",
    )
    for feature in rank_features:
        values = np.asarray([float(row[feature]) for row in rows])
        percentiles = (rankdata(values, method="average") - 0.5) / len(values)
        for row, percentile in zip(rows, percentiles):
            row[f"rank_{feature}"] = float(percentile)
    return rows


def _feature_summary(
    rows: Sequence[dict[str, Any]], features: Sequence[str]
) -> dict[str, Any]:
    training = [row for row in rows if row["oracle_kept"] is not None]
    y = np.asarray([row["oracle_kept"] for row in training], dtype=int)
    feature_rows = []
    for feature in features:
        values = np.asarray([float(row[feature]) for row in training])
        kept = values[y == 1]
        excluded = values[y == 0]
        raw_auc = float(roc_auc_score(y, values))
        feature_rows.append(
            {
                "feature": feature,
                "kept_median": float(np.median(kept)),
                "excluded_median": float(np.median(excluded)),
                "auc_high_means_kept": raw_auc,
                "best_univariate_auc": max(raw_auc, 1.0 - raw_auc),
                "direction": "high_kept" if raw_auc >= 0.5 else "low_kept",
            }
        )

    x = np.asarray(
        [[float(row[feature]) for feature in features] for row in training]
    )
    groups = np.asarray([row["maximum"] for row in training], dtype=int)
    cross_validation = []
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = groups == held_out
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.2,
                class_weight="balanced",
                max_iter=5000,
                random_state=17,
            ),
        )
        model.fit(x[train], y[train])
        probabilities = model.predict_proba(x[test])[:, 1]
        predictions = probabilities >= 0.5
        cross_validation.append(
            {
                "held_out_maximum": int(held_out),
                "roc_auc": float(roc_auc_score(y[test], probabilities)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[test], predictions)
                ),
            }
        )
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=5000,
            random_state=17,
        ),
    )
    final_model.fit(x, y)
    scaler = final_model.named_steps["standardscaler"]
    logistic = final_model.named_steps["logisticregression"]
    coefficients = [
        {
            "feature": feature,
            "standardized_coefficient": float(coefficient),
        }
        for feature, coefficient in zip(features, logistic.coef_[0])
    ]
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=23,
    )
    tree.fit(x, y)
    return {
        "samples": len(training),
        "kept": int(np.count_nonzero(y)),
        "excluded": int(np.count_nonzero(1 - y)),
        "feature_summaries": sorted(
            feature_rows,
            key=lambda row: row["best_univariate_auc"],
            reverse=True,
        ),
        "leave_prefix_out": cross_validation,
        "logistic_coefficients": sorted(
            coefficients,
            key=lambda row: abs(row["standardized_coefficient"]),
            reverse=True,
        ),
        "decision_tree": export_text(tree, feature_names=list(features)),
        "model_feature_order": list(features),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "logistic_intercept": float(logistic.intercept_[0]),
        "logistic_coefficients_raw_order": logistic.coef_[0].tolist(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    for name in (
        "oracle_dir",
        "baseline_dir",
        "m50_dir",
        "shadow_archive",
        "frozen_mo_npz",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(args.oracle_dir)
    shadows = load_blind_shadow_npz(args.shadow_archive, load_design=False)
    with np.load(args.shadow_archive, allow_pickle=False) as archive:
        is_complex = np.asarray(archive["is_complex_basis"], dtype=bool)
    with np.load(args.frozen_mo_npz, allow_pickle=False) as archive:
        frozen_mo = np.asarray(archive["mo_coeff"], dtype=float)
    selection = LeakGuardReference(
        build_c2_selection_reference(
            1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo
        )
    )
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    variable_rows, variable_cols = _symmetric_d2_variables(selection)
    blocks = shadow_design_blocks(
        LeakGuardShadowData.from_shadow_data(shadows),
        len(selection.pairs),
        variable_rows,
        variable_cols,
    )
    rows: list[dict[str, Any]] = []
    for maximum in (20, 30, 40, 50):
        checkpoint = _checkpoint_path(
            maximum, args.baseline_dir, args.oracle_dir, args.m50_dir
        )
        with np.load(checkpoint, allow_pickle=False) as archive:
            d2 = np.asarray(archive["d2"])
            gamma = np.asarray(archive["gamma"])
        rows.extend(
            _prefix_features(
                maximum,
                d2,
                gamma,
                objective,
                selection,
                shadows,
                blocks,
                is_complex,
                variable_rows,
                variable_cols,
                args.information_ridge_fraction,
                labels.get(maximum),
            )
        )
        print(f"[M={maximum}] extracted {maximum} frame rows", flush=True)
    summary = {
        "raw_feature_model": _feature_summary(rows, MODEL_FEATURES),
        "rank_feature_model": _feature_summary(rows, RANK_MODEL_FEATURES),
    }
    _write_csv(args.output_dir / "frame_features.csv", rows)
    _atomic_json(
        args.output_dir / "analysis.json",
        {
            "configuration": {
                "features_use_exact": False,
                "oracle_labels_use_exact": True,
                "training_prefixes": [20, 30, 40],
                "locked_feature_prefix": 50,
                "model_features": list(MODEL_FEATURES),
                "rank_model_features": list(RANK_MODEL_FEATURES),
            },
            "summary": summary,
        },
    )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
