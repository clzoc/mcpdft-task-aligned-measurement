"""Exact validation sampler for occupation measurements in orbital frames."""

from __future__ import annotations

import numpy as np


class FCIFrameSampler:
    """Sample full alpha/beta bitstrings after a real active-orbital rotation.

    This class is a validation backend, not a quantum simulator. It rotates a
    PySCF CI vector exactly and then samples the determinant distribution. The
    returned full bitstrings retain every within-frame covariance needed by
    the joint estimator.
    """

    def __init__(self, ci: np.ndarray, n_modes: int, n_electrons: tuple[int, int]):
        from pyscf.fci import cistring

        self.ci = np.asarray(ci)
        self.n_modes = int(n_modes)
        self.n_electrons = tuple(map(int, n_electrons))
        strings_alpha = cistring.make_strings(range(n_modes), self.n_electrons[0])
        strings_beta = cistring.make_strings(range(n_modes), self.n_electrons[1])
        self.occupation_alpha = np.asarray(
            [[(int(string) >> mode) & 1 for mode in range(n_modes)] for string in strings_alpha],
            dtype=np.int8,
        )
        self.occupation_beta = np.asarray(
            [[(int(string) >> mode) & 1 for mode in range(n_modes)] for string in strings_beta],
            dtype=np.int8,
        )

    def determinant_probabilities(self, frame: np.ndarray) -> np.ndarray:
        from pyscf import fci

        rotation = np.asarray(frame, dtype=float)
        if rotation.shape != (self.n_modes, self.n_modes):
            raise ValueError("frame has an incompatible shape")
        if np.max(np.abs(rotation @ rotation.T - np.eye(self.n_modes))) > 1e-9:
            raise ValueError("frame must be orthonormal")
        rotated_ci = fci.addons.transform_ci_for_orbital_rotation(
            self.ci,
            self.n_modes,
            self.n_electrons,
            rotation.T,
        )
        probability = np.abs(rotated_ci.ravel()) ** 2
        return np.asarray(probability / probability.sum(), dtype=float)

    def sample_bits(
        self, frame: np.ndarray, shots: int, rng: np.random.Generator
    ) -> np.ndarray:
        if shots < 1:
            raise ValueError("shots must be positive")
        probability = self.determinant_probabilities(frame)
        flat = rng.choice(len(probability), size=shots, p=probability)
        n_beta = len(self.occupation_beta)
        alpha_index = flat // n_beta
        beta_index = flat % n_beta
        return np.stack(
            [self.occupation_alpha[alpha_index], self.occupation_beta[beta_index]],
            axis=1,
        )

    def exact_moments(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probability = self.determinant_probabilities(frame).reshape(
            len(self.occupation_alpha), len(self.occupation_beta)
        )
        pair = np.einsum(
            "ab,ai,bi->i",
            probability,
            self.occupation_alpha,
            self.occupation_beta,
            optimize=True,
        )
        total = (
            probability.sum(axis=1) @ self.occupation_alpha
            + probability.sum(axis=0) @ self.occupation_beta
        )
        return np.asarray(pair), np.asarray(total)

