#!/usr/bin/env python3
"""Pilot-only backward frame screening; all 30 pilots count toward the budget."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve

from allocate_measurements import OUT, PILOT, save


def screen(designs, covariances, h, f, budgets):
    dimension = designs.shape[-1]
    number = len(designs)
    grams = np.array([a.T @ a for a in designs])
    noises = np.array([a.T @ v @ a for a, v in zip(designs, covariances)])
    energy = np.stack((h, f))
    def covariance(active):
        gram = grams[active].sum(0)
        eigenvalues = eigh(gram, eigvals_only=True)
        if eigenvalues[0] < eigenvalues[-1] * 1e-10:
            return None
        inverse = cho_solve(cho_factor(gram), np.eye(dimension))
        return inverse @ noises[active].sum(0) @ inverse
    full = covariance(list(range(number)))
    full_energy = energy @ full @ energy.T
    values, vectors = eigh(full_energy)
    whitening = (vectors / np.sqrt(values)).T
    def risks(covariance, count, budget):
        # Selected pilots are reused; every unused pilot still costs 500 shots.
        n = PILOT + (budget-number*PILOT)/count
        ratio = budget / number / n
        joint = ratio * whitening @ energy @ covariance @ energy.T @ whitening.T
        d2 = float(ratio * np.trace(covariance) / np.trace(full))
        plane = float(eigh(joint, eigvals_only=True)[-1])
        return dict(worst=max(d2, plane), d2=d2, plane=plane,
                    h=float(ratio*(h @ covariance @ h)/full_energy[0, 0]),
                    f=float(ratio*(f @ covariance @ f)/full_energy[1, 1]))
    output = {}
    for budget in budgets:
        active = list(range(number))
        path = [dict(indices=active.copy(), **risks(full, number, budget))]
        while len(active) > 2:
            candidates = []
            for frame in active:
                subset = [i for i in active if i != frame]
                cov = covariance(subset)
                if cov is not None:
                    candidates.append(dict(indices=subset, removed=frame,
                                           **risks(cov, len(subset), budget)))
            if not candidates:
                break
            best = min(candidates, key=lambda item: (item["worst"], item["removed"]))
            active = best["indices"]
            path.append(best)
        output[budget] = dict(best=min(path, key=lambda item:item["worst"]), path=path)
        print(budget, output[budget]["best"], flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", type=int, default=0)
    args = parser.parse_args()
    model = np.load(OUT / "joint" / f"r{args.stream}" / "pilot_model.npz")
    result = screen(model["designs"], model["covariances"], model["h"], model["f"],
                    (30000, 60000, 120000))
    save(OUT / f"subset_screen_r{args.stream}.json", result)
