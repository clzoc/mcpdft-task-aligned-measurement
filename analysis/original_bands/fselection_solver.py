#!/usr/bin/env python3
"""Reusable band solver with optional extra linear RDM objectives.

Keeps the frozen ``reusable_band_solver.py`` untouched (the three-frame freeze
hashes it). The vendor builder already accepts ``additional_d2_objective`` and
``additional_gamma_objective``; this module temporarily wraps the builder hook
inside ``reusable_band_solver`` so those constants reach the same graph.
"""
from __future__ import annotations

import contextlib

import reusable_band_solver as rbs


@contextlib.contextmanager
def _patched_builder(d2_objective, gamma_objective, nucleus_weight):
    original = rbs._make_builder

    def wrapper(band, spin_constrained, observation_basis):
        build, digest = original(band, spin_constrained, observation_basis)

        def build_with_objectives(*args, **kwargs):
            if d2_objective is not None:
                kwargs["additional_d2_objective"] = d2_objective
            if gamma_objective is not None:
                kwargs["additional_gamma_objective"] = gamma_objective
            if nucleus_weight is not None:
                kwargs["shadow_error_weight"] = nucleus_weight
            return build(*args, **kwargs)

        return build_with_objectives, digest

    rbs._make_builder = wrapper
    try:
        yield
    finally:
        rbs._make_builder = original


def make(context, raw_shadow, basis="spin", *, band=None, observation_basis=None,
         solver_threads=1, additional_d2_objective=None,
         additional_gamma_objective=None, nucleus_weight=None):
    with _patched_builder(additional_d2_objective, additional_gamma_objective,
                          nucleus_weight):
        return rbs.ReusableBandSolver(
            context, raw_shadow, basis, band=band,
            observation_basis=observation_basis, solver_threads=solver_threads)
