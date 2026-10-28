"""Uniform arm interface: torch baselines and evoxc engine routes behind one API.

An *arm* answers the three evaluation workloads of a cell:

* ``sr_fitness()`` -> ``(pop,)`` positive MSE loss (``Forest.SR_fitness``);
* ``batch_forward()`` -> ``(pop, N, out)`` (``Forest.batch_forward``);
* ``forward()`` -> ``(pop, out)`` (``Forest.forward``, tree ``i`` sees row ``i``);

plus ``fitness(forest)`` -- the maximized-fitness callable a
``StandardPipeline`` generation step needs for *any* forest (the population
changes every generation, so torch arms rebuild their decode/plan per step and
that rebuild is charged to the eval side, exactly like the engine's own host
decode is).

The torch arms are created once per workload (decode/plan = setup, reported
separately, never inside a timed region).  The evoxc arms are the library's own
public methods with the arm's ``EVOGP_EVAL_MODE`` / ``EVOGP_EVOXC_BACKEND`` in
the environment -- nothing is monkeypatched, so what they measure is what a user
gets.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import torch

import arms_torch
import common


class TorchArm:
    """A plain-PyTorch baseline (``torch-per-tree`` / ``torch-batched``)."""

    family = "torch"

    def __init__(self, name: str, impl: str, workload: Dict[str, Any]):
        self.name = name
        self.impl = impl
        self.workload = workload
        self.extra: Dict[str, Any] = {}
        start = time.perf_counter()
        self.evaluator = arms_torch.make_torch_arm(impl, workload["forest"])
        torch.cuda.synchronize()
        self.setup_ms = (time.perf_counter() - start) * 1e3
        self.evaluator.setup_ms = self.setup_ms

    # -- workloads ----------------------------------------------------------
    def sr_fitness(self) -> torch.Tensor:
        return self.evaluator.sr_fitness(self.workload["x"], self.workload["labels"])

    def batch_forward(self) -> torch.Tensor:
        return self.evaluator.batch_forward(self.workload["x"])

    def forward(self) -> torch.Tensor:
        return self.evaluator.forward(self.workload["x_per_row"])

    # -- GP generation step -------------------------------------------------
    def fitness(self, forest, report: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        start = time.perf_counter()
        evaluator = arms_torch.make_torch_arm(self.impl, forest)
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - start) * 1e3
        loss = evaluator.sr_fitness(self.workload["x"], self.workload["labels"])
        if report is not None:
            report["setup_ms"] = setup_ms
        return loss


class EvoxcArm:
    """The library's own engine route (``Forest.SR_fitness`` / ``forward``)."""

    family = "evoxc"

    def __init__(self, name: str, workload: Dict[str, Any]):
        self.name = name
        self.workload = workload
        self.extra: Dict[str, Any] = {}
        self.setup_ms = 0.0

    def sr_fitness(self) -> torch.Tensor:
        return self.workload["forest"].SR_fitness(self.workload["x"], self.workload["labels"])

    def batch_forward(self) -> torch.Tensor:
        return self.workload["forest"].batch_forward(self.workload["x"])

    def forward(self) -> torch.Tensor:
        return self.workload["forest"].forward(self.workload["x_per_row"])

    def fitness(self, forest, report: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        return forest.SR_fitness(self.workload["x"], self.workload["labels"])


def apply_arm_env(arm: str) -> Dict[str, Optional[str]]:
    """Set the arm's engine knobs in ``os.environ`` (one arm per process)."""
    if arm in common.TORCH_ARMS:
        # the torch arms must not be helped by an engine route: pin the engine to
        # its default so an inherited environment cannot change anything
        os.environ.pop("EVOGP_EVAL_MODE", None)
        os.environ.pop("EVOGP_EVOXC_BACKEND", None)
        return {"eval_mode": None, "backend": None}
    spec = common.EVOXC_ARMS[arm]
    for key, env in (("eval_mode", "EVOGP_EVAL_MODE"), ("backend", "EVOGP_EVOXC_BACKEND")):
        value = spec[key]
        if value is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = value
    return {"eval_mode": spec["eval_mode"], "backend": spec["backend"]}


def make_arm(arm: str, workload: Dict[str, Any]):
    if arm in common.TORCH_ARMS:
        return TorchArm(arm, common.TORCH_ARMS[arm]["impl"], workload)
    if arm in common.EVOXC_ARMS:
        return EvoxcArm(arm, workload)
    raise ValueError(f"unknown arm {arm!r}; known: {common.ALL_ARMS}")


# ---------------------------------------------------------------------------
# one full GP generation step
# ---------------------------------------------------------------------------
class SRLossProblem:
    """The minimal problem a pipeline step needs: maximized = ``-positive loss``."""

    def __init__(self, fitness):
        self._fitness = fitness

    def evaluate(self, forest):
        return -self._fitness(forest, {})


def make_pipeline(workload: Dict[str, Any], fitness):
    """``GeneticProgramming`` + ``StandardPipeline`` over the cell's descriptor."""
    from evogp.algorithm import (
        DefaultCrossover,
        DefaultMutation,
        DefaultSelection,
        GeneticProgramming,
    )
    from evogp.pipeline import StandardPipeline

    descriptor = workload["descriptor"]
    algorithm = GeneticProgramming(
        initial_forest=workload["forest"],
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(
            mutation_rate=0.2, descriptor=descriptor.update(max_layer_cnt=3)
        ),
        selection=DefaultSelection(survival_rate=0.3, elite_rate=0.01),
    )
    pipeline = StandardPipeline(
        algorithm=algorithm,
        problem=SRLossProblem(fitness),
        is_show_details=False,
    )
    return pipeline
