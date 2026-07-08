"""Reproducibility utilities: global seeding for Python, NumPy, and PyTorch.

Ensures that experiments are reproducible across CPU and GPU (including
multi-GPU) execution, and provides a ``DataLoader``-compatible
``worker_init_fn`` so that dataloader worker processes are also seeded
deterministically.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

_logger = logging.getLogger(__name__)

_MAX_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class SeedReport:
    """Summary of the seeding operation performed by :func:`set_global_seed`.

    Attributes:
        seed: The base seed that was applied.
        deterministic: Whether CuDNN deterministic mode was requested.
        cudnn_deterministic: Actual resulting value of
            ``torch.backends.cudnn.deterministic``.
        cudnn_benchmark: Actual resulting value of
            ``torch.backends.cudnn.benchmark``.
        cuda_seeded: Whether CUDA RNGs were seeded (False if CUDA unavailable).
    """

    seed: int
    deterministic: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_seeded: bool


def set_global_seed(seed: int, *, deterministic: bool = True) -> SeedReport:
    """Seed all relevant random number generators for reproducibility.

    This seeds, in order: the ``PYTHONHASHSEED`` environment variable
    (informational only — it does not affect the already-running process),
    Python's ``random`` module, NumPy's global RNG, and PyTorch's CPU and
    (if available) CUDA RNGs across all visible devices.

    Args:
        seed: Non-negative integer seed value.
        deterministic: If True, configure CuDNN for deterministic
            algorithms (``cudnn.deterministic = True``,
            ``cudnn.benchmark = False``). This can reduce throughput but
            improves run-to-run reproducibility. If False, CuDNN is left in
            its default, higher-throughput (non-deterministic) mode.

    Returns:
        A :class:`SeedReport` describing exactly what was configured.

    Raises:
        ValueError: If ``seed`` is negative or exceeds the valid 32-bit
            unsigned range.
    """
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - depends on torch build
            _logger.debug("torch.use_deterministic_algorithms unavailable: %s", exc)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    report = SeedReport(
        seed=seed,
        deterministic=deterministic,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_seeded=cuda_seeded,
    )
    _logger.info(
        "Global seed set to %d (deterministic=%s, cuda_seeded=%s).",
        seed,
        deterministic,
        cuda_seeded,
    )
    return report


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` that deterministically seeds workers.

    Derives a distinct, reproducible seed per worker from PyTorch's initial
    seed so that different dataloader workers do not draw identical random
    sequences (e.g. for augmentation).

    Args:
        worker_id: The dataloader worker index, provided automatically by
            ``torch.utils.data.DataLoader`` when this function is passed as
            ``worker_init_fn``.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, num_workers=4, worker_init_fn=seed_worker)  # doctest: +SKIP
    """
    worker_seed = (torch.initial_seed() + worker_id) % _MAX_SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    _logger.debug("Dataloader worker %d seeded with %d.", worker_id, worker_seed)
