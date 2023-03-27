# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch

from torchdata.dataloader2.random._philox import PhiloxEngine


_UINT64_UPPER_BOUND = 2 ** 64


def _get_torch_random_seed():
    iinfo = torch.iinfo(torch.int64)
    seed = torch.randint(iinfo.min, iinfo.max, ()).item()
    # Convert int64 to uint64
    seed += 2 ** 63
    return seed


class SeedGenerator:
    r"""
    ``SeedGenerator`` is used to generate seeds in a deterministic and randomized manner
    based on a user-provided initial seed. Internally, it utilizes a counter-based PRNG
    called Philox to generate random seeds.

    Args:
        seed: The base seed to generate random seeds
    """
    _shared_rng: PhiloxEngine
    _worker_rng: PhiloxEngine

    def __init__(self, seed: Optional[int] = None, _rngs: Optional[Tuple[PhiloxEngine, PhiloxEngine]] = None) -> None:
        if seed is not None and _rngs is not None:
            raise ValueError("SeedGenerator doesn't allow both seed and _rng specified at the same time")
        if _rngs is None:
            self._shared_rng = PhiloxEngine()
            self._worker_rng = PhiloxEngine()
            self.seed(seed)
        else:
            assert len(_rngs) == 2
            self._shared_rng, self._worker_rng = _rngs

    def seed(self, seed: Optional[int] = None) -> None:
        r"""
        Re-seed the ``SeedGenerator``. When ``None`` is provided, a random seed generated
        by the default PyTorch RNG.
        """
        if seed is None:
            seed = _get_torch_random_seed()
        if seed >= _UINT64_UPPER_BOUND:
            raise ValueError(f"Expected an uint64 seed, but got {seed}.")
        self._shared_rng.seed(seed)
        self._worker_rng.seed(seed)

    def generate_shared_seed(self) -> int:
        r"""
        Generate one uint64 random seed that is supposed to be the same across
        distributed processes.
        """
        return self._shared_rng.generate()

    def generate_seed(self) -> int:
        r"""
        Generate one unique uint64 random seed based on distributed and multiprocessing
        information.
        """
        return self._worker_rng.generate()

    def spawn(self, worker_id: int, inplace: bool = False) -> "SeedGenerator":
        r"""
        Spawn a sub-SeedGenerator based on the provided worker_id. If inplace is turn on, the SeedGenerator
        will evolve itself rather than spawning a new
        """
        if worker_id < 0:
            raise ValueError(f"Expected `rank` equal or larger than 0, but got {worker_id}.")

        if inplace:
            self._worker_rng = self._worker_rng.spawn(worker_id)
            return self
        return SeedGenerator(seed=None, _rngs=(self._shared_rng.clone(), self._worker_rng.spawn(worker_id)))

    def __getstate__(self):
        state = (
            self._shared_rng,
            self._worker_rng,
        )
        return state

    def __setstate__(self, state):
        self._shared_rng, self._worker_rng = state
