# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

# Note [Philox Engine implementation]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Refer to: http://www.thesalmons.org/john/random123/papers/random123sc11.pdf for details regarding the engine.
# Using Philox4×32-10 for the sake of performance, randomness and crush-resistance.
# The following code could be optimized into C++ bindings

# Philox Constants
kPhilox10A = 0x9E3779B9
kPhilox10B = 0xBB67AE85
kPhiloxSA = 0xD2511F53
kPhiloxSB = 0xCD9E8D57

MASK_32b = 0xFFFFFFFF
MASK_64b = 0xFFFFFFFFFFFFFFFF
HALF_UINT64 = 0x8000000000000000


def mulhilo32(a: int, b: int) -> Tuple[int, int]:
    product = a * b
    return product & MASK_32b, (product >> 32) & MASK_32b


def single_round(key: List[int], ctr: List[int]) -> List[int]:
    lo0, hi0 = mulhilo32(kPhiloxSA, ctr[0])
    lo1, hi1 = mulhilo32(kPhiloxSB, ctr[2])
    res = [0] * 4
    res[0] = hi1 ^ ctr[1] ^ key[0]
    res[1] = lo1
    res[2] = hi0 ^ ctr[3] ^ key[1]
    res[3] = lo0
    return res


def philox_10_round(key: Tuple[int, int], ctr: List[int]) -> List[int]:
    _key = list(key)
    _ctr = list(ctr)
    for _ in range(9):
        _ctr = single_round(_key, _ctr)
        _key[0] = (_key[0] + kPhilox10A) & MASK_32b
        _key[1] = (_key[1] + kPhilox10B) & MASK_32b
    return single_round(_key, _ctr)


class PhiloxEngine:
    r"""
    Philox is a counter-based RNG with a certain properties:
        - High performance
        - Statistiacl random
        - Crush-resistance Bijection

    Generate new seeds or spawn parallel seeds for worker processes.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed: Tuple[int, int] = (-1, -1)
        self._ctr: List[int] = [0] * 4
        self._generated_seeds: Optional[List[int]] = None
        self._spawn_seed: Tuple[int, int] = (-1, -1)
        if seed is not None:
            self.seed(seed)

    def _incr_ctr(self) -> None:
        for i in range(3):
            self._ctr[i] += 1
            if self._ctr[i] <= MASK_32b:
                return
            self._ctr[i] = 0
        self._ctr[3] += 1
        # if overflow (2^128) has occurred during addition, back to the initial counter
        if self._ctr[3] > MASK_32b:
            self._ctr[3] = 0
            self._incr_ctr()

    def seed(self, seed: int) -> "PhiloxEngine":
        seed = seed & MASK_64b
        # Convert seed from int64 to uint64
        if seed < 0:
            seed = seed + HALF_UINT64
        lo = seed & MASK_32b
        hi = (seed >> 32) & MASK_32b
        self._seed = (lo, hi)
        # Reset counter and cached seed
        self._ctr = [0] * 4
        self._generated_seeds = None
        # Generate the spawn seed
        self._spawn_seed = tuple(philox_10_round(self._seed, self._ctr)[:2])  # type: ignore[assignment]
        self._incr_ctr()
        return self

    def generate(self) -> int:
        assert self._seed != (-1, -1), "Please provide seed to PhiloxEngine"

        if self._generated_seeds is None:
            self._generated_seeds = philox_10_round(self._seed, self._ctr)
            self._incr_ctr()
            res = self._generated_seeds[:2]
        else:
            res = self._generated_seeds[2:]
            self._generated_seeds = None
        return (res[1] << 32) + res[0]

    def clone(self) -> "PhiloxEngine":
        new_engine = PhiloxEngine(None)
        new_engine._seed = self._seed  # immutable tuple
        new_engine._ctr = self._ctr.copy()
        new_engine._generated_seeds = None if self._generated_seeds is None else self._generated_seeds.copy()
        new_engine._spawn_seed = self._spawn_seed  # immutable tuple
        return new_engine

    def spawn(self, index: int) -> "PhiloxEngine":
        assert index >= 0, f"Expected a non-negative value for spawn, but found {index}"
        assert self._spawn_seed != (-1, -1), "Please provide seed to PhiloxEngine"

        offset = index % 2
        val = index if offset == 0 else index - 1

        ctr = []
        for _ in range(4):
            ctr.append(val & MASK_32b)
            val = val >> 32

        res = philox_10_round(self._spawn_seed, ctr)[offset * 2 : offset * 2 + 2]
        sub_seed = (res[1] << 32) + res[0]
        return PhiloxEngine(sub_seed)
