# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchdata.dataloader2.random import SeedGenerator
from torchdata.dataloader2.random._philox import PhiloxEngine


class TestPhilox(unittest.TestCase):
    def test_philox_engine_generate(self):
        prng = PhiloxEngine()
        with self.assertRaisesRegex(AssertionError, "Please provide seed"):
            prng.generate()

        prng.seed(123)
        s0 = [prng.generate() for _ in range(10)]

        # Same seed
        prng = PhiloxEngine(seed=123)
        s1 = [prng.generate() for _ in range(10)]
        self.assertEqual(s0, s1)

        # Reset
        prng.seed(123)
        s2 = [prng.generate() for _ in range(10)]
        self.assertEqual(s1, s2)

        # Different seeds
        prng = PhiloxEngine(seed=321)
        s3 = [prng.generate() for _ in range(10)]
        self.assertNotEqual(s0, s3)

    def test_philox_engine_spawn(self):
        prng = PhiloxEngine()
        with self.assertRaisesRegex(AssertionError, "Expected a non-negative value"):
            prng.spawn(-1)
        with self.assertRaisesRegex(AssertionError, "Please provide seed"):
            prng.spawn(0)

        prng.seed(123)
        s0 = [prng.spawn(i)._seed for i in range(10)]

        # Same seed
        prng = PhiloxEngine(seed=123)
        s1 = [prng.spawn(i)._seed for i in range(10)]
        self.assertEqual(s0, s1)

        # Generate after spawn
        sprng1 = prng.spawn(1)
        sprng2 = prng.spawn(1)
        ss1 = [sprng1.generate() for _ in range(10)]
        ss2 = [sprng2.generate() for _ in range(10)]
        self.assertEqual(ss1, ss2)

        sprng3 = prng.spawn(2)
        ss3 = [sprng3.generate() for _ in range(10)]
        self.assertNotEqual(ss1, ss3)

        # Reset
        prng.seed(123)
        s2 = [prng.spawn(i)._seed for i in range(10)]
        self.assertEqual(s1, s2)

        # Different seeds
        prng = PhiloxEngine(seed=321)
        s3 = [prng.spawn(i)._seed for i in range(10)]
        self.assertNotEqual(s0, s3)


class TestSeedGenerator(unittest.TestCase):
    def test_seed_generator_generate(self):
        # Generate seeds
        sg = SeedGenerator(123)
        s0 = [sg.generate_seed() for _ in range(10)]

        # Reset
        sg.seed(123)
        s1 = [sg.generate_seed() for _ in range(10)]
        self.assertEqual(s0, s1)

        # Different Seeds
        sg.seed(321)
        s2 = [sg.generate_seed() for _ in range(10)]
        self.assertNotEqual(s0, s2)

    def test_seed_generator_spawn(self):
        sg = SeedGenerator(123)

        # Spawn new Seed Generators
        sg1 = sg.spawn(1)
        sg2 = sg.spawn(2)

        for _ in range(10):
            self.assertNotEqual(sg1.generate_seed(), sg2.generate_seed())
            # Generate shared seeds
            self.assertEqual(sg1.generate_shared_seed(), sg2.generate_shared_seed())

        sg1_1 = sg.spawn(1)
        sg1_2 = sg.spawn(1)
        for _ in range(10):
            self.assertEqual(sg1_1.generate_seed(), sg1_2.generate_seed())


if __name__ == "__main__":
    unittest.main()
