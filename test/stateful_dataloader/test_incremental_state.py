# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict

import torch

from torch.testing._internal.common_utils import TestCase
from torchdata.stateful_dataloader import (
    _DATASET_ITER_STATE,
    _DATASET_STATE,
    _FETCHER_ENDED,
    _FETCHER_STATE,
    _WORKER_ID,
    DeletionTombStone,
    Flattener,
    IncrementalState,
    IncrementalWorkerState,
)


class TestFlattener(TestCase):
    def test(self):
        test_dict_pairs = [
            (None, 0),
            ({}, 0),
            ({"kt": torch.rand(2, 2)}, 1),
            ({"k1": "v1"}, 1),
            ({"k2": ["v1", "v2"]}, 1),
            ({"k1": {"k2": "v2"}}, 1),
            ({"k1": {"k2": "v2"}, "k2": {"k2": "v2"}}, 2),
        ]

        for kv, flat_key_count in test_dict_pairs:
            flat_dict = Flattener.flatten(kv)
            nest_dict = Flattener.unflatten(flat_dict)
            self.assertEqual(kv, nest_dict)
            if kv is None:
                continue
            self.assertEqual(len(flat_dict), flat_key_count)
            for _, val in flat_dict.items():
                self.assertFalse(isinstance(val, Dict))


class TestIncrementalState(TestCase):
    def test_basic(self):
        incr_state = IncrementalState({"a": 4})
        delta = incr_state.generate_delta({"a": 4, "b": 3})
        self.assertEqual(delta, {"b": 3})
        incr_state.apply_delta({"a": 5})
        self.assertEqual(incr_state.get_state(), {"a": 5, "b": 3})

    def test_removal(self):
        incr_state = IncrementalState({"a": 4})
        delta = incr_state.generate_delta({"b": {"x": "y"}})
        self.assertEqual(len(delta), 2)
        self.assertEqual(delta["b/x"], "y")
        self.assertTrue(isinstance(delta["a"], DeletionTombStone))
        incr_state.apply_delta({"c": 5})
        self.assertEqual(incr_state.get_state(), {"b": {"x": "y"}, "c": 5})

    def test_none(self):
        incr_state = IncrementalState(None)
        self.assertEqual(incr_state.get_state(), None)
        delta = incr_state.generate_delta({"a": 1})
        self.assertEqual(delta, {"a": 1})
        delta = incr_state.generate_delta({})
        self.assertEqual(len(delta), 1)
        self.assertTrue(delta["a"], DeletionTombStone)
        self.assertEqual(incr_state.get_state(), {})


class TestIncrementalWorkerState(TestCase):
    def test_basic(self):
        worker_state = IncrementalWorkerState(None)
        state = {
            _WORKER_ID: 0,
            _DATASET_STATE: {"abc": "xyz"},
            _FETCHER_STATE: {
                _DATASET_ITER_STATE: {"def": 123},
                _FETCHER_ENDED: False,
            },
        }
        delta = worker_state.generate_delta(state)
        # No nested dicts, so state should be same as delta
        self.assertEqual(state, delta)
        state[_DATASET_STATE]["abc"] = "tuv"
        delta = worker_state.generate_delta(state)
        self.assertEqual(delta[_DATASET_STATE], {"abc": "tuv"})
        self.assertEqual(delta[_FETCHER_STATE][_DATASET_ITER_STATE], {})
        final_state = worker_state.get_state()
        self.assertEqual(state, final_state)


if __name__ == "__main__":
    unittest.main()
