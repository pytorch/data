# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict

import torch

from torch.testing._internal.common_utils import TestCase
from torchdata.stateful_dataloader.incremental_state import (
    _DATASET_ITER_STATE,
    _DATASET_STATE,
    _FETCHER_ENDED,
    _FETCHER_STATE,
    _flatten,
    _IncrementalState,
    _IncrementalWorkerState,
    _Tombstone,
    _unflatten,
    _WORKER_ID,
)


class TestFlattener(TestCase):
    def test(self):
        test_dict_pairs = [
            (None, 1),
            ({}, 1),
            ({"kt": torch.rand(2, 2)}, 1),
            ({"k1": "v1"}, 1),
            ({"k2": ["v1", "v2"]}, 1),
            ({"k1": {"k2": "v2"}}, 1),
            ({"k1": {"k2": "v2"}, "k2": {"k2": "v2"}}, 2),
            ({"k1": None}, 1),
            ({"k1": {"k2": None}}, 1),
            ("abcd", 1),
            ({"k1": {"k2": {}, "k3": None}}, 2),
        ]

        for kv, flat_key_count in test_dict_pairs:
            flat_dict = _flatten(kv)
            nest_dict = _unflatten(flat_dict)
            self.assertEqual(kv, nest_dict)
            if kv is None:
                continue
            self.assertTrue(isinstance(flat_dict, dict))
            self.assertTrue(len(flat_dict) > 0)
            self.assertEqual(len(flat_dict), flat_key_count)
            for _, val in flat_dict.items():
                # Only empty dicts are allowed in flattened dict
                if isinstance(val, Dict):
                    self.assertFalse(len(val))


class TestIncrementalState(TestCase):
    def test_basic(self):
        incr_state = _IncrementalState({"a": 4})
        delta = incr_state.generate_delta({"a": 4, "b": 3})
        self.assertEqual(delta, {("b",): 3})
        incr_state.apply_delta({("a",): 5})
        self.assertEqual(incr_state.get_state(), {"a": 5, "b": 3})

    def test_non_dict(self):
        incr_state = _IncrementalState("5")
        self.assertEqual(incr_state.get_state(), "5")
        delta = incr_state.generate_delta("4")
        self.assertEqual(delta, {(): "4"})
        self.assertEqual(incr_state.get_state(), "4")

    def test_tensor_state(self):
        incr_state = _IncrementalState(torch.rand(5))
        ns = torch.rand(5)
        delta = incr_state.generate_delta(ns)
        self.assertEqual(len(delta), 1)
        self.assertTrue(torch.equal(delta[()], ns))
        self.assertTrue(torch.equal(incr_state.get_state(), ns))

    def test_removal(self):
        incr_state = _IncrementalState({"a": 4})
        recv_state = _IncrementalState({"a": 4})
        delta = incr_state.generate_delta({"b": {"x": "y"}})
        self.assertEqual(len(delta), 2)
        self.assertEqual(delta[("b", "x")], "y")
        self.assertTrue(isinstance(delta[("a",)], _Tombstone))
        recv_state.apply_delta(delta)
        self.assertEqual(recv_state.get_state(), incr_state.get_state())
        delta = {("c",): 5}
        incr_state.apply_delta(delta)
        recv_state.apply_delta(delta)
        self.assertEqual(incr_state.get_state(), {"b": {"x": "y"}, "c": 5})
        self.assertEqual(incr_state.get_state(), recv_state.get_state())

    def test_none(self):
        incr_state = _IncrementalState(None)
        recv_state = _IncrementalState(None)
        self.assertEqual(incr_state.get_state(), None)

        delta = incr_state.generate_delta({"a": 1})
        self.assertEqual(len(delta), 2)
        self.assertEqual(delta[("a",)], 1)
        self.assertTrue(isinstance(delta[()], _Tombstone))
        recv_state.apply_delta(delta)
        self.assertEqual(recv_state.get_state(), {"a": 1})
        self.assertEqual(incr_state.get_state(), {"a": 1})

        delta = incr_state.generate_delta({})
        recv_state.apply_delta(delta)
        self.assertEqual(len(delta), 2)
        self.assertTrue(isinstance(delta[("a",)], _Tombstone))
        self.assertEqual(delta[()], {})
        self.assertEqual(recv_state.get_state(), {})
        self.assertEqual(incr_state.get_state(), {})


class TestIncrementalWorkerState(TestCase):
    def test_basic(self):
        worker_state = _IncrementalWorkerState(None)
        state = {
            _WORKER_ID: 0,
            _DATASET_STATE: {"abc": "xyz"},
            _FETCHER_STATE: {
                _DATASET_ITER_STATE: {"def": 123},
                _FETCHER_ENDED: False,
            },
        }
        delta = worker_state.generate_delta(state)
        state[_DATASET_STATE]["abc"] = "tuv"
        delta = worker_state.generate_delta(state)
        self.assertEqual(delta[_DATASET_STATE], {("abc",): "tuv"})
        self.assertEqual(delta[_FETCHER_STATE][_DATASET_ITER_STATE], {})
        final_state = worker_state.get_state()
        self.assertEqual(state, final_state)

    def test_tensor_state(self):
        worker_state = _IncrementalWorkerState(None)
        state = {
            _WORKER_ID: 0,
            _DATASET_STATE: None,
            _FETCHER_STATE: {
                _DATASET_ITER_STATE: None,
                _FETCHER_ENDED: False,
            },
        }
        delta = worker_state.generate_delta(state)
        ts = torch.rand(5)
        state[_DATASET_STATE] = ts
        delta = worker_state.generate_delta(state)
        self.assertEqual(delta[_FETCHER_STATE][_DATASET_ITER_STATE], None)
        final_state = worker_state.get_state()
        self.assertEqual(state, final_state)

    def test_nested_state(self):
        worker_state = _IncrementalWorkerState(None)
        state = {
            _WORKER_ID: 0,
            _DATASET_STATE: {"z": {"a1": "b1"}},
            _FETCHER_STATE: {
                _DATASET_ITER_STATE: {"a": {"b": "c"}},
                _FETCHER_ENDED: False,
            },
        }
        delta = worker_state.generate_delta(state)
        state[_DATASET_STATE]["z"]["a1"] = "z1"
        delta = worker_state.generate_delta(state)
        self.assertEqual(delta[_DATASET_STATE], {("z", "a1"): "z1"})
        self.assertEqual(delta[_FETCHER_STATE][_DATASET_ITER_STATE], {})
        final_state = worker_state.get_state()
        self.assertEqual(state, final_state)

    def test_none_state(self):
        worker_state = _IncrementalWorkerState(None)
        state = {
            _WORKER_ID: 0,
            _DATASET_STATE: {"z": {"a1": None}},
            _FETCHER_STATE: {
                _DATASET_ITER_STATE: None,
                _FETCHER_ENDED: False,
            },
        }
        delta = worker_state.generate_delta(state)
        state[_DATASET_STATE]["z"]["a1"] = 1
        delta = worker_state.generate_delta(state)
        self.assertEqual(delta[_DATASET_STATE], {("z", "a1"): 1})
        self.assertEqual(delta[_FETCHER_STATE][_DATASET_ITER_STATE], None)
        final_state = worker_state.get_state()
        self.assertEqual(state, final_state)

    def test_none_replacement(self):
        worker_state = _IncrementalWorkerState(None)
        state = {
            _WORKER_ID: 0,
            _DATASET_STATE: {"z": None},
            _FETCHER_STATE: {
                _DATASET_ITER_STATE: {"a": None},
                _FETCHER_ENDED: False,
            },
        }
        delta = worker_state.generate_delta(state)
        state[_DATASET_STATE]["z"] = "abc"
        state[_FETCHER_STATE][_DATASET_ITER_STATE]["a"] = {"b": None}
        delta = worker_state.generate_delta(state)
        self.assertEqual(delta[_DATASET_STATE], {("z",): "abc"})
        self.assertEqual(delta[_FETCHER_STATE][_DATASET_ITER_STATE][("a", "b")], None)
        self.assertTrue(isinstance(delta[_FETCHER_STATE][_DATASET_ITER_STATE][("a",)], _Tombstone))
        final_state = worker_state.get_state()
        self.assertEqual(state, final_state)


if __name__ == "__main__":
    unittest.main()
