# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_utils import TestCase
from torchdata.nodes.adapters import IterableWrapper
from torchdata.nodes.base_node import BaseNodeIterator
from torchdata.nodes.snapshot_store import DequeSnapshotStore

from .utils import run_test_save_load_state


class TestDequeSnapshotStore(TestCase):
    def test_snapshot_store(self) -> None:
        store = DequeSnapshotStore()
        store.append({"a": 1}, 0)
        store.append({"a": 2}, 10)

        self.assertEqual(len(store._deque), 2)

        val = store.pop_version(0)
        self.assertEqual(val, {"a": 1})
        self.assertEqual(len(store._deque), 1)
        val = store.pop_version(1)
        self.assertIsNone(val)
        self.assertEqual(len(store._deque), 1)
        val = store.pop_version(7)
        self.assertIsNone(val)
        self.assertEqual(len(store._deque), 1)
        val = store.pop_version(10)
        self.assertEqual(val, {"a": 2})
        self.assertEqual(len(store._deque), 0)

        val = store.pop_version(11)
        self.assertIsNone(val)
        self.assertEqual(len(store._deque), 0)

        with self.assertRaisesRegex(ValueError, "is not strictly greater than"):
            store.append({"a": 3}, 3)

        self.assertEqual(len(store._deque), 0)

        with self.assertRaisesRegex(ValueError, "is not strictly greater than"):
            store.append({"a": 4}, 10)
        self.assertEqual(len(store._deque), 0)

        store.append({"a": 4}, 11)
        store.append({"a": 5}, 19)
        val = store.pop_version(19)
        self.assertEqual(val, {"a": 5})
        self.assertEqual(len(store._deque), 0)
