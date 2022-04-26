# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
import warnings
from functools import partial

import expecttest
import numpy as np

import torch

from _utils._common_utils_for_test import reset_after_n_next_calls
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    FSSpecFileLister,
    FSSpecFileOpener,
    FSSpecSaver,
    IterableWrapper,
    TFRecordLoader,
)


class TestDataPipeTFRecord(expecttest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fakedata", "tfrecord")

    def assertArrayEqual(self, arr1, arr2):
        np.testing.assert_array_equal(arr1, arr2)

    def _ground_truth_data(self):
        for i in range(4):
            x = torch.range(i * 10, (i + 1) * 10 - 1)
            yield {
                "x_float": x,
                "x_int": (x * 10).long(),
                "x_byte": [b"test str"],
            }

    def _ground_truth_seq_data(self):
        for i in range(4):
            x = torch.range(i * 10, (i + 1) * 10 - 1)
            rep = 2 * i + 3
            yield {"x_float": x, "x_int": (x * 10).long(), "x_byte": [b"test str"]}, {
                "x_float_seq": [x] * rep,
                "x_int_seq": [(x * 10).long()] * rep,
                "x_byte_seq": [[b"test str"]] * rep,
            }

    @torch.no_grad()
    def test_tfrecord_loader_example_iterdatapipe(self):
        filename = f"{self.temp_dir}/example.tfrecord"
        datapipe1 = IterableWrapper([filename])
        datapipe2 = FileOpener(datapipe1, mode="b")

        # Functional Test: test if the returned data is correct
        tfrecord_parser = datapipe2.load_from_tfrecord()
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        expected_res = final_expected_res = list(self._ground_truth_data())
        for true_data, loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data[key].numpy(), loaded_data[key].numpy())
            self.assertEqual(len(loaded_data["x_byte"]), 1)
            self.assertEqual(true_data["x_byte"][0], loaded_data["x_byte"][0])

        # Functional Test: test if the shape of the returned data is correct when using spec
        tfrecord_parser = datapipe2.load_from_tfrecord(
            {
                "x_float": ((5, 2), torch.float64),
                "x_int": ((5, 2), torch.int32),
                "x_byte": (tuple(), None),
            }
        )
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        expected_res = [
            {
                "x_float": x["x_float"].reshape(5, 2),
                "x_int": x["x_int"].reshape(5, 2),
                "x_byte": x["x_byte"][0],
            }
            for x in self._ground_truth_data()
        ]
        for true_data, loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            self.assertArrayEqual(true_data["x_float"].numpy(), loaded_data["x_float"].float().numpy())
            self.assertArrayEqual(true_data["x_int"].numpy(), loaded_data["x_int"].long().numpy())
            self.assertEqual(loaded_data["x_float"].dtype, torch.float64)
            self.assertEqual(loaded_data["x_int"].dtype, torch.int32)
            self.assertEqual(true_data["x_byte"], loaded_data["x_byte"])

        # Functional Test: ignore features missing from spec
        tfrecord_parser = datapipe2.load_from_tfrecord(
            {
                "x_float": ((10,), torch.float32),
            }
        )
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        expected_res = [
            {
                "x_float": x["x_float"],
            }
            for x in self._ground_truth_data()
        ]
        for true_data, loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            self.assertArrayEqual(true_data["x_float"].numpy(), loaded_data["x_float"].float().numpy())

        # Functional Test: raises error if missing spec feature
        with self.assertRaises(RuntimeError):
            tfrecord_parser = datapipe2.load_from_tfrecord(
                {
                    "x_float_unknown": ((5, 2), torch.float64),
                    "x_int": ((5, 2), torch.int32),
                    "x_byte": (tuple(), None),
                }
            )
            result = list(tfrecord_parser)

        # Reset Test:
        tfrecord_parser = TFRecordLoader(datapipe2)
        expected_res = final_expected_res
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(tfrecord_parser, n_elements_before_reset)
        self.assertEqual(len(expected_res[:n_elements_before_reset]), len(res_before_reset))
        for true_data, loaded_data in zip(expected_res[:n_elements_before_reset], res_before_reset):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data[key].numpy(), loaded_data[key].numpy())
            self.assertEqual(true_data["x_byte"][0], loaded_data["x_byte"][0])
        self.assertEqual(len(expected_res), len(res_after_reset))
        for true_data, loaded_data in zip(expected_res, res_after_reset):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data[key].numpy(), loaded_data[key].numpy())
            self.assertEqual(true_data["x_byte"][0], loaded_data["x_byte"][0])

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "doesn't have valid length"):
            len(tfrecord_parser)

    @torch.no_grad()
    def test_tfrecord_loader_sequence_example_iterdatapipe(self):
        filename = f"{self.temp_dir}/sequence_example.tfrecord"
        datapipe1 = IterableWrapper([filename])
        datapipe2 = FileOpener(datapipe1, mode="b")

        # Functional Test: test if the returned data is correct
        tfrecord_parser = datapipe2.load_from_tfrecord()
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        expected_res = final_expected_res = list(self._ground_truth_seq_data())
        for (true_data_ctx, true_data_seq), loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data_ctx.keys()).union(true_data_seq.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data_ctx[key].numpy(), loaded_data[key].numpy())
                self.assertEqual(len(true_data_seq[key + "_seq"]), len(loaded_data[key + "_seq"]))
                self.assertIsInstance(loaded_data[key + "_seq"], list)
                for a1, a2 in zip(true_data_seq[key + "_seq"], loaded_data[key + "_seq"]):
                    self.assertArrayEqual(a1, a2)
            self.assertEqual(true_data_ctx["x_byte"], loaded_data["x_byte"])
            self.assertListEqual(true_data_seq["x_byte_seq"], loaded_data["x_byte_seq"])

        # Functional Test: test if the shape of the returned data is correct when using spec
        tfrecord_parser = datapipe2.load_from_tfrecord(
            {
                "x_float": ((5, 2), torch.float64),
                "x_int": ((5, 2), torch.int32),
                "x_byte": (tuple(), None),
                "x_float_seq": ((-1, 5, 2), torch.float64),
                "x_int_seq": ((-1, 5, 2), torch.int32),
                "x_byte_seq": ((-1,), None),
            }
        )
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)

        expected_res = [
            (
                {
                    "x_float": x["x_float"].reshape(5, 2),
                    "x_int": x["x_int"].reshape(5, 2),
                    "x_byte": x["x_byte"][0],
                },
                {
                    "x_float_seq": [y.reshape(5, 2).numpy() for y in z["x_float_seq"]],
                    "x_int_seq": [y.reshape(5, 2).numpy() for y in z["x_int_seq"]],
                    "x_byte_seq": [y[0] for y in z["x_byte_seq"]],
                },
            )
            for x, z in self._ground_truth_seq_data()
        ]
        for (true_data_ctx, true_data_seq), loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data_ctx.keys()).union(true_data_seq.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                l_loaded_data = loaded_data[key]
                if key == "x_float":
                    l_loaded_data = l_loaded_data.float()
                else:
                    l_loaded_data = l_loaded_data.int()
                self.assertArrayEqual(true_data_ctx[key].numpy(), l_loaded_data.numpy())
                self.assertArrayEqual(true_data_seq[key + "_seq"], loaded_data[key + "_seq"])
            self.assertEqual(true_data_ctx["x_byte"], loaded_data["x_byte"])
            self.assertListEqual(true_data_seq["x_byte_seq"], loaded_data["x_byte_seq"])

        # Functional Test: ignore features missing from spec
        tfrecord_parser = datapipe2.load_from_tfrecord(
            {
                "x_float": ((10,), torch.float32),
            }
        )
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        expected_res = [
            {
                "x_float": x["x_float"],
            }
            for x, z in self._ground_truth_seq_data()
        ]
        for true_data, loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            self.assertArrayEqual(true_data["x_float"].numpy(), loaded_data["x_float"].float().numpy())

        # Functional Test: raises error if missing spec feature
        with self.assertRaises(RuntimeError):
            tfrecord_parser = datapipe2.load_from_tfrecord(
                {"x_float_unknown": ((5, 2), torch.float64), "x_int": ((5, 2), torch.int32), "x_byte": None}
            )
            result = list(tfrecord_parser)

        # Reset Test:
        tfrecord_parser = TFRecordLoader(datapipe2)
        expected_res = final_expected_res
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(tfrecord_parser, n_elements_before_reset)
        self.assertEqual(len(expected_res[:n_elements_before_reset]), len(res_before_reset))
        for (true_data_ctx, true_data_seq), loaded_data in zip(
            expected_res[:n_elements_before_reset], res_before_reset
        ):
            self.assertSetEqual(set(true_data_ctx.keys()).union(true_data_seq.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data_ctx[key].numpy(), loaded_data[key].numpy())
                self.assertEqual(len(true_data_seq[key + "_seq"]), len(loaded_data[key + "_seq"]))
                self.assertIsInstance(loaded_data[key + "_seq"], list)
                for a1, a2 in zip(true_data_seq[key + "_seq"], loaded_data[key + "_seq"]):
                    self.assertArrayEqual(a1, a2)
            self.assertEqual(true_data_ctx["x_byte"], loaded_data["x_byte"])
            self.assertListEqual(true_data_seq["x_byte_seq"], loaded_data["x_byte_seq"])
        self.assertEqual(len(expected_res), len(res_after_reset))
        for (true_data_ctx, true_data_seq), loaded_data in zip(expected_res, res_after_reset):
            self.assertSetEqual(set(true_data_ctx.keys()).union(true_data_seq.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data_ctx[key].numpy(), loaded_data[key].numpy())
                self.assertEqual(len(true_data_seq[key + "_seq"]), len(loaded_data[key + "_seq"]))
                self.assertIsInstance(loaded_data[key + "_seq"], list)
                for a1, a2 in zip(true_data_seq[key + "_seq"], loaded_data[key + "_seq"]):
                    self.assertArrayEqual(a1, a2)
            self.assertEqual(true_data_ctx["x_byte"], loaded_data["x_byte"])
            self.assertListEqual(true_data_seq["x_byte_seq"], loaded_data["x_byte_seq"])

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "doesn't have valid length"):
            len(tfrecord_parser)


if __name__ == "__main__":
    unittest.main()
