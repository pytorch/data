# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest
import warnings
from functools import partial

import expecttest
import numpy as np

import torch

from _utils._common_utils_for_test import create_temp_dir, reset_after_n_next_calls
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    FSSpecFileLister,
    FSSpecFileOpener,
    FSSpecSaver,
    IterableWrapper,
    TFRecordLoader,
)

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False
skipIfNoTF = unittest.skipIf(not HAS_TF, "no tensorflow")


def create_temp_tfrecord_files(temp_dir: str):
    with tf.io.TFRecordWriter(os.path.join(temp_dir, "example.tfrecord")) as writer:
        for _ in range(4):
            x = tf.random.uniform(
                [
                    10,
                ]
            )

            record_bytes = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "x_float": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                        "x_int": tf.train.Feature(int64_list=tf.train.Int64List(value=tf.cast(x * 10, "int64"))),
                        "x_byte": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test str"])),
                    }
                )
            ).SerializeToString()
            writer.write(record_bytes)

    with tf.io.TFRecordWriter(os.path.join(temp_dir, "sequence_example.tfrecord")) as writer:
        for _ in range(4):
            x = tf.random.uniform(
                [
                    10,
                ]
            )
            rep = int(
                tf.random.uniform(
                    [
                        1,
                    ]
                ).numpy()[0]
                * 10
                + 1
            )

            record_bytes = tf.train.SequenceExample(
                context=tf.train.Features(
                    feature={
                        "x_float": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                        "x_int": tf.train.Feature(int64_list=tf.train.Int64List(value=tf.cast(x * 10, "int64"))),
                        "x_byte": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test str"])),
                    }
                ),
                feature_lists=tf.train.FeatureLists(
                    feature_list={
                        "x_float_seq": tf.train.FeatureList(
                            feature=[tf.train.Feature(float_list=tf.train.FloatList(value=x))] * rep
                        ),
                        "x_int_seq": tf.train.FeatureList(
                            feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=tf.cast(x * 10, "int64")))]
                            * rep
                        ),
                        "x_byte_seq": tf.train.FeatureList(
                            feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test str"]))] * rep
                        ),
                    }
                ),
            ).SerializeToString()
            writer.write(record_bytes)


class TestDataPipeTFRecord(expecttest.TestCase):
    def setUp(self):
        self.temp_dir = create_temp_dir()
        self.temp_files = create_temp_tfrecord_files(self.temp_dir.name)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f"TestDataPipeTFRecord was not able to cleanup temp dir due to {e}")

    def assertArrayEqual(self, arr1, arr2):
        np.testing.assert_array_equal(arr1, arr2)

    @skipIfNoTF
    @torch.no_grad()
    def test_tfrecord_loader_example_iterdatapipe(self):
        filename = f"{self.temp_dir.name}/example.tfrecord"
        datapipe1 = IterableWrapper([filename])
        datapipe2 = FileOpener(datapipe1, mode="b")

        # Functional Test: test if the returned data is correct
        tfrecord_parser = datapipe2.load_from_tfrecord()
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        decode_fn = partial(
            tf.io.parse_single_example,
            features={
                "x_float": tf.io.FixedLenFeature([10], tf.float32),
                "x_int": tf.io.FixedLenFeature([10], tf.int64),
                "x_byte": tf.io.FixedLenFeature([], tf.string),
            },
        )
        expected_res = final_expected_res = list(tf.data.TFRecordDataset([filename]).map(decode_fn))
        for true_data, loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data[key].numpy(), loaded_data[key].numpy())
            self.assertEqual(len(loaded_data["x_byte"]), 1)
            self.assertEqual(true_data["x_byte"].numpy(), loaded_data["x_byte"][0])

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
        decode_fn = partial(
            tf.io.parse_single_example,
            features={
                "x_float": tf.io.FixedLenFeature([5, 2], tf.float32),
                "x_int": tf.io.FixedLenFeature([5, 2], tf.int64),
                "x_byte": tf.io.FixedLenFeature([], tf.string),
            },
        )
        expected_res = list(tf.data.TFRecordDataset([filename]).map(decode_fn))
        for true_data, loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            self.assertArrayEqual(true_data["x_float"].numpy(), loaded_data["x_float"].float().numpy())
            self.assertArrayEqual(true_data["x_int"].numpy(), loaded_data["x_int"].long().numpy())
            self.assertEqual(loaded_data["x_float"].dtype, torch.float64)
            self.assertEqual(loaded_data["x_int"].dtype, torch.int32)
            self.assertEqual(true_data["x_byte"].numpy(), loaded_data["x_byte"])

        # Functional Test: ignore features missing from spec
        tfrecord_parser = datapipe2.load_from_tfrecord(
            {
                "x_float": ((10,), torch.float32),
            }
        )
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        decode_fn = partial(
            tf.io.parse_single_example,
            features={
                "x_float": tf.io.FixedLenFeature([10], tf.float32),
            },
        )
        expected_res = list(tf.data.TFRecordDataset([filename]).map(decode_fn))
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
            self.assertEqual(true_data["x_byte"].numpy(), loaded_data["x_byte"][0])
        self.assertEqual(len(expected_res), len(res_after_reset))
        for true_data, loaded_data in zip(expected_res, res_after_reset):
            self.assertSetEqual(set(true_data.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data[key].numpy(), loaded_data[key].numpy())
            self.assertEqual(true_data["x_byte"].numpy(), loaded_data["x_byte"][0])

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "doesn't have valid length"):
            len(tfrecord_parser)

    @skipIfNoTF
    @torch.no_grad()
    def test_tfrecord_loader_sequence_example_iterdatapipe(self):
        filename = f"{self.temp_dir.name}/sequence_example.tfrecord"
        datapipe1 = IterableWrapper([filename])
        datapipe2 = FileOpener(datapipe1, mode="b")

        # Functional Test: test if the returned data is correct
        tfrecord_parser = datapipe2.load_from_tfrecord()
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        decode_fn = partial(
            tf.io.parse_single_sequence_example,
            context_features={
                "x_float": tf.io.FixedLenFeature([10], tf.float32),
                "x_int": tf.io.FixedLenFeature([10], tf.int64),
                "x_byte": tf.io.FixedLenFeature([1], tf.string),
            },
            sequence_features={
                "x_float_seq": tf.io.RaggedFeature(tf.float32),
                "x_int_seq": tf.io.RaggedFeature(tf.int64),
                "x_byte_seq": tf.io.RaggedFeature(tf.string),
            },
        )
        expected_res = final_expected_res = list(tf.data.TFRecordDataset([filename]).map(decode_fn))
        for (true_data_ctx, true_data_seq), loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data_ctx.keys()).union(true_data_seq.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data_ctx[key].numpy(), loaded_data[key].numpy())
                self.assertEqual(true_data_seq[key + "_seq"].to_tensor().shape[0], len(loaded_data[key + "_seq"]))
                self.assertIsInstance(loaded_data[key + "_seq"], list)
                for a1, a2 in zip(true_data_seq[key + "_seq"], loaded_data[key + "_seq"]):
                    self.assertArrayEqual(a1, a2)
            self.assertEqual(true_data_ctx["x_byte"].numpy(), loaded_data["x_byte"])
            self.assertListEqual(list(true_data_seq["x_byte_seq"].to_tensor().numpy()), loaded_data["x_byte_seq"])

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
        decode_fn = partial(
            tf.io.parse_single_sequence_example,
            context_features={
                "x_float": tf.io.FixedLenFeature([5, 2], tf.float32),
                "x_int": tf.io.FixedLenFeature([5, 2], tf.int64),
                "x_byte": tf.io.FixedLenFeature([], tf.string),
            },
            sequence_features={
                "x_float_seq": tf.io.RaggedFeature(tf.float32),
                "x_int_seq": tf.io.RaggedFeature(tf.int64),
                "x_byte_seq": tf.io.RaggedFeature(tf.string),
            },
        )
        expected_res = list(tf.data.TFRecordDataset([filename]).map(decode_fn))
        for (true_data_ctx, true_data_seq), loaded_data in zip(expected_res, result):
            self.assertSetEqual(set(true_data_ctx.keys()).union(true_data_seq.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                l_loaded_data = loaded_data[key]
                if key == "x_float":
                    l_loaded_data = l_loaded_data.float()
                else:
                    l_loaded_data = l_loaded_data.int()
                self.assertArrayEqual(true_data_ctx[key].numpy(), l_loaded_data.numpy())
                self.assertArrayEqual(
                    tf.reshape(true_data_seq[key + "_seq"].to_tensor(), [-1, 5, 2]), loaded_data[key + "_seq"]
                )
            self.assertEqual(true_data_ctx["x_byte"].numpy(), loaded_data["x_byte"])
            self.assertListEqual(list(true_data_seq["x_byte_seq"].to_tensor().numpy()), loaded_data["x_byte_seq"])

        # Functional Test: ignore features missing from spec
        tfrecord_parser = datapipe2.load_from_tfrecord(
            {
                "x_float": ((10,), torch.float32),
            }
        )
        result = list(tfrecord_parser)
        self.assertEqual(len(result), 4)
        decode_fn = partial(
            tf.io.parse_single_example,
            features={
                "x_float": tf.io.FixedLenFeature([10], tf.float32),
            },
        )
        expected_res = list(tf.data.TFRecordDataset([filename]).map(decode_fn))
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
                self.assertEqual(true_data_seq[key + "_seq"].to_tensor().shape[0], len(loaded_data[key + "_seq"]))
                self.assertIsInstance(loaded_data[key + "_seq"], list)
                for a1, a2 in zip(true_data_seq[key + "_seq"], loaded_data[key + "_seq"]):
                    self.assertArrayEqual(a1, a2)
            self.assertEqual(true_data_ctx["x_byte"].numpy(), loaded_data["x_byte"])
            self.assertListEqual(list(true_data_seq["x_byte_seq"].to_tensor().numpy()), loaded_data["x_byte_seq"])
        self.assertEqual(len(expected_res), len(res_after_reset))
        for (true_data_ctx, true_data_seq), loaded_data in zip(expected_res, res_after_reset):
            self.assertSetEqual(set(true_data_ctx.keys()).union(true_data_seq.keys()), set(loaded_data.keys()))
            for key in ["x_float", "x_int"]:
                self.assertArrayEqual(true_data_ctx[key].numpy(), loaded_data[key].numpy())
                self.assertEqual(true_data_seq[key + "_seq"].to_tensor().shape[0], len(loaded_data[key + "_seq"]))
                self.assertIsInstance(loaded_data[key + "_seq"], list)
                for a1, a2 in zip(true_data_seq[key + "_seq"], loaded_data[key + "_seq"]):
                    self.assertArrayEqual(a1, a2)
            self.assertEqual(true_data_ctx["x_byte"].numpy(), loaded_data["x_byte"])
            self.assertListEqual(list(true_data_seq["x_byte_seq"].to_tensor().numpy()), loaded_data["x_byte_seq"])

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "doesn't have valid length"):
            len(tfrecord_parser)


if __name__ == "__main__":
    unittest.main()
