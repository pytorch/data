# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import unittest
import warnings
from functools import partial
from io import StringIO
from operator import itemgetter
from typing import List

import expecttest
import torchdata.datapipes.iter as iterdp
import torchdata.datapipes.map as mapdp
from _utils._common_utils_for_test import create_temp_dir, create_temp_files
from torch.utils.data.datapipes.utils.common import DILL_AVAILABLE
from torchdata.datapipes.iter import IterableWrapper
from torchdata.datapipes.map import SequenceWrapper

if DILL_AVAILABLE:
    import dill

    dill.extend(use_dill=False)

try:
    import fsspec
except ImportError:
    fsspec = None

try:
    import iopath
except ImportError:
    iopath = None

try:
    import subprocess

    import rarfile

    try:
        rarfile.tool_setup()
        subprocess.run(("rar", "-?"), check=True)
    except (rarfile.RarCannotExec, subprocess.CalledProcessError):
        rarfile = None
except (ModuleNotFoundError, FileNotFoundError):
    rarfile = None

try:
    import torcharrow
    import torcharrow.dtypes as dt

    DTYPE = dt.Struct([dt.Field("Values", dt.int32)])
except ImportError:
    torcharrow = None
    dt = None
    DTYPE = None


def _fake_batch_fn(batch):
    return [d + 1 for d in batch]


def _fake_fn_ls(x):
    return [x, x]


def _filepath_fn(name: str, dir) -> str:
    return os.path.join(dir, os.path.basename(name))


def _filter_by_module_availability(datapipes):
    filter_set = set()
    if fsspec is None:
        filter_set.update([iterdp.FSSpecFileLister, iterdp.FSSpecFileOpener, iterdp.FSSpecSaver])
    if iopath is None:
        filter_set.update([iterdp.IoPathFileLister, iterdp.IoPathFileOpener, iterdp.IoPathSaver])
    if rarfile is None:
        filter_set.update([iterdp.RarArchiveLoader])
    if torcharrow is None or not DILL_AVAILABLE:
        filter_set.update([iterdp.DataFrameMaker, iterdp.ParquetDataFrameLoader])
    return [dp for dp in datapipes if dp[0] not in filter_set]


class TestIterDataPipeSerialization(expecttest.TestCase):
    def setUp(self):
        self.temp_dir = create_temp_dir()
        self.temp_files = create_temp_files(self.temp_dir)
        self.temp_sub_dir = create_temp_dir(self.temp_dir.name)
        self.temp_sub_files = create_temp_files(self.temp_sub_dir, 4, False)

    def tearDown(self):
        try:
            self.temp_sub_dir.cleanup()
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f"TestIterDataPipeSerialization was not able to cleanup temp dir due to {e}")

    def _serialization_test_helper(self, datapipe):
        serialized_dp = pickle.dumps(datapipe)
        deserialized_dp = pickle.loads(serialized_dp)
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            print(f"{datapipe} is failing.")
            raise e

    def _serialization_dataframe_test_helper(self, datapipe):
        serialized_dp = pickle.dumps(datapipe)
        deserialized_dp = pickle.loads(serialized_dp)
        for df1, df2 in zip(datapipe, deserialized_dp):
            for exp, act in zip(df1, df2):
                self.assertEqual(exp, act)

    def _serialization_test_for_single_dp(self, dp, is_dataframe=False):
        test_helper_fn = self._serialization_dataframe_test_helper if is_dataframe else self._serialization_test_helper
        # 1. Testing for serialization before any iteration starts
        test_helper_fn(dp)
        # 2. Testing for serialization afterDataPipe is partially read
        it = iter(dp)
        _ = next(it)
        test_helper_fn(dp)
        # 3. Testing for serialization after DataPipe is fully read
        _ = list(it)
        test_helper_fn(dp)

    def _serialization_test_for_dp_with_children(self, dp1, dp2):
        # 1. Testing for serialization before any iteration starts
        self._serialization_test_helper(dp1)
        self._serialization_test_helper(dp2)
        # 2. Testing for serialization after DataPipe is partially read
        it1, it2 = iter(dp1), iter(dp2)
        _, _ = next(it1), next(it2)
        self._serialization_test_helper(dp1)
        self._serialization_test_helper(dp2)
        # 2.5. Testing for serialization after one child DataPipe is fully read
        #      (Only for DataPipes with children DataPipes)
        _ = list(it1)  # fully read one child
        self._serialization_test_helper(dp1)
        self._serialization_test_helper(dp2)
        # 3. Testing for serialization after DataPipe is fully read
        _ = list(it2)  # fully read the other child
        self._serialization_test_helper(dp1)
        self._serialization_test_helper(dp2)

    def test_serializable(self):
        picklable_datapipes: List = [
            (iterdp.BatchMapper, IterableWrapper([(0, 0), (0, 0), (0, 0), (0, 0)]), (_fake_batch_fn, 2, 1), {}),
            (iterdp.BucketBatcher, IterableWrapper([0, 0, 0, 0, 0, 0, 0]), (5,), {}),
            (iterdp.Bz2FileLoader, None, (), {}),
            (
                iterdp.CSVDictParser,
                IterableWrapper(
                    [("f1", StringIO("Label,1,1\nLabel,2,2\nLabel,3,3")), ("f2", StringIO("L,1,1\r\nL,2,2\r\nL,3,3"))]
                ),
                (),
                {},
            ),
            (
                iterdp.CSVParser,
                IterableWrapper(
                    [("f1", StringIO("Label,1,1\nLabel,2,2\nLabel,3,3")), ("f2", StringIO("L,1,1\r\nL,2,2\r\nL,3,3"))]
                ),
                (),
                {},
            ),
            (iterdp.Cycler, None, (2,), {}),
            (iterdp.DataFrameMaker, IterableWrapper([(i,) for i in range(3)]), (), {"dtype": DTYPE}),
            (iterdp.Decompressor, None, (), {}),
            (iterdp.Enumerator, None, (2,), {}),
            (iterdp.FlatMapper, None, (_fake_fn_ls,), {}),
            (iterdp.FSSpecFileLister, ".", (), {}),
            (iterdp.FSSpecFileOpener, None, (), {}),
            (
                iterdp.FSSpecSaver,
                IterableWrapper([("1.txt", b"DATA1"), ("2.txt", b"DATA2"), ("3.txt", b"DATA3")]),
                (),
                {"mode": "wb", "filepath_fn": partial(_filepath_fn, dir=self.temp_dir.name)},
            ),
            (iterdp.GDriveReader, None, (), {}),
            (iterdp.HashChecker, None, ({},), {}),
            (iterdp.Header, None, (3,), {}),
            (iterdp.HttpReader, None, (), {}),
            # TODO (ejguan): Deterministic serialization is required
            #  (iterdp.InBatchShuffler, IterableWrapper(range(10)).batch(3), (), {}),
            (iterdp.InMemoryCacheHolder, None, (), {}),
            (iterdp.IndexAdder, IterableWrapper([{"a": 1, "b": 2}, {"c": 3, "a": 1}]), ("label",), {}),
            (iterdp.IoPathFileLister, ".", (), {}),
            (iterdp.IoPathFileOpener, None, (), {}),
            (
                iterdp.IoPathSaver,
                IterableWrapper([("1.txt", b"DATA1"), ("2.txt", b"DATA2"), ("3.txt", b"DATA3")]),
                (),
                {"mode": "wb", "filepath_fn": partial(_filepath_fn, dir=self.temp_dir.name)},
            ),
            (
                iterdp.IterKeyZipper,
                IterableWrapper([("a", 100), ("b", 200), ("c", 300)]),
                (IterableWrapper([("a", 1), ("b", 2), ("c", 3)]), itemgetter(0), itemgetter(0)),
                {},
            ),
            (
                iterdp.JsonParser,
                IterableWrapper(
                    [
                        ("1.json", StringIO('["fo", {"ba":["baz", null, 1.0, 2]}]')),
                        ("2.json", StringIO('{"__cx__": true, "r": 1, "i": 2}')),
                    ]
                ),
                (),
                {},
            ),
            (
                iterdp.LineReader,
                IterableWrapper(
                    [("file1", StringIO("Line1\nLine2")), ("file2", StringIO("Line2,1\r\nLine2,2\r\nLine2,3"))]
                ),
                (),
                {},
            ),
            (iterdp.MapToIterConverter, SequenceWrapper(range(10)), (), {}),
            (
                iterdp.MaxTokenBucketizer,
                IterableWrapper(["1", "22", "1", "4444", "333", "1", "22", "22", "333"]),
                (4,),
                {},
            ),
            (
                iterdp.MapKeyZipper,
                IterableWrapper([("a", 1), ("b", 2), ("c", 3)]),
                (SequenceWrapper({"a": 100, "b": 200, "c": 300}), itemgetter(0)),
                {},
            ),
            (iterdp.OnDiskCacheHolder, None, (), {}),
            (iterdp.OnlineReader, None, (), {}),
            (
                iterdp.ParagraphAggregator,
                IterableWrapper([("f1", "L1"), ("f1", "L2"), ("f2", "21"), ("f2", "22")]),
                (),
                {},
            ),
            (iterdp.ParquetDataFrameLoader, None, (), {"dtype": DTYPE}),
            (iterdp.RarArchiveLoader, None, (), {}),
            (
                iterdp.Rows2Columnar,
                IterableWrapper([[{"a": 1}, {"b": 2, "a": 1}], [{"a": 1, "b": 200}, {"c": 3}]]),
                (),
                {},
            ),
            (iterdp.SampleMultiplexer, {IterableWrapper([0] * 10): 0.5, IterableWrapper([1] * 10): 0.5}, (), {}),
            (
                iterdp.Saver,
                IterableWrapper([("1.txt", b"DATA1"), ("2.txt", b"DATA2"), ("3.txt", b"DATA3")]),
                (),
                {"mode": "wb", "filepath_fn": partial(_filepath_fn, dir=self.temp_dir.name)},
            ),
            (iterdp.TarArchiveLoader, None, (), {}),
            (iterdp.TFRecordLoader, None, (), {}),
            (iterdp.UnZipper, IterableWrapper([(i, i + 10) for i in range(10)]), (), {"sequence_length": 2}),
            (iterdp.WebDataset, IterableWrapper([("foo.txt", b"1"), ("bar.txt", b"2")]), (), {}),
            (iterdp.XzFileLoader, None, (), {}),
            (iterdp.ZipArchiveLoader, None, (), {}),
        ]

        picklable_datapipes = _filter_by_module_availability(picklable_datapipes)

        # Skipping value comparison for these DataPipes
        # Most of them return streams not comparable by `self.assertEqual`
        # Others are similar to caching where the outputs depend on other DataPipes
        dp_skip_comparison = {
            iterdp.Bz2FileLoader,
            iterdp.Decompressor,
            iterdp.FileOpener,
            iterdp.FSSpecFileOpener,
            iterdp.GDriveReader,
            iterdp.IoPathFileOpener,
            iterdp.HashChecker,
            iterdp.HttpReader,
            iterdp.OnDiskCacheHolder,
            iterdp.OnlineReader,
            iterdp.ParquetDataFrameLoader,
            iterdp.SampleMultiplexer,
            iterdp.RarArchiveLoader,
            iterdp.TarArchiveLoader,
            iterdp.TFRecordLoader,
            iterdp.XzFileLoader,
            iterdp.ZipArchiveLoader,
        }
        # These DataPipes produce multiple DataPipes as outputs and those should be compared
        dp_compare_children = {iterdp.UnZipper}

        for dpipe, custom_input, dp_args, dp_kwargs in picklable_datapipes:
            try:
                # Creating input (usually a DataPipe) for the specific dpipe being tested
                if custom_input is None:
                    custom_input = IterableWrapper(range(10))

                if dpipe in dp_skip_comparison:  # Mke sure they are picklable and loadable (no value comparison)
                    datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    serialized_dp = pickle.dumps(datapipe)
                    _ = pickle.loads(serialized_dp)
                elif dpipe in dp_compare_children:  # DataPipes that have children
                    dp1, dp2 = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_dp_with_children(dp1, dp2)
                else:  # Single DataPipe that requires comparison
                    datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    is_dataframe = issubclass(dpipe, (iterdp.DataFrameMaker, iterdp.ParquetDataFrameLoader))
                    self._serialization_test_for_single_dp(datapipe, is_dataframe=is_dataframe)
            except Exception as e:
                print(f"{dpipe} is failing.")
                raise e

    def test_serializable_with_dill(self):
        """Only for DataPipes that take in a function as argument"""
        input_dp = IterableWrapper(range(10))
        ref_idp = IterableWrapper(range(10))
        ref_mdp = SequenceWrapper(range(10))

        unpicklable_datapipes: List = [
            (iterdp.BatchMapper, (lambda batch: [d + 1 for d in batch], 2), {}),
            (iterdp.FlatMapper, (lambda x: [x, x],), {}),
            (iterdp.IterKeyZipper, (ref_idp, lambda x: x, None, True, 100), {}),
            (iterdp.MapKeyZipper, (ref_mdp, lambda x: x), {}),
            (iterdp.OnDiskCacheHolder, (lambda x: x,), {}),
            (iterdp.ParagraphAggregator, (lambda x: x,), {}),
        ]
        # Skipping value comparison for these DataPipes
        dp_skip_comparison = {iterdp.OnDiskCacheHolder, iterdp.ParagraphAggregator}
        for dpipe, dp_args, dp_kwargs in unpicklable_datapipes:
            if DILL_AVAILABLE:
                try:
                    if dpipe in dp_skip_comparison:  # Make sure they are picklable/loadable (no value comparison)
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                        serialized_dp = dill.dumps(datapipe)
                        _ = dill.loads(serialized_dp)
                    else:
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                        self._serialization_test_for_single_dp(datapipe)
                except Exception as e:
                    print(f"{dpipe} is failing.")
                    raise e

            else:
                dp_no_attribute_error = (iterdp.OnDiskCacheHolder,)
                try:
                    with warnings.catch_warnings(record=True) as wa:
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                        self.assertEqual(len(wa), 1)
                        self.assertRegex(str(wa[0].message), r"^Lambda function is not supported for pickle")
                        if isinstance(datapipe, dp_no_attribute_error):
                            _ = pickle.dumps(datapipe)
                        else:
                            with self.assertRaises(AttributeError):
                                _ = pickle.dumps(datapipe)
                except Exception as e:
                    print(f"{dpipe} is failing.")
                    raise e


class TestMapDataPipeSerialization(expecttest.TestCase):
    def _serialization_test_helper(self, datapipe):
        serialized_dp = pickle.dumps(datapipe)
        deserialized_dp = pickle.loads(serialized_dp)
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            print(f"{datapipe} is failing.")
            raise e

    def _serialization_test_for_dp_with_children(self, dp1, dp2):
        self._serialization_test_helper(dp1)
        self._serialization_test_helper(dp2)

    def test_serializable(self):
        picklable_datapipes: List = [
            (mapdp.InMemoryCacheHolder, None, (), {}),
            (mapdp.IterToMapConverter, IterableWrapper([(i, i) for i in range(10)]), (), {}),
            (mapdp.UnZipper, SequenceWrapper([(i, i + 10) for i in range(10)]), (), {"sequence_length": 2}),
        ]

        dp_skip_comparison = set()
        # These DataPipes produce multiple DataPipes as outputs and those should be compared
        dp_compare_children = {mapdp.UnZipper}

        for dpipe, custom_input, dp_args, dp_kwargs in picklable_datapipes:
            try:
                # Creating input (usually a DataPipe) for the specific dpipe being tested
                if custom_input is None:
                    custom_input = SequenceWrapper(range(10))

                if dpipe in dp_skip_comparison:  # Mke sure they are picklable and loadable (no value comparison)
                    datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    serialized_dp = pickle.dumps(datapipe)
                    _ = pickle.loads(serialized_dp)
                elif dpipe in dp_compare_children:  # DataPipes that have children
                    dp1, dp2 = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_dp_with_children(dp1, dp2)
                else:  # Single DataPipe that requires comparison
                    datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_helper(datapipe)
            except Exception as e:
                print(f"{dpipe} is failing.")
                raise e

    def test_serializable_with_dill(self):
        """Only for DataPipes that take in a function as argument"""
        pass


if __name__ == "__main__":
    unittest.main()
