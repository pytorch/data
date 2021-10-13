# Copyright (c) Facebook, Inc. and its affiliates.
import hashlib
import io

import expecttest
import itertools
import lzma
import os
import tarfile
import warnings
import unittest
import zipfile

from collections import defaultdict
from json.decoder import JSONDecodeError
import torch.utils.data.datapipes.iter
from torch.utils.data.datapipes.map import SequenceWrapper
from torch.testing._internal.common_utils import slowTest
import torchdata
from torchdata.datapipes.iter import (
    IterDataPipe,
    FileLister,
    FileLoader,
    IterableWrapper,
    InMemoryCacheHolder,
    KeyZipper,
    Cycler,
    Header,
    MapZipper,
    IndexAdder,
    IoPathFileLister,
    IoPathFileLoader,
    LineReader,
    ParagraphAggregator,
    Rows2Columnar,
    SampleMultiplexer,
    BucketBatcher,
    CSVParser,
    CSVDictParser,
    HashChecker,
    JsonParser,
    Saver,
    TarArchiveReader,
    ZipArchiveReader,
    XzFileReader,
    HttpReader,
    GDriveReader,
    OnlineReader,
)
from typing import Dict
from _utils._common_utils_for_test import (
    create_temp_dir_and_files,
    IDP_NoLen,
    get_name,
    reset_after_n_next_calls,
)

try:
    import iopath  # type: ignore[import] # noqa: F401 F403

    HAS_IOPATH = True
except ImportError:
    HAS_IOPATH = False
skipIfNoIOPath = unittest.skipIf(not HAS_IOPATH, "no iopath")


def test_torchdata_pytorch_consistency():
    def extract_datapipe_names(module):
        return {
            name
            for name, dp_type in module.__dict__.items()
            if not name.startswith("_") and isinstance(dp_type, type) and issubclass(dp_type, IterDataPipe)
        }

    pytorch_datapipes = extract_datapipe_names(torch.utils.data.datapipes.iter)
    torchdata_datapipes = extract_datapipe_names(torchdata.datapipes.iter)

    missing_datapipes = pytorch_datapipes - torchdata_datapipes
    if any(missing_datapipes):
        msg = (
            "The following datapipes are exposed under `torch.utils.data.datapipes.iter`, "
            "but not under `torchdata.datapipes.iter`:\n"
        )
        raise AssertionError(msg + "\n".join(sorted(missing_datapipes)))


class TestDataPipe(expecttest.TestCase):
    def test_in_memory_cache_holder_iterdatapipe(self):
        source_dp = IterableWrapper(range(10))
        cache_dp = source_dp.in_memory_cache(size=5)

        # Functional Test: Cache DP should just return the data without changing the values
        res1 = list(cache_dp)
        self.assertEqual(list(range(10)), res1)

        # Functional Test: Ensure the objects are the same ones from source DataPipe
        res1 = list(cache_dp)
        res2 = list(cache_dp)
        self.assertTrue(id(source) == id(cache) for source, cache in zip(source_dp, res1))
        self.assertTrue(id(source) == id(cache) for source, cache in zip(source_dp, res2))

        # TODO: Figure out a way to consistently test caching when size is in megabytes

        # Reset Test: reset the DataPipe after reading part of it
        cache_dp = InMemoryCacheHolder(source_dp, size=5)
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(cache_dp, n_elements_before_reset)
        self.assertEqual(list(range(5)), res_before_reset)
        self.assertEqual(list(range(10)), res_after_reset)

        # __len__ Test: inherits length from source_dp
        self.assertEqual(10, len(cache_dp))

        # __len__ Test: source_dp has no len and cache is not yet loaded
        source_dp_no_len = IDP_NoLen(range(10))
        cache_dp = InMemoryCacheHolder(source_dp_no_len, size=5)
        with self.assertRaisesRegex(TypeError, "doesn't have valid length until the cache is loaded"):
            len(cache_dp)

        # __len__ Test: source_dp has no len but we still can calculate after cache is loaded
        list(cache_dp)
        self.assertEqual(10, len(cache_dp))

    def test_keyzipper_iterdatapipe(self):

        source_dp = IterableWrapper(range(10))
        ref_dp = IterableWrapper(range(20))

        # Functional Test: Output should be a zip list of tuple
        zip_dp = source_dp.zip_by_key(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )
        self.assertEqual([(i, i) for i in range(10)], list(zip_dp))

        # Functional Test: keep_key=True, and key should show up as the first element
        zip_dp_w_key = source_dp.zip_by_key(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=True, buffer_size=10
        )
        self.assertEqual([(i, i, i) for i in range(10)], list(zip_dp_w_key))

        # Functional Test: element is in source but missing in reference
        ref_dp_missing = IterableWrapper(range(1, 10))
        zip_dp = source_dp.zip_by_key(
            ref_datapipe=ref_dp_missing, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )
        with self.assertRaisesRegex(BufferError, r"No matching key can be found"):
            list(zip_dp)

        # Functional Test: Buffer is not large enough, hence, element can't be found and raises error
        ref_dp_end = IterableWrapper(list(range(1, 10)) + [0])
        zip_dp = source_dp.zip_by_key(
            ref_datapipe=ref_dp_end, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=5
        )
        it = iter(zip_dp)
        with warnings.catch_warnings(record=True) as wa:
            # In order to find '0' at the end, the buffer is filled, hence the warning
            # and ref_dp is fully traversed
            self.assertEqual((0, 0,), next(it))
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Buffer reaches the upper limit")
        with self.assertRaisesRegex(BufferError, r"No matching key can be found"):
            # '1' cannot be find because the value was thrown out when buffer was filled
            next(it)

        # Functional Test: Buffer is just big enough
        zip_dp = source_dp.zip_by_key(
            ref_datapipe=ref_dp_end, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=10
        )
        self.assertEqual([(i, i) for i in range(10)], list(zip_dp))

        # Reset Test: reset the DataPipe after reading part of it
        zip_dp = KeyZipper(
            source_datapipe=source_dp,
            ref_datapipe=ref_dp,
            key_fn=lambda x: x,
            ref_key_fn=lambda x: x,
            keep_key=False,
            buffer_size=10,
        )
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(zip_dp, n_elements_before_reset)
        self.assertEqual([(i, i) for i in range(5)], res_before_reset)
        self.assertEqual([(i, i) for i in range(10)], res_after_reset)

        # __len__ Test: inherits length from source_dp
        self.assertEqual(10, len(zip_dp))

    def test_cycler_iterdatapipe(self):
        source_dp = IterableWrapper(range(5))

        # Functional Test: cycle for finite number of times and ends
        cycler_dp = source_dp.cycle(3)
        self.assertEqual(list(range(5)) * 3, list(cycler_dp))

        # Functional Test: cycle for indefinitely
        cycler_dp = source_dp.cycle()
        it = iter(cycler_dp)
        for expected_val in list(range(5)) * 10:
            self.assertEqual(expected_val, next(it))

        # Functional Test: zero is allowed but immediately triggers StopIteration
        cycler_dp = source_dp.cycle(0)
        self.assertEqual([], list(cycler_dp))

        # Functional Test: negative value is not allowed
        with self.assertRaisesRegex(ValueError, "Expected non-negative count"):
            source_dp.cycle(-1)

        # Reset Test:
        cycler_dp = Cycler(source_dp, count=2)
        n_elements_before_reset = 4
        res_before_reset, res_after_reset = reset_after_n_next_calls(cycler_dp, n_elements_before_reset)
        self.assertEqual(list(range(4)), res_before_reset)
        self.assertEqual(list(range(5)) * 2, res_after_reset)

        # __len__ Test: returns length when count is not None
        self.assertEqual(10, len(cycler_dp))

        # __len__ Test: inherits length from source_dp
        cycler_dp = Cycler(source_dp)
        with self.assertRaisesRegex(TypeError, "instance cycles forever, and therefore doesn't have valid length"):
            len(cycler_dp)

    def test_header_iterdatapipe(self):
        # Functional Test: ensure the limit is enforced
        source_dp = IterableWrapper(range(20))
        header_dp = source_dp.header(5)
        self.assertEqual(list(range(5)), list(header_dp))

        # Functional Test: ensure it works when the source has less elements than the limit
        source_dp = IterableWrapper(range(5))
        header_dp = source_dp.header(100)
        self.assertEqual(list(range(5)), list(header_dp))

        # Reset Test:
        source_dp = IterableWrapper(range(20))
        header_dp = Header(source_dp, 5)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(header_dp, n_elements_before_reset)
        self.assertEqual(list(range(2)), res_before_reset)
        self.assertEqual(list(range(5)), res_after_reset)
        self.assertEqual(list(range(5)), list(header_dp))

        # __len__ Test: returns the limit when it is less than the length of source
        self.assertEqual(5, len(header_dp))

        # TODO: __len__ Test: returns the length of source when it is less than the limit
        # header_dp = source_dp.header(30)
        # self.assertEqual(20, len(header_dp))

    def test_index_adder_iterdatapipe(self):
        letters = "abcdefg"
        source_dp = IterableWrapper([{i: i} for i in letters])
        index_adder_dp = source_dp.add_index()
        it = iter(index_adder_dp)

        def dict_content_test_helper(iterator):
            for i, curr_dict in enumerate(iterator):
                self.assertEqual(i, curr_dict["index"])
                self.assertTrue(letters[i] in curr_dict)

        # Functional Test: ensure that the correct index value is added to each element (dict)
        dict_content_test_helper(it)

        # Functional Test: raises error when the elements of source_dp is not of type Dict
        source_dp = IterableWrapper(range(10))
        index_adder_dp = source_dp.add_index()
        it = iter(index_adder_dp)
        with self.assertRaisesRegex(NotImplementedError, "We only support adding index to row or batch in dict type"):
            next(it)

        # Reset Test
        source_dp = IterableWrapper([{i: i} for i in "abcdefg"])
        index_adder_dp = IndexAdder(source_dp)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(index_adder_dp, n_elements_before_reset)
        dict_content_test_helper(iter(res_before_reset))
        dict_content_test_helper(iter(res_after_reset))

        # __len__ Test: returns length of source DataPipe
        self.assertEqual(7, len(index_adder_dp))

    def test_line_reader_iterdatapipe(self):
        text1 = "Line1\nLine2"
        text2 = "Line2,1\nLine2,2\nLine2,3"

        # Functional Test: read lines correctly
        source_dp = IterableWrapper([("file1", io.StringIO(text1)), ("file2", io.StringIO(text2))])
        line_reader_dp = source_dp.readlines()
        expected_result = [("file1", line) for line in text1.split("\n")] + [
            ("file2", line) for line in text2.split("\n")
        ]
        self.assertEqual(expected_result, list(line_reader_dp))

        # Functional Test: strip new lines for bytes
        source_dp = IterableWrapper(
            [("file1", io.BytesIO(text1.encode("utf-8"))), ("file2", io.BytesIO(text2.encode("utf-8")))]
        )
        line_reader_dp = source_dp.readlines()
        expected_result_bytes = [("file1", line.encode("utf-8")) for line in text1.split("\n")] + [
            ("file2", line.encode("utf-8")) for line in text2.split("\n")
        ]
        self.assertEqual(expected_result_bytes, list(line_reader_dp))

        # Functional Test: do not strip new lines
        source_dp = IterableWrapper([("file1", io.StringIO(text1)), ("file2", io.StringIO(text2))])
        line_reader_dp = source_dp.readlines(strip_newline=False)
        expected_result = [
            ("file1", "Line1\n"),
            ("file1", "Line2"),
            ("file2", "Line2,1\n"),
            ("file2", "Line2,2\n"),
            ("file2", "Line2,3"),
        ]
        self.assertEqual(expected_result, list(line_reader_dp))

        # Reset Test:
        source_dp = IterableWrapper([("file1", io.StringIO(text1)), ("file2", io.StringIO(text2))])
        line_reader_dp = LineReader(source_dp, strip_newline=False)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(line_reader_dp, n_elements_before_reset)
        self.assertEqual(expected_result[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_result, res_after_reset)

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "has no len"):
            len(line_reader_dp)

    def test_paragraph_aggregator_iterdatapipe(self):
        # Functional Test: aggregate lines correctly
        source_dp = IterableWrapper(
            [("file1", "Line1"), ("file1", "Line2"), ("file2", "Line2,1"), ("file2", "Line2,2"), ("file2", "Line2,3")]
        )
        para_agg_dp = source_dp.lines_to_paragraphs()
        self.assertEqual([("file1", "Line1\nLine2"), ("file2", "Line2,1\nLine2,2\nLine2,3")], list(para_agg_dp))

        # Functional Test: aggregate lines correctly with different joiner
        para_agg_dp = source_dp.lines_to_paragraphs(joiner=lambda ls: " ".join(ls))
        self.assertEqual([("file1", "Line1 Line2"), ("file2", "Line2,1 Line2,2 Line2,3")], list(para_agg_dp))

        # Reset Test: each yield is for a single file
        para_agg_dp = ParagraphAggregator(source_dp)
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(para_agg_dp, n_elements_before_reset)
        self.assertEqual([("file1", "Line1\nLine2")], res_before_reset)
        self.assertEqual([("file1", "Line1\nLine2"), ("file2", "Line2,1\nLine2,2\nLine2,3")], res_after_reset)

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "has no len"):
            len(para_agg_dp)

    def test_rows_to_columnar_iterdatapipe(self):
        # Functional Test: working with DataPipe with dict
        column_names_dict = {"a", "b", "c"}
        source_dp = IterableWrapper(
            [
                [{l: i for i, l in enumerate("abc")}, {l: i * 10 for i, l in enumerate("abc")}],
                [{l: i + 100 for i, l in enumerate("abc")}, {l: (i + 100) * 10 for i, l in enumerate("abc")}],
            ]
        )
        result_dp = source_dp.rows2columnar(column_names_dict)
        batch1 = defaultdict(list, {"a": [0, 0], "b": [1, 10], "c": [2, 20]})
        batch2 = defaultdict(list, {"a": [100, 1000], "b": [101, 1010], "c": [102, 1020]})
        expected_output = [batch1, batch2]
        self.assertEqual(expected_output, list(result_dp))

        # Functional Test: working with DataPipe with list
        column_names_list = ["a", "b", "c"]
        source_dp = IterableWrapper(
            [
                [[i for i, _ in enumerate("abc")], [i * 10 for i, _ in enumerate("abc")]],
                [[i + 100 for i, _ in enumerate("abc")], [(i + 100) * 10 for i, _ in enumerate("abc")]],
            ]
        )
        result_dp = source_dp.rows2columnar(column_names_list)
        self.assertEqual(expected_output, list(result_dp))

        # Reset Test:
        result_dp = Rows2Columnar(source_dp, column_names_list)
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(result_dp, n_elements_before_reset)
        self.assertEqual([expected_output[0]], res_before_reset)
        self.assertEqual(expected_output, res_after_reset)

        # __len__ Test: returns length of source DataPipe
        self.assertEqual(2, len(result_dp))

    def test_sample_multiplexer_iterdatapipe(self):
        # Functional Test: yields all values from the sources
        source_dp1 = IterableWrapper([0] * 10)
        source_dp2 = IterableWrapper([1] * 10)
        d: Dict[IterDataPipe, float] = {source_dp1: 99999999, source_dp2: 0.0000001}
        sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d, seed=0)
        result = list(sample_mul_dp)
        self.assertEqual([0] * 10 + [1] * 10, result)

        # Functional Test: raises error for empty dict
        with self.assertRaisesRegex(ValueError, "Empty dictionary"):
            SampleMultiplexer(pipes_to_weights_dict={}, seed=0)

        # Functional Test: raises error for negative or zero weight
        d = {source_dp1: 99999999, source_dp2: 0}
        with self.assertRaisesRegex(ValueError, "Expecting a positive and non-zero weight"):
            SampleMultiplexer(pipes_to_weights_dict=d, seed=0)

        # Reset Test
        d = {source_dp1: 99999999, source_dp2: 0.0000001}
        sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d, seed=0)
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(sample_mul_dp, n_elements_before_reset)
        self.assertEqual([0] * n_elements_before_reset, res_before_reset)
        self.assertEqual([0] * 10 + [1] * 10, res_after_reset)

        # __len__ Test: returns the sum of the lengths of the sources
        self.assertEqual(20, len(sample_mul_dp))

    def test_bucket_batcher_iterdatapipe(self):
        source_dp = IterableWrapper(range(10))

        # Functional Test: drop last reduces length
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=True, batch_num=100, bucket_num=1, in_batch_shuffle=True
        )
        self.assertEqual(3, len(batch_dp))
        self.assertEqual(9, len(list(batch_dp.unbatch())))

        # Functional Test: drop last is False preserves length
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=False, batch_num=100, bucket_num=1, in_batch_shuffle=False
        )
        self.assertEqual(4, len(batch_dp))
        self.assertEqual(10, len(list(batch_dp.unbatch())))

        # Functional Test: using sort_key, with in_batch_shuffle
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=True, batch_num=100, bucket_num=1, in_batch_shuffle=True, sort_key=lambda x: x
        )
        # bucket_num = 1 means there will be no shuffling if a sort key is given
        self.assertEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]], list(batch_dp))
        self.assertEqual(3, len(batch_dp))
        self.assertEqual(9, len(list(batch_dp.unbatch())))

        # Functional Test: using sort_key, without in_batch_shuffle
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=True, batch_num=100, bucket_num=2, in_batch_shuffle=False, sort_key=lambda x: x
        )
        self.assertEqual(3, len(batch_dp))
        self.assertEqual(9, len(list(batch_dp.unbatch())))

        # Reset Test:
        batch_dp = BucketBatcher(
            source_dp,
            batch_size=3,
            drop_last=True,
            batch_num=100,
            bucket_num=2,
            in_batch_shuffle=False,
            sort_key=lambda x: x,
        )
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        self.assertEqual(n_elements_before_reset, len(res_before_reset))
        self.assertEqual(6, len([item for batch in res_before_reset for item in batch]))
        self.assertEqual(3, len(res_after_reset))
        self.assertEqual(9, len([item for batch in res_after_reset for item in batch]))

        # __len__ Test: returns the number of batches
        self.assertEqual(3, len(batch_dp))


class TestDataPipeWithIO(expecttest.TestCase):
    def setUp(self):
        ret = create_temp_dir_and_files()
        self.temp_dir = ret[0][0]
        self.temp_files = ret[0][1:]
        self.temp_sub_dir = ret[1][0]
        self.temp_sub_files = ret[1][1:]

    def tearDown(self):
        try:
            self.temp_sub_dir.cleanup()
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(f"TestDataPipeWithIO was not able to cleanup temp dir due to {e}")

    def _custom_files_set_up(self, files):
        for fname, content in files.items():
            temp_file_path = os.path.join(self.temp_dir.name, fname)
            with open(temp_file_path, "w") as f:
                f.write(content)

    def _compressed_files_comparison_helper(self, expected_files, result, check_length: bool = True):
        if check_length:
            self.assertEqual(len(expected_files), len(result))
        for res, expected_file in itertools.zip_longest(result, expected_files):
            self.assertTrue(res is not None and expected_file is not None)
            self.assertEqual(os.path.basename(res[0]), os.path.basename(expected_file))
            with open(expected_file, "rb") as f:
                self.assertEqual(res[1].read(), f.read())
            res[1].close()

    def _unordered_compressed_files_comparison_helper(self, expected_files, result, check_length: bool = True):
        expected_names_to_files = {os.path.basename(f): f for f in expected_files}
        if check_length:
            self.assertEqual(len(expected_files), len(result))
        for res in result:
            fname = os.path.basename(res[0])
            self.assertTrue(fname is not None)
            self.assertTrue(fname in expected_names_to_files)
            with open(expected_names_to_files[fname], "rb") as f:
                self.assertEqual(res[1].read(), f.read())
            res[1].close()

    def test_csv_parser_iterdatapipe(self):
        def make_path(fname):
            return f"{self.temp_dir.name}/{fname}"

        csv_files = {"1.csv": "key,item\na,1\nb,2", "empty.csv": "", "empty2.csv": "\n"}
        self._custom_files_set_up(csv_files)
        datapipe1 = IterableWrapper([make_path(fname) for fname in ["1.csv", "empty.csv", "empty2.csv"]])
        datapipe2 = FileLoader(datapipe1)
        datapipe3 = datapipe2.map(get_name)

        # Functional Test: yield one row at time from each file, skipping over empty content
        csv_parser_dp = datapipe3.parse_csv()
        expected_res = [["key", "item"], ["a", "1"], ["b", "2"], []]
        self.assertEqual(expected_res, list(csv_parser_dp))

        # Functional Test: yield one row at time from each file, skipping over empty content and header
        csv_parser_dp = datapipe3.parse_csv(skip_lines=1)
        expected_res = [["a", "1"], ["b", "2"]]
        self.assertEqual(expected_res, list(csv_parser_dp))

        # Functional Test: yield one row at time from each file with file name, skipping over empty content
        csv_parser_dp = datapipe3.parse_csv(return_path=True)
        expected_res = [("1.csv", ["key", "item"]), ("1.csv", ["a", "1"]), ("1.csv", ["b", "2"]), ("empty2.csv", [])]
        self.assertEqual(expected_res, list(csv_parser_dp))

        # Reset Test:
        csv_parser_dp = CSVParser(datapipe3, return_path=True)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(csv_parser_dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "has no len"):
            len(csv_parser_dp)

    def test_csv_dict_parser_iterdatapipe(self):
        def get_name(path_and_stream):
            return os.path.basename(path_and_stream[0]), path_and_stream[1]

        csv_files = {"1.csv": "key,item\na,1\nb,2", "empty.csv": "", "empty2.csv": "\n"}
        self._custom_files_set_up(csv_files)
        datapipe1 = FileLister(self.temp_dir.name, "*.csv")
        datapipe2 = FileLoader(datapipe1)
        datapipe3 = datapipe2.map(get_name)

        # Functional Test: yield one row at a time as dict, with the first row being the header (key)
        csv_dict_parser_dp = datapipe3.parse_csv_as_dict()
        expected_res1 = [{"key": "a", "item": "1"}, {"key": "b", "item": "2"}]
        self.assertEqual(expected_res1, list(csv_dict_parser_dp))

        # Functional Test: yield one row at a time as dict, skip over first row, with the second row being the header
        csv_dict_parser_dp = datapipe3.parse_csv_as_dict(skip_lines=1)
        expected_res2 = [{"a": "b", "1": "2"}]
        self.assertEqual(expected_res2, list(csv_dict_parser_dp))

        # Functional Test: yield one row at a time as dict with file name, and the first row being the header (key)
        csv_dict_parser_dp = datapipe3.parse_csv_as_dict(return_path=True)
        expected_res3 = [("1.csv", {"key": "a", "item": "1"}), ("1.csv", {"key": "b", "item": "2"})]
        self.assertEqual(expected_res3, list(csv_dict_parser_dp))

        # Reset Test
        csv_dict_parser_dp = CSVDictParser(datapipe3)
        expected_res4 = [{"key": "a", "item": "1"}, {"key": "b", "item": "2"}]
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(csv_dict_parser_dp, n_elements_before_reset)
        self.assertEqual(expected_res4[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res4, res_after_reset)

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "has no len"):
            len(csv_dict_parser_dp)

    def test_hash_checker_iterdatapipe(self):
        hash_dict = {}

        def fill_hash_dict():
            for path in self.temp_files:
                with open(path, "r") as f:
                    hash_func = hashlib.sha256()
                    content = f.read().encode("utf-8")
                    hash_func.update(content)
                    hash_dict[path] = hash_func.hexdigest()

        fill_hash_dict()

        datapipe1 = FileLister(self.temp_dir.name, "*")
        datapipe2 = FileLoader(datapipe1)
        hash_check_dp = HashChecker(datapipe2, hash_dict)

        # Functional Test: Ensure the DataPipe values are unchanged if the hashes are the same
        for (expected_path, expected_stream), (actual_path, actual_stream) in zip(datapipe2, hash_check_dp):
            self.assertEqual(expected_path, actual_path)
            self.assertEqual(expected_stream.read(), actual_stream.read())

        # Functional Test: Ensure the rewind option works, and the stream is empty when there is no rewind
        hash_check_dp_no_reset = HashChecker(datapipe2, hash_dict, rewind=False)
        for (expected_path, _), (actual_path, actual_stream) in zip(datapipe2, hash_check_dp_no_reset):
            self.assertEqual(expected_path, actual_path)
            self.assertEqual(b"", actual_stream.read())

        # Functional Test: Error when file/path is not in hash_dict
        hash_check_dp = HashChecker(datapipe2, {})
        it = iter(hash_check_dp)
        with self.assertRaisesRegex(RuntimeError, "Unspecified hash for file"):
            next(it)

        # Functional Test: Error when the hash is different
        hash_dict[self.temp_files[0]] = "WRONG HASH"
        hash_check_dp = HashChecker(datapipe2, hash_dict)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            list(hash_check_dp)

        # Reset Test:
        fill_hash_dict()  # Reset the dict with correct values because we changed it in the last test case
        hash_check_dp = datapipe2.check_hash(hash_dict)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(hash_check_dp, n_elements_before_reset)
        for (expected_path, expected_stream), (actual_path, actual_stream) in zip(datapipe2, res_before_reset):
            self.assertEqual(expected_path, actual_path)
            self.assertEqual(expected_stream.read(), actual_stream.read())
        for (expected_path, expected_stream), (actual_path, actual_stream) in zip(datapipe2, res_after_reset):
            self.assertEqual(expected_path, actual_path)
            self.assertEqual(expected_stream.read(), actual_stream.read())

        # __len__ Test: returns the length of source DataPipe
        with self.assertRaisesRegex(TypeError, "FileLoaderIterDataPipe instance doesn't have valid length"):
            len(hash_check_dp)

    def test_map_zipper_datapipe(self):
        source_dp = IterableWrapper(range(10))
        map_dp = SequenceWrapper(["even", "odd"])

        # Functional Test: ensure the hash join is working and return tuple by default
        def odd_even(i: int) -> int:
            return i % 2

        result_dp = source_dp.zip_with_map(map_dp, odd_even)

        def odd_even_string(i: int) -> str:
            return "odd" if i % 2 else "even"

        expected_res = [(i, odd_even_string(i)) for i in range(10)]
        self.assertEqual(expected_res, list(result_dp))

        # Functional Test: ensure that a custom merge function works
        def custom_merge(a, b):
            return f"{a} is a {b} number."

        result_dp = source_dp.zip_with_map(map_dp, odd_even, custom_merge)
        expected_res2 = [f"{i} is a {odd_even_string(i)} number." for i in range(10)]
        self.assertEqual(expected_res2, list(result_dp))

        # Functional Test: raises error when key is invalid
        def odd_even_bug(i: int) -> int:
            return 2 if i == 0 else i % 2

        result_dp = MapZipper(source_dp, map_dp, odd_even_bug)
        it = iter(result_dp)
        with self.assertRaisesRegex(KeyError, "is not a valid key in the given MapDataPipe"):
            next(it)

        # Reset Test:
        n_elements_before_reset = 4
        result_dp = source_dp.zip_with_map(map_dp, odd_even)
        res_before_reset, res_after_reset = reset_after_n_next_calls(result_dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

        # __len__ Test: returns the length of source DataPipe
        result_dp = source_dp.zip_with_map(map_dp, odd_even)
        self.assertEqual(len(source_dp), len(result_dp))

    def test_json_parser_iterdatapipe(self):
        def is_empty_json(path_and_stream):
            return path_and_stream[0] == "empty.json"

        def is_nonempty_json(path_and_stream):
            return path_and_stream[0] != "empty.json"

        json_files = {
            "1.json": '["foo", {"bar":["baz", null, 1.0, 2]}]',
            "empty.json": "",
            "2.json": '{"__complex__": true, "real": 1, "imag": 2}',
        }
        self._custom_files_set_up(json_files)
        datapipe1 = IterableWrapper([f"{self.temp_dir.name}/{fname}" for fname in ["empty.json", "1.json", "2.json"]])
        datapipe2 = FileLoader(datapipe1)
        datapipe3 = datapipe2.map(get_name)
        datapipe_empty = datapipe3.filter(is_empty_json)
        datapipe_nonempty = datapipe3.filter(is_nonempty_json)

        empty_json_dp = datapipe_empty.parse_json_files()
        it = iter(empty_json_dp)
        # Functional Test: dp fails when empty JSON file is given
        with self.assertRaisesRegex(JSONDecodeError, "Expecting value"):
            next(it)

        # Functional Test: dp yields one json file at a time
        json_dp = datapipe_nonempty.parse_json_files()
        expected_res = [
            ("1.json", ["foo", {"bar": ["baz", None, 1.0, 2]}]),
            ("2.json", {"__complex__": True, "real": 1, "imag": 2}),
        ]
        self.assertEqual(expected_res, list(json_dp))

        # Reset Test:
        json_dp = JsonParser(datapipe_nonempty)
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(json_dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

        # __len__ Test: length isn't implemented since it cannot be known ahead of time
        with self.assertRaisesRegex(TypeError, "len"):
            len(json_dp)

    def test_saver_iterdatapipe(self):
        def filepath_fn(name: str) -> str:
            return os.path.join(self.temp_dir.name, os.path.basename(name))

        # Functional Test: Saving some data
        name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
        source_dp = IterableWrapper(sorted(name_to_data.items()))
        saver_dp = source_dp.save_to_disk(filepath_fn=filepath_fn)
        res_file_paths = list(saver_dp)
        expected_paths = [filepath_fn(name) for name in name_to_data.keys()]
        self.assertEqual(expected_paths, res_file_paths)
        for name in name_to_data.keys():
            p = filepath_fn(name)
            with open(p, "r") as f:
                self.assertEqual(name_to_data[name], f.read().encode())

        # Reset Test:
        saver_dp = Saver(source_dp, filepath_fn=filepath_fn)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(saver_dp, n_elements_before_reset)
        self.assertEqual([filepath_fn("1.txt"), filepath_fn("2.txt")], res_before_reset)
        self.assertEqual(expected_paths, res_after_reset)
        for name in name_to_data.keys():
            p = filepath_fn(name)
            with open(p, "r") as f:
                self.assertEqual(name_to_data[name], f.read().encode())

        # __len__ Test: returns the length of source DataPipe
        self.assertEqual(3, len(saver_dp))

    def test_tar_archive_reader_iterdatapipe(self):
        temp_tarfile_pathname = os.path.join(self.temp_dir.name, "test_tar.tar")
        with tarfile.open(temp_tarfile_pathname, "w:gz") as tar:
            tar.add(self.temp_files[0])
            tar.add(self.temp_files[1])
            tar.add(self.temp_files[2])
        datapipe1 = FileLister(self.temp_dir.name, "*.tar")
        datapipe2 = FileLoader(datapipe1)
        tar_reader_dp = TarArchiveReader(datapipe2)

        # Functional Test: Read extracted files before reaching the end of the tarfile
        self._compressed_files_comparison_helper(self.temp_files, tar_reader_dp, check_length=False)

        # Functional Test: Read extracted files after reaching the end of the tarfile
        data_refs = list(tar_reader_dp)
        self._compressed_files_comparison_helper(self.temp_files, data_refs)

        # Reset Test: reset the DataPipe after reading part of it
        tar_reader_dp = datapipe2.read_from_tar()
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(tar_reader_dp, n_elements_before_reset)
        # Check result accumulated before reset
        self._compressed_files_comparison_helper(self.temp_files[:n_elements_before_reset], res_before_reset)
        # Check result accumulated after reset
        self._compressed_files_comparison_helper(self.temp_files, res_after_reset)

        # __len__ Test: doesn't have valid length
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length"):
            len(tar_reader_dp)

    def test_zip_archive_reader_iterdatapipe(self):
        temp_zipfile_pathname = os.path.join(self.temp_dir.name, "test_zip.zip")
        with zipfile.ZipFile(temp_zipfile_pathname, "w") as myzip:
            myzip.write(self.temp_files[0])
            myzip.write(self.temp_files[1])
            myzip.write(self.temp_files[2])
        datapipe1 = FileLister(self.temp_dir.name, "*.zip")
        datapipe2 = FileLoader(datapipe1)
        zip_reader_dp = ZipArchiveReader(datapipe2)

        # Functional Test: read extracted files before reaching the end of the zipfile
        self._compressed_files_comparison_helper(self.temp_files, zip_reader_dp, check_length=False)

        # Functional Test: read extracted files after reaching the end of the zipile
        data_refs = list(zip_reader_dp)
        self._compressed_files_comparison_helper(self.temp_files, data_refs)

        # Reset Test: reset the DataPipe after reading part of it
        zip_reader_dp = datapipe2.read_from_zip()
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(zip_reader_dp, n_elements_before_reset)
        # Check the results accumulated before reset
        self._compressed_files_comparison_helper(self.temp_files[:n_elements_before_reset], res_before_reset)
        # Check the results accumulated after reset
        self._compressed_files_comparison_helper(self.temp_files, res_after_reset)

        # __len__ Test: doesn't have valid length
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length"):
            len(zip_reader_dp)

    def test_xz_archive_reader_iterdatapipe(self):
        # Worth noting that the .tar and .zip tests write multiple files into the same compressed file
        # Whereas we create multiple .xz files in the same directories below.
        for path in self.temp_files:
            fname = os.path.basename(path)
            temp_xzfile_pathname = os.path.join(self.temp_dir.name, f"{fname}.xz")
            with open(path, "r") as f:
                with lzma.open(temp_xzfile_pathname, "w") as xz:
                    xz.write(f.read().encode("utf-8"))
        datapipe1 = FileLister(self.temp_dir.name, "*.xz")
        datapipe2 = FileLoader(datapipe1)
        xz_reader_dp = XzFileReader(datapipe2)

        # Functional Test: Read extracted files before reaching the end of the xzfile
        self._unordered_compressed_files_comparison_helper(self.temp_files, xz_reader_dp, check_length=False)

        # Functional Test: Read extracted files after reaching the end of the xzfile
        data_refs = list(xz_reader_dp)
        self._unordered_compressed_files_comparison_helper(self.temp_files, data_refs)

        # Reset Test: reset the DataPipe after reading part of it
        xz_reader_dp = datapipe2.read_from_xz()
        n_elements_before_reset = 1
        res_before_reset, res_after_reset = reset_after_n_next_calls(xz_reader_dp, n_elements_before_reset)
        # Check result accumulated before reset
        self.assertEqual(n_elements_before_reset, len(res_before_reset))
        self._unordered_compressed_files_comparison_helper(self.temp_files, res_before_reset, check_length=False)
        # Check result accumulated after reset
        self._unordered_compressed_files_comparison_helper(self.temp_files, res_after_reset)

        # Reset Test: Ensure the order is consistent between iterations
        for r1, r2 in zip(xz_reader_dp, xz_reader_dp):
            self.assertEqual(r1[0], r2[0])

        # __len__ Test: doesn't have valid length
        with self.assertRaisesRegex(TypeError, "instance doesn't have valid length"):
            len(xz_reader_dp)

    # TODO (ejguan): this test currently only covers reading from local
    # filesystem. It needs to be modified once test data can be stored on
    # gdrive/s3/onedrive
    @skipIfNoIOPath
    def test_io_path_file_lister_iterdatapipe(self):
        datapipe = IoPathFileLister(root=self.temp_sub_dir.name)

        # check all file paths within sub_folder are listed
        for path in datapipe:
            self.assertTrue(path in self.temp_sub_files)

    @skipIfNoIOPath
    def test_io_path_file_loader_iterdatapipe(self):
        datapipe1 = IoPathFileLister(root=self.temp_sub_dir.name)
        datapipe2 = IoPathFileLoader(datapipe1)

        # check contents of file match
        for _, f in datapipe2:
            self.assertEqual(f.read(), "0123456789abcdef")


class TestDataPipeConnection(expecttest.TestCase):
    @slowTest
    def test_http_reader_iterdatapipe(self):

        file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        expected_file_name = "LICENSE"
        expected_MD5_hash = "6fc98cce3570de1956f7dbfcb9ca9dd1"
        http_reader_dp = HttpReader(IterableWrapper([file_url]))

        # Functional Test: test if the Http Reader can download and read properly
        reader_dp = http_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(b"BSD" in line)

        # Reset Test: http_reader_dp has been read, but we reset when calling check_hash()
        check_cache_dp = http_reader_dp.check_hash({file_url: expected_MD5_hash}, "md5", rewind=False)
        it = iter(check_cache_dp)
        path, stream = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(io.BufferedReader, type(stream))

        # __len__ Test: returns the length of source DataPipe
        source_dp = IterableWrapper([file_url])
        http_dp = HttpReader(source_dp)
        self.assertEqual(1, len(http_dp))

    @slowTest
    def test_gdrive_iterdatapipe(self):

        amazon_review_url = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
        expected_file_name = "amazon_review_polarity_csv.tar.gz"
        expected_MD5_hash = "fe39f8b653cada45afd5792e0f0e8f9b"
        gdrive_reader_dp = GDriveReader(IterableWrapper([amazon_review_url]))

        # Functional Test: test if the GDrive Reader can download and read properly
        reader_dp = gdrive_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(line != b"")

        # Reset Test: gdrive_reader_dp has been read, but we reset when calling check_hash()
        check_cache_dp = gdrive_reader_dp.check_hash({expected_file_name: expected_MD5_hash}, "md5", rewind=False)
        it = iter(check_cache_dp)
        path, stream = next(it)
        self.assertEqual(expected_file_name, os.path.basename(path))
        self.assertTrue(io.BufferedReader, type(stream))

        # __len__ Test: returns the length of source DataPipe
        source_dp = IterableWrapper([amazon_review_url])
        gdrive_dp = GDriveReader(source_dp)
        self.assertEqual(1, len(gdrive_dp))

    @slowTest
    def test_online_iterdatapipe(self):

        license_file_url = "https://raw.githubusercontent.com/pytorch/data/main/LICENSE"
        amazon_review_url = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM"
        expected_license_file_name = "LICENSE"
        expected_amazon_file_name = "amazon_review_polarity_csv.tar.gz"
        expected_license_MD5_hash = "6fc98cce3570de1956f7dbfcb9ca9dd1"
        expected_amazon_MD5_hash = "fe39f8b653cada45afd5792e0f0e8f9b"

        file_hash_dict = {
            license_file_url: expected_license_MD5_hash,
            expected_amazon_file_name: expected_amazon_MD5_hash,
        }

        # Functional Test: can read from GDrive links
        online_reader_dp = OnlineReader(IterableWrapper([amazon_review_url]))
        reader_dp = online_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_amazon_file_name, os.path.basename(path))
        self.assertTrue(line != b"")

        # Functional Test: can read from other links
        online_reader_dp = OnlineReader(IterableWrapper([license_file_url]))
        reader_dp = online_reader_dp.readlines()
        it = iter(reader_dp)
        path, line = next(it)
        self.assertEqual(expected_license_file_name, os.path.basename(path))
        self.assertTrue(line != b"")

        # Reset Test: reset online_reader_dp by calling check_hash
        check_cache_dp = online_reader_dp.check_hash(file_hash_dict, "md5", rewind=False)
        it = iter(check_cache_dp)
        path, stream = next(it)
        self.assertEqual(expected_license_file_name, os.path.basename(path))
        self.assertTrue(io.BufferedReader, type(stream))

        # Functional Test: works with multiple URLs of different sources
        online_reader_dp = OnlineReader(IterableWrapper([license_file_url, amazon_review_url]))
        check_cache_dp = online_reader_dp.check_hash(file_hash_dict, "md5", rewind=False)
        it = iter(check_cache_dp)
        for expected_file_name, (path, stream) in zip([expected_license_file_name, expected_amazon_file_name], it):
            self.assertEqual(expected_file_name, os.path.basename(path))
            self.assertTrue(io.BufferedReader, type(stream))

        # __len__ Test: returns the length of source DataPipe
        self.assertEqual(2, len(online_reader_dp))


if __name__ == "__main__":
    unittest.main()
