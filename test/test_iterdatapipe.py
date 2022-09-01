# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import itertools
import unittest
import warnings

from collections import defaultdict
from typing import Dict

import expecttest
import torch.utils.data.datapipes.iter

import torchdata

from _utils._common_utils_for_test import IDP_NoLen, reset_after_n_next_calls
from torchdata.datapipes.iter import (
    BucketBatcher,
    Cycler,
    Header,
    InBatchShuffler,
    IndexAdder,
    InMemoryCacheHolder,
    IterableWrapper,
    IterDataPipe,
    IterKeyZipper,
    LineReader,
    MapKeyZipper,
    MaxTokenBucketizer,
    ParagraphAggregator,
    RenameKeys,
    Rows2Columnar,
    SampleMultiplexer,
    UnZipper,
)
from torchdata.datapipes.map import MapDataPipe, SequenceWrapper


def test_torchdata_pytorch_consistency() -> None:
    def extract_datapipe_names(module):
        return {
            name
            for name, dp_type in module.__dict__.items()
            if not name.startswith("_") and isinstance(dp_type, type) and issubclass(dp_type, IterDataPipe)
        }

    pytorch_datapipes = extract_datapipe_names(torch.utils.data.datapipes.iter)
    torchdata_datapipes = extract_datapipe_names(torchdata.datapipes.iter)

    missing_datapipes = pytorch_datapipes - torchdata_datapipes
    deprecated_datapipes = {"FileLoader"}
    for dp in deprecated_datapipes:
        if dp in missing_datapipes:
            missing_datapipes.remove("FileLoader")

    if any(missing_datapipes):
        msg = (
            "The following datapipes are exposed under `torch.utils.data.datapipes.iter`, "
            "but not under `torchdata.datapipes.iter`:\n"
        )
        raise AssertionError(msg + "\n".join(sorted(missing_datapipes)))


class TestIterDataPipe(expecttest.TestCase):
    def test_in_memory_cache_holder_iterdatapipe(self) -> None:
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

        # TODO(122): Figure out a way to consistently test caching when size is in megabytes

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

    def test_iter_key_zipper_iterdatapipe(self) -> None:

        source_dp = IterableWrapper(range(10))
        ref_dp = IterableWrapper(range(20))

        # Functional Test: Output should be a zip list of tuple
        zip_dp = source_dp.zip_with_iter(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )
        self.assertEqual([(i, i) for i in range(10)], list(zip_dp))

        # Functional Test: keep_key=True, and key should show up as the first element
        zip_dp_w_key = source_dp.zip_with_iter(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=True, buffer_size=10
        )
        self.assertEqual([(i, (i, i)) for i in range(10)], list(zip_dp_w_key))

        # Functional Test: using a different merge function
        def merge_to_string(item1, item2):
            return f"{item1},{item2}"

        zip_dp_w_str_merge = source_dp.zip_with_iter(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, buffer_size=10, merge_fn=merge_to_string
        )
        self.assertEqual([f"{i},{i}" for i in range(10)], list(zip_dp_w_str_merge))

        # Functional Test: using a different merge function and keep_key=True
        zip_dp_w_key_str_merge = source_dp.zip_with_iter(
            ref_datapipe=ref_dp,
            key_fn=lambda x: x,
            ref_key_fn=lambda x: x,
            keep_key=True,
            buffer_size=10,
            merge_fn=merge_to_string,
        )
        self.assertEqual([(i, f"{i},{i}") for i in range(10)], list(zip_dp_w_key_str_merge))

        # Functional Test: testing nested zipping
        zip_dp = source_dp.zip_with_iter(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )

        # Without a custom merge function, there will be nested tuples
        zip_dp2 = zip_dp.zip_with_iter(
            ref_datapipe=ref_dp, key_fn=lambda x: x[0], ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )
        self.assertEqual([((i, i), i) for i in range(10)], list(zip_dp2))

        # With a custom merge function, nesting can be prevented
        zip_dp2_w_merge = zip_dp.zip_with_iter(
            ref_datapipe=ref_dp,
            key_fn=lambda x: x[0],
            ref_key_fn=lambda x: x,
            keep_key=False,
            buffer_size=100,
            merge_fn=lambda x, y: list(x) + [y],
        )
        self.assertEqual([[i, i, i] for i in range(10)], list(zip_dp2_w_merge))

        # Functional Test: element is in source but missing in reference
        ref_dp_missing = IterableWrapper(range(1, 10))
        zip_dp = source_dp.zip_with_iter(
            ref_datapipe=ref_dp_missing, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )
        with self.assertRaisesRegex(BufferError, r"No matching key can be found"):
            list(zip_dp)

        # Functional Test: Buffer is not large enough, hence, element can't be found and raises error
        ref_dp_end = IterableWrapper(list(range(1, 10)) + [0])
        zip_dp = source_dp.zip_with_iter(
            ref_datapipe=ref_dp_end, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=5
        )
        it = iter(zip_dp)
        with warnings.catch_warnings(record=True) as wa:
            # In order to find '0' at the end, the buffer is filled, hence the warning
            # and ref_dp is fully traversed
            self.assertEqual(
                (
                    0,
                    0,
                ),
                next(it),
            )
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Buffer reaches the upper limit")
        with self.assertRaisesRegex(BufferError, r"No matching key can be found"):
            # '1' cannot be find because the value was thrown out when buffer was filled
            next(it)

        # Functional Test: Buffer is just big enough
        zip_dp = source_dp.zip_with_iter(
            ref_datapipe=ref_dp_end, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=10
        )
        self.assertEqual([(i, i) for i in range(10)], list(zip_dp))

        # Reset Test: reset the DataPipe after reading part of it
        zip_dp = IterKeyZipper(
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

    def test_map_key_zipper_datapipe(self) -> None:
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

        result_dp = MapKeyZipper(source_dp, map_dp, odd_even_bug)
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

    def test_cycler_iterdatapipe(self) -> None:
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

    def test_header_iterdatapipe(self) -> None:
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

        # __len__ Test: returns the length of source when it is less than the limit
        header_dp = source_dp.header(30)
        self.assertEqual(20, len(header_dp))

        # __len__ Test: returns limit if source doesn't have length
        source_dp_NoLen = IDP_NoLen(list(range(20)))
        header_dp = source_dp_NoLen.header(30)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(30, len(header_dp))
            self.assertEqual(len(wa), 1)
            self.assertRegex(
                str(wa[0].message), r"length of this HeaderIterDataPipe is inferred to be equal to its limit"
            )

        # __len__ Test: returns limit if source doesn't have length, but it has been iterated through once
        for _ in header_dp:
            pass
        self.assertEqual(20, len(header_dp))

    def test_enumerator_iterdatapipe(self) -> None:
        letters = "abcde"
        source_dp = IterableWrapper(letters)
        enum_dp = source_dp.enumerate()

        # Functional Test: ensure that the correct index value is added to each element (tuple)
        self.assertEqual([(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")], list(enum_dp))

        # Functional Test: start index from non-zero
        enum_dp = source_dp.enumerate(starting_index=10)
        self.assertEqual([(10, "a"), (11, "b"), (12, "c"), (13, "d"), (14, "e")], list(enum_dp))

        # Reset Test:
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(enum_dp, n_elements_before_reset)
        self.assertEqual([(10, "a"), (11, "b")], res_before_reset)
        self.assertEqual([(10, "a"), (11, "b"), (12, "c"), (13, "d"), (14, "e")], res_after_reset)

        # __len__ Test: returns length of source DataPipe
        self.assertEqual(5, len(enum_dp))

    def test_index_adder_iterdatapipe(self) -> None:
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

    def test_line_reader_iterdatapipe(self) -> None:
        text1 = "Line1\nLine2"
        text2 = "Line2,1\r\nLine2,2\r\nLine2,3"

        # Functional Test: read lines correctly
        source_dp = IterableWrapper([("file1", io.StringIO(text1)), ("file2", io.StringIO(text2))])
        line_reader_dp = source_dp.readlines()
        expected_result = [("file1", line) for line in text1.splitlines()] + [
            ("file2", line) for line in text2.splitlines()
        ]
        self.assertEqual(expected_result, list(line_reader_dp))

        # Functional Test: strip new lines for bytes
        source_dp = IterableWrapper(
            [("file1", io.BytesIO(text1.encode("utf-8"))), ("file2", io.BytesIO(text2.encode("utf-8")))]
        )
        line_reader_dp = source_dp.readlines()
        expected_result_bytes = [("file1", line.encode("utf-8")) for line in text1.splitlines()] + [
            ("file2", line.encode("utf-8")) for line in text2.splitlines()
        ]
        self.assertEqual(expected_result_bytes, list(line_reader_dp))

        # Functional Test: do not strip new lines
        source_dp = IterableWrapper([("file1", io.StringIO(text1)), ("file2", io.StringIO(text2))])
        line_reader_dp = source_dp.readlines(strip_newline=False)
        expected_result = [
            ("file1", "Line1\n"),
            ("file1", "Line2"),
            ("file2", "Line2,1\r\n"),
            ("file2", "Line2,2\r\n"),
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

    def test_paragraph_aggregator_iterdatapipe(self) -> None:
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

    def test_rows_to_columnar_iterdatapipe(self) -> None:
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

    def test_sample_multiplexer_iterdatapipe(self) -> None:
        # Functional Test: yields all values from the sources
        source_dp1 = IterableWrapper([0] * 10)
        source_dp2 = IterableWrapper([1] * 10)
        d: Dict[IterDataPipe, float] = {source_dp1: 99999999, source_dp2: 0.0000001}
        sample_mul_dp = SampleMultiplexer(pipes_to_weights_dict=d, seed=0)
        result = list(sample_mul_dp)
        self.assertEqual([0] * 10 + [1] * 10, result)

        # Functional Test: raises error for empty dict
        with self.assertRaisesRegex(ValueError, "Empty dictionary"):
            SampleMultiplexer(pipes_to_weights_dict={}, seed=0)  # type: ignore[arg-type]

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

    def test_in_batch_shuffler_iterdatapipe(self) -> None:
        source_dp = IterableWrapper(range(10)).batch(3)

        # Functional Test: drop last reduces length
        filtered_dp = source_dp.in_batch_shuffle()
        for ret_batch, exp_batch in zip(filtered_dp, source_dp):
            ret_batch.sort()
            self.assertEqual(ret_batch, exp_batch)

        # Reset Test:
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(filtered_dp, n_elements_before_reset)
        self.assertEqual(n_elements_before_reset, len(res_before_reset))
        for ret_batch, exp_batch in zip(res_before_reset, source_dp):
            ret_batch.sort()
            self.assertEqual(ret_batch, exp_batch)
        for ret_batch, exp_batch in zip(res_after_reset, source_dp):
            ret_batch.sort()
            self.assertEqual(ret_batch, exp_batch)

        # __len__ Test: returns the number of batches
        self.assertEqual(4, len(filtered_dp))

    def test_bucket_batcher_iterdatapipe(self) -> None:
        source_dp = IterableWrapper(range(10))

        # Functional Test: drop last reduces length
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=True, batch_num=100, bucket_num=1, use_in_batch_shuffle=True
        )
        self.assertEqual(9, len(list(batch_dp.unbatch())))

        # Functional Test: drop last is False preserves length
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=False, batch_num=100, bucket_num=1, use_in_batch_shuffle=False
        )
        self.assertEqual(10, len(list(batch_dp.unbatch())))

        def _return_self(x):
            return x

        # Functional Test: using sort_key, with in_batch_shuffle
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=True, batch_num=100, bucket_num=1, use_in_batch_shuffle=True, sort_key=_return_self
        )
        # bucket_num = 1 means there will be no shuffling if a sort key is given
        self.assertEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]], list(batch_dp))
        self.assertEqual(9, len(list(batch_dp.unbatch())))

        # Functional Test: using sort_key, without use_in_batch_shuffle
        batch_dp = source_dp.bucketbatch(
            batch_size=3, drop_last=True, batch_num=100, bucket_num=2, use_in_batch_shuffle=False, sort_key=_return_self
        )
        self.assertEqual(9, len(list(batch_dp.unbatch())))

        # Reset Test:
        batch_dp = BucketBatcher(
            source_dp,
            batch_size=3,
            drop_last=True,
            batch_num=100,
            bucket_num=2,
            use_in_batch_shuffle=False,
            sort_key=_return_self,
        )
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        self.assertEqual(n_elements_before_reset, len(res_before_reset))
        self.assertEqual(6, len([item for batch in res_before_reset for item in batch]))
        self.assertEqual(3, len(res_after_reset))
        self.assertEqual(9, len([item for batch in res_after_reset for item in batch]))

        # __len__ Test: returns the number of batches
        with self.assertRaises(TypeError):
            len(batch_dp)

    def test_max_token_bucketizer_iterdatapipe(self) -> None:
        source_data = ["1" * d for d in range(1, 6)] + ["2" * d for d in range(1, 6)]
        source_dp = IterableWrapper(source_data)

        # Functional Test: Invalid arguments
        with self.assertRaisesRegex(ValueError, "``min_len`` should be larger than 0"):
            source_dp.max_token_bucketize(max_token_count=2, min_len=-1)

        with self.assertRaisesRegex(ValueError, "``min_len`` should be larger than 0"):
            source_dp.max_token_bucketize(max_token_count=2, min_len=3, max_len=2)

        with self.assertRaises(ValueError, msg="``max_token_count`` must be equal to or greater than ``max_len``."):
            source_dp.max_token_bucketize(max_token_count=2, max_len=3)

        # Functional Test: Filter out min_len
        batch_dp = source_dp.max_token_bucketize(max_token_count=5, min_len=2, buffer_size=10)
        exp_batch = [["11", "22"], ["111"], ["222"], ["1111"], ["2222"], ["11111"], ["22222"]]
        self.assertEqual(list(batch_dp), exp_batch)

        # Functional Test: Filter out max_len
        batch_dp = source_dp.max_token_bucketize(max_token_count=5, max_len=4, buffer_size=10)
        exp_batch = [["1", "2", "11"], ["22", "111"], ["222"], ["1111"], ["2222"]]
        self.assertEqual(list(batch_dp), exp_batch)

        def _custom_len_fn(token):
            return len(token) + 1

        # Functional Test: Custom length function
        batch_dp = source_dp.max_token_bucketize(max_token_count=7, len_fn=_custom_len_fn, buffer_size=10)
        exp_batch = [["1", "2", "11"], ["22", "111"], ["222"], ["1111"], ["2222"], ["11111"], ["22222"]]
        self.assertEqual(list(batch_dp), exp_batch)

        # Functional Test: Small buffer
        batch_dp = source_dp.max_token_bucketize(max_token_count=10, buffer_size=4)
        exp_batch = [["1", "11", "2", "22", "111"], ["222", "1111"], ["2222", "11111"], ["22222"]]
        self.assertEqual(list(batch_dp), exp_batch)

        # Reset Test:
        batch_dp = MaxTokenBucketizer(source_dp, max_token_count=5, buffer_size=10)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        exp_before_reset = [["1", "2", "11"], ["22", "111"]]
        exp_after_reset = [["1", "2", "11"], ["22", "111"], ["222"], ["1111"], ["2222"], ["11111"], ["22222"]]
        self.assertEqual(res_before_reset, exp_before_reset)
        self.assertEqual(res_after_reset, exp_after_reset)

        # __len__ Test: returns the number of batches
        with self.assertRaises(TypeError):
            len(batch_dp)

    def test_map_batches_iterdatapipe(self):
        source_dp = IterableWrapper(list(range(20)))

        def fn(batch):
            return [d + 1 for d in batch]

        batch_mapped_dp = source_dp.map_batches(fn, batch_size=9)
        expected_list = list(range(1, 21))
        self.assertEqual(expected_list, list(batch_mapped_dp))

        # Reset Test: reset the DataPipe after reading part of it
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(batch_mapped_dp, n_elements_before_reset)

        self.assertEqual(expected_list[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_list, res_after_reset)

        # Functional Test: Different sizes between input and output
        def fn_less(batch):
            return [batch[idx] // 2 for idx in range(0, len(batch), 2)]

        less_batch_mapped_dp = source_dp.map_batches(fn_less, batch_size=8)
        self.assertEqual(list(range(10)), list(less_batch_mapped_dp))

        # Functional Test: Specify input_col
        source_dp = IterableWrapper([(d - 1, d, d + 1) for d in range(20)])

        batch_mapped_input_1_dp = source_dp.map_batches(fn, batch_size=9, input_col=0)
        self.assertEqual(list(range(20)), list(batch_mapped_input_1_dp))

        def fn_2_cols(batch):
            return [(d1, d2 - 1) for d1, d2 in batch]

        batch_mapped_input_2_dp = source_dp.map_batches(fn_2_cols, batch_size=9, input_col=[1, 2])
        self.assertEqual([(d, d) for d in range(20)], list(batch_mapped_input_2_dp))

        # __len__ Test: length should be determined by ``fn`` which we can't know
        with self.assertRaisesRegex(TypeError, "length relies on the output of its function."):
            len(batch_mapped_dp)

    def test_flatmap_iterdatapipe(self):
        source_dp = IterableWrapper(list(range(20)))

        def fn(e):
            return [e, e * 10]

        flatmapped_dp = source_dp.flatmap(fn)
        expected_list = list(itertools.chain(*[(e, e * 10) for e in source_dp]))
        self.assertEqual(expected_list, list(flatmapped_dp))

        # Funtional Test: Specify input_col
        tuple_source_dp = IterableWrapper([(d - 1, d, d + 1) for d in range(20)])

        # Single input_col
        input_col_1_dp = tuple_source_dp.flatmap(fn, input_col=1)
        self.assertEqual(expected_list, list(input_col_1_dp))

        # Multiple input_col
        def mul_fn(a, b):
            return [a - b, b - a]

        input_col_2_dp = tuple_source_dp.flatmap(mul_fn, input_col=(0, 2))
        self.assertEqual(list(itertools.chain(*[(-2, 2) for _ in range(20)])), list(input_col_2_dp))

        # Reset Test: reset the DataPipe after reading part of it
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(flatmapped_dp, n_elements_before_reset)

        self.assertEqual(expected_list[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_list, res_after_reset)

        # __len__ Test: length should be len(source_dp)*len(fn->out_shape) which we can't know
        with self.assertRaisesRegex(TypeError, "length relies on the output of its function."):
            len(flatmapped_dp)

    def test_unzipper_iterdatapipe(self):
        source_dp = IterableWrapper([(i, i + 10, i + 20) for i in range(10)])

        # Functional Test: unzips each sequence, no `sequence_length` specified
        dp1, dp2, dp3 = UnZipper(source_dp, sequence_length=3)
        self.assertEqual(list(range(10)), list(dp1))
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        # Functional Test: unzips each sequence, with `sequence_length` specified
        dp1, dp2, dp3 = source_dp.unzip(sequence_length=3)
        self.assertEqual(list(range(10)), list(dp1))
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        # Functional Test: skipping over specified values
        dp2, dp3 = source_dp.unzip(sequence_length=3, columns_to_skip=[0])
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        (dp2,) = source_dp.unzip(sequence_length=3, columns_to_skip=[0, 2])
        self.assertEqual(list(range(10, 20)), list(dp2))

        source_dp = IterableWrapper([(i, i + 10, i + 20, i + 30) for i in range(10)])
        dp2, dp3 = source_dp.unzip(sequence_length=4, columns_to_skip=[0, 3])
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        # Functional Test: one child DataPipe yields all value first, but buffer_size = 5 being too small, raises error
        source_dp = IterableWrapper([(i, i + 10) for i in range(10)])
        dp1, dp2 = source_dp.unzip(sequence_length=2, buffer_size=5)
        it1 = iter(dp1)
        for _ in range(5):
            next(it1)
        with self.assertRaises(BufferError):
            next(it1)
        with self.assertRaises(BufferError):
            list(dp2)

        # Reset Test: reset the DataPipe after reading part of it
        dp1, dp2 = source_dp.unzip(sequence_length=2)
        i1, i2 = iter(dp1), iter(dp2)
        output2 = []
        for i, n2 in enumerate(i2):
            output2.append(n2)
            if i == 4:
                i1 = iter(dp1)  # Doesn't reset because i1 hasn't been read
        self.assertEqual(list(range(10, 20)), output2)

        # Reset Test: DataPipe reset when some of it have been read
        dp1, dp2 = source_dp.unzip(sequence_length=2)
        i1, i2 = iter(dp1), iter(dp2)
        output1, output2 = [], []
        for i, (n1, n2) in enumerate(zip(i1, i2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    i1 = iter(dp1)  # Reset both all child DataPipe
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
        self.assertEqual(list(range(5)) + list(range(10)), output1)
        self.assertEqual(list(range(10, 15)) + list(range(10, 20)), output2)

        # Reset Test: DataPipe reset, even when some other child DataPipes are not read
        source_dp = IterableWrapper([(i, i + 10, i + 20) for i in range(10)])
        dp1, dp2, dp3 = source_dp.unzip(sequence_length=3)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10, 20)), output2)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(list(range(10)), list(dp1))  # Resets even though dp3 has not been read
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
        output3 = []
        for i, n3 in enumerate(dp3):
            output3.append(n3)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    output1 = list(dp1)  # Resets even though dp3 is only partially read
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
                self.assertEqual(list(range(20, 25)), output3)
                self.assertEqual(list(range(10)), output1)
                break
        self.assertEqual(list(range(20, 30)), list(dp3))  # dp3 has to read from the start again

        # __len__ Test: Each DataPipe inherits the source datapipe's length
        dp1, dp2, dp3 = source_dp.unzip(sequence_length=3)
        self.assertEqual(len(source_dp), len(dp1))
        self.assertEqual(len(source_dp), len(dp2))
        self.assertEqual(len(source_dp), len(dp3))

    def test_itertomap_mapdatapipe(self):
        # Functional Test with None key_value_fn
        values = list(range(10))
        keys = ["k" + str(i) for i in range(10)]
        source_dp = IterableWrapper(list(zip(keys, values)))

        map_dp = source_dp.to_map_datapipe()
        self.assertTrue(isinstance(map_dp, MapDataPipe))

        # Lazy loading
        self.assertTrue(map_dp._map is None)

        # __len__ Test: Each DataPipe inherits the source datapipe's length
        self.assertEqual(len(map_dp), 10)

        # Functional Test
        self.assertEqual(list(range(10)), [map_dp["k" + str(idx)] for idx in range(10)])
        self.assertFalse(map_dp._map is None)

        source_dp = IterableWrapper(range(10))

        # TypeError test for invalid data type
        map_dp = source_dp.to_map_datapipe()
        with self.assertRaisesRegex(TypeError, "Cannot convert dictionary update element"):
            _ = list(map_dp)

        # ValueError test for wrong length
        map_dp = source_dp.to_map_datapipe(lambda d: (d,))
        with self.assertRaisesRegex(ValueError, "dictionary update sequence element has length"):
            _ = list(map_dp)

        # Functional Test with key_value_fn
        map_dp = source_dp.to_map_datapipe(lambda d: ("k" + str(d), d + 1))
        self.assertEqual(list(range(1, 11)), [map_dp["k" + str(idx)] for idx in range(10)])
        self.assertFalse(map_dp._map is None)

        # No __len__ from prior DataPipe
        no_len_dp = source_dp.filter(lambda x: x % 2 == 0)
        map_dp = no_len_dp.to_map_datapipe(lambda x: (x, x + 2))
        with warnings.catch_warnings(record=True) as wa:
            length = len(map_dp)
            self.assertEqual(length, 5)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Data from prior DataPipe")

        # Duplicate Key Test
        dup_map_dp = source_dp.to_map_datapipe(lambda x: (x % 1, x))
        with warnings.catch_warnings(record=True) as wa:
            dup_map_dp._load_map()
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Found duplicate key")

    def test_mux_longest_iterdatapipe(self):

        # Functional Test: Elements are yielded one at a time from each DataPipe, until they are all exhausted
        input_dp1 = IterableWrapper(range(4))
        input_dp2 = IterableWrapper(range(4, 8))
        input_dp3 = IterableWrapper(range(8, 12))
        output_dp = input_dp1.mux_longest(input_dp2, input_dp3)
        expected_output = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Uneven input Data Pipes
        input_dp1 = IterableWrapper([1, 2, 3, 4])
        input_dp2 = IterableWrapper([10])
        input_dp3 = IterableWrapper([100, 200, 300])
        output_dp = input_dp1.mux_longest(input_dp2, input_dp3)
        expected_output = [1, 10, 100, 2, 200, 3, 300, 4]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Empty Data Pipe
        input_dp1 = IterableWrapper([0, 1, 2, 3])
        input_dp2 = IterableWrapper([])
        output_dp = input_dp1.mux_longest(input_dp2)
        self.assertEqual(len(input_dp1), len(output_dp))
        self.assertEqual(list(input_dp1), list(output_dp))

        # __len__ Test: raises TypeError when __len__ is called and an input doesn't have __len__
        input_dp1 = IterableWrapper(range(10))
        input_dp_no_len = IDP_NoLen(range(10))
        output_dp = input_dp1.mux_longest(input_dp_no_len)
        with self.assertRaises(TypeError):
            len(output_dp)

    def test_renamer(self):

        # Functional Test: verify that renaming by patterns yields correct output
        stage1 = IterableWrapper([
            {"1.txt": "1", "1.bin": "1b"},
            {"2.txt": "2", "2.bin": "2b"},
        ])
        stage2 = stage1.rename_keys(t="*.txt", b="*.bin")
        output = list(iter(stage2))
        assert len(output) == 2
        assert set(output[0].keys()) == set(["t", "b"])

    def test_zip_longest_iterdatapipe(self):

        # Functional Test: raises TypeError when an input is not of type `IterDataPipe`
        with self.assertRaises(TypeError):
            input_dp1 = IterableWrapper(range(10))
            input_no_dp = list(range(10))
            output_dp = input_dp1.zip_longest(input_no_dp)  # type: ignore[arg-type]

        # Functional Test: raises TypeError when an input does not have valid length
        input_dp1 = IterableWrapper(range(10))
        input_dp_no_len = IDP_NoLen(range(5))
        output_dp = input_dp1.zip_longest(input_dp_no_len)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(output_dp)

        # Functional Test: zips the results properly even when lengths are different
        # (zips to the longest, filling missing values with default value None.)
        input_dp1 = IterableWrapper(range(10))
        input_dp2 = IterableWrapper(range(5))
        output_dp = input_dp1.zip_longest(input_dp2)
        exp = [(i, i) for i in range(5)] + [(i, None) for i in range(5, 10)]
        self.assertEqual(list(output_dp), exp)

        # Functional Test: zips the results properly even when lengths are different
        # (zips to the longest, filling missing values with user input)
        input_dp1 = IterableWrapper(range(10))
        input_dp2 = IterableWrapper(range(5))
        output_dp = input_dp1.zip_longest(input_dp2, fill_value=-1)
        exp = [(i, i) for i in range(5)] + [(i, -1) for i in range(5, 10)]
        self.assertEqual(list(output_dp), exp)

        # __len__ Test: length matches the length of the shortest input
        self.assertEqual(len(output_dp), 10)


if __name__ == "__main__":
    unittest.main()
