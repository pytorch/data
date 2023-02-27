# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import itertools
import pickle
import unittest
import warnings

from collections import defaultdict
from typing import Dict

import expecttest
import torch

import torchdata

from _utils._common_utils_for_test import IDP_NoLen, reset_after_n_next_calls

from torch.utils.data.datapipes.utils.snapshot import _simple_graph_snapshot_restoration
from torchdata.datapipes.iter import (
    BucketBatcher,
    Cycler,
    Header,
    IndexAdder,
    InMemoryCacheHolder,
    IterableWrapper,
    IterDataPipe,
    IterKeyZipper,
    LineReader,
    MapKeyZipper,
    MaxTokenBucketizer,
    ParagraphAggregator,
    Repeater,
    Rows2Columnar,
    SampleMultiplexer,
    ShardExpander,
    UnZipper,
)
from torchdata.datapipes.map import MapDataPipe, SequenceWrapper

skipIfNoCUDA = unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")


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


def _convert_to_tensor(data):
    if isinstance(data, dict):
        return {k: _convert_to_tensor(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_to_tensor(v) for v in data]
    return torch.tensor(data)


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
        ref_dp2 = IterableWrapper(range(20))

        # Functional Test: Output should be a zip list of tuple
        zip_dp = source_dp.zip_with_iter(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )
        self.assertEqual([(i, i) for i in range(10)], list(zip_dp))

        # Functional Test: keep_key=True, and key should show up as the first element
        zip_dp_w_key = source_dp.zip_with_iter(
            ref_datapipe=ref_dp2, key_fn=lambda x: x, ref_key_fn=lambda x: x, keep_key=True, buffer_size=10
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
            ref_datapipe=ref_dp2, key_fn=lambda x: x[0], ref_key_fn=lambda x: x, keep_key=False, buffer_size=100
        )
        self.assertEqual([((i, i), i) for i in range(10)], list(zip_dp2))

        # With a custom merge function, nesting can be prevented
        zip_dp2_w_merge = zip_dp.zip_with_iter(
            ref_datapipe=ref_dp2,
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

        # Functional test: ensure that keep_key option works
        result_dp = source_dp.zip_with_map(map_dp, odd_even, keep_key=True)
        expected_res_keep_key = [(key, (i, odd_even_string(i))) for i, key in zip(range(10), [0, 1] * 5)]
        self.assertEqual(expected_res_keep_key, list(result_dp))

        # Reset Test:
        n_elements_before_reset = 4
        result_dp = source_dp.zip_with_map(map_dp, odd_even)
        res_before_reset, res_after_reset = reset_after_n_next_calls(result_dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

        # __len__ Test: returns the length of source DataPipe
        result_dp = source_dp.zip_with_map(map_dp, odd_even)
        self.assertEqual(len(source_dp), len(result_dp))

    def test_prefetcher_iterdatapipe(self) -> None:
        source_dp = IterableWrapper(range(50000))
        prefetched_dp = source_dp.prefetch(10)
        # check if early termination resets child thread properly
        for _, _ in zip(range(100), prefetched_dp):
            pass
        expected = list(source_dp)
        actual = list(prefetched_dp)
        self.assertEqual(expected, actual)

    def test_repeater_iterdatapipe(self) -> None:
        import itertools

        source_dp = IterableWrapper(range(5))

        # Functional Test: repeat for correct number of times
        repeater_dp = source_dp.repeat(3)
        self.assertEqual(
            list(itertools.chain.from_iterable(itertools.repeat(x, 3) for x in range(5))), list(repeater_dp)
        )

        # Functional Test: `times` must be > 1
        with self.assertRaisesRegex(ValueError, "The number of repetition must be > 1"):
            source_dp.repeat(1)

        # Reset Test:
        repeater_dp = Repeater(source_dp, times=2)
        n_elements_before_reset = 4
        res_before_reset, res_after_reset = reset_after_n_next_calls(repeater_dp, n_elements_before_reset)
        self.assertEqual([0, 0, 1, 1], res_before_reset)
        self.assertEqual(list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in range(5))), res_after_reset)

        # __len__ Test: returns correct length
        self.assertEqual(10, len(repeater_dp))

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

        # Functional Test: ensure the source is not modified if limit is set to None
        source_dp = IterableWrapper(range(5))
        header_dp = source_dp.header(None)
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

        # __len__ Test: returns the length of source when limit is set to None
        header_dp = source_dp.header(None)
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

        # __len__ Test: raises TypeError if source doesn't have length and limit is set to None
        header_dp = source_dp_NoLen.header(None)
        with self.assertRaisesRegex(TypeError, "The length of this HeaderIterDataPipe cannot be determined."):
            len(header_dp)

        # __len__ Test: returns limit if source doesn't have length, even when it has been iterated through once
        header_dp = source_dp_NoLen.header(30)
        for _ in header_dp:
            pass
        self.assertEqual(30, len(header_dp))

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

    def test_in_batch_shuffler_iterdatapipe(self):
        input_dp = IterableWrapper(list(range(23))).batch(3)
        expected = list(input_dp)

        # Functional Test: No seed
        shuffler_dp = input_dp.in_batch_shuffle()
        for exp, res in zip(expected, shuffler_dp):
            self.assertEqual(sorted(res), exp)

        # Functional Test: With global seed
        torch.manual_seed(123)
        res = list(shuffler_dp)
        torch.manual_seed(123)
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: Set seed
        shuffler_dp = input_dp.in_batch_shuffle().set_seed(123)
        res = list(shuffler_dp)
        shuffler_dp.set_seed(123)
        self.assertEqual(list(shuffler_dp), res)

        # Functional Test: deactivate shuffling via set_shuffle
        unshuffled_dp = shuffler_dp.set_shuffle(False)
        self.assertEqual(list(unshuffled_dp), expected)

        # Reset Test:
        shuffler_dp = input_dp.in_batch_shuffle()
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(shuffler_dp, n_elements_before_reset)
        self.assertEqual(5, len(res_before_reset))
        for exp, res in zip(expected, res_before_reset):
            self.assertEqual(sorted(res), exp)
        for exp, res in zip(expected, res_after_reset):
            self.assertEqual(sorted(res), exp)

        # __len__ Test: returns the length of the input DataPipe
        shuffler_dp = input_dp.in_batch_shuffle()
        self.assertEqual(8, len(shuffler_dp))

        # Serialization Test
        from torch.utils.data.datapipes._hook_iterator import _SnapshotState

        shuffler_dp = input_dp.in_batch_shuffle()
        it = iter(shuffler_dp)
        for _ in range(2):
            next(it)
        shuffler_dp_copy = pickle.loads(pickle.dumps(shuffler_dp))
        _simple_graph_snapshot_restoration(shuffler_dp_copy.datapipe, shuffler_dp.datapipe._number_of_samples_yielded)

        exp = list(it)
        shuffler_dp_copy._snapshot_state = _SnapshotState.Restored
        self.assertEqual(exp, list(shuffler_dp_copy))

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

        def _validate_batch_size(res, exp_batch_len, len_fn=lambda d: len(d)):
            self.assertEqual(len(res), len(exp_batch_len))

            for batch, exp_token_lens in zip(res, exp_batch_len):
                self.assertEqual(len(batch), len(exp_token_lens))
                for token, exp_token_len in zip(batch, exp_token_lens):
                    self.assertEqual(len_fn(token), exp_token_len)

        # Functional Test: Filter out min_len
        batch_dp = source_dp.max_token_bucketize(max_token_count=5, min_len=2, buffer_size=10)
        exp_batch_len = [(2, 2), (3,), (3,), (4,), (4,), (5,), (5,)]
        _validate_batch_size(list(batch_dp), exp_batch_len)

        # Functional Test: Filter out max_len
        batch_dp = source_dp.max_token_bucketize(max_token_count=5, max_len=4, buffer_size=10)
        exp_batch_len = [(1, 1, 2), (2, 3), (3,), (4,), (4,)]
        _validate_batch_size(list(batch_dp), exp_batch_len)

        def _custom_len_fn(token):
            return len(token) + 1

        # Functional Test: Custom length function
        batch_dp = source_dp.max_token_bucketize(max_token_count=7, len_fn=_custom_len_fn, buffer_size=10)
        exp_batch_len = [(1, 1, 2), (2, 3), (3,), (4,), (4,), (5,), (5,)]
        _validate_batch_size(list(batch_dp), exp_batch_len)

        # Functional Test: Small buffer
        batch_dp = source_dp.max_token_bucketize(max_token_count=10, buffer_size=4)
        exp_batch_len = [(1, 2, 1, 2, 3), (3, 4), (4, 5), (5,)]
        _validate_batch_size(list(batch_dp), exp_batch_len)

        # Reset Test:
        batch_dp = MaxTokenBucketizer(source_dp, max_token_count=5, buffer_size=10)
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(batch_dp, n_elements_before_reset)
        exp_batch_len_before_reset = [(1, 1, 2), (2, 3)]
        exp_batch_len_after_reset = [(1, 1, 2), (2, 3), (3,), (4,), (4,), (5,), (5,)]
        _validate_batch_size(res_before_reset, exp_batch_len_before_reset)
        _validate_batch_size(res_after_reset, exp_batch_len_after_reset)

        # Functional test: Padded tokens exceeding max_token_count
        source_data = ["111", "1111", "11111"]  # 3, 4, 5
        source_dp = IterableWrapper(source_data)
        batch_dp = source_dp.max_token_bucketize(max_token_count=7)
        exp_batch_len = [(3, 4), (5,)]
        _validate_batch_size(list(batch_dp), exp_batch_len)

        # Functional test: Padded tokens not exceeding max_token_count
        source_data = ["111", "111", "111", "1111"]  # 3, 3, 3, 4
        source_dp = IterableWrapper(source_data)
        batch_dp = source_dp.max_token_bucketize(max_token_count=7, include_padding=True)
        exp_batch_len = [(3, 3), (3,), (4,)]
        _validate_batch_size(list(batch_dp), exp_batch_len)

        # Functional test: sample length exceeding max_token_count
        source_data = ["111"]
        source_dp = IterableWrapper(source_data)
        batch_dp = source_dp.max_token_bucketize(max_token_count=2)
        exp_batch = []
        self.assertEqual(list(batch_dp), exp_batch)

        # Functional test: incomparable data for heapq
        def _custom_len_fn(data):
            return data["len"]

        source_data = [{"len": 1}, {"len": 2}, {"len": 1}, {"len": 3}, {"len": 1}]
        source_dp = IterableWrapper(source_data)
        batch_dp = source_dp.max_token_bucketize(max_token_count=3, len_fn=_custom_len_fn)
        exp_batch_len = [(1, 1, 1), (2,), (3,)]
        _validate_batch_size(list(batch_dp), exp_batch_len, len_fn=_custom_len_fn)

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

        # flatmap with no fn specified
        default_dp = tuple_source_dp.flatmap()
        self.assertEqual(list(itertools.chain(*[(n - 1, n, n + 1) for n in range(20)])), list(default_dp))

        # flatmap with no fn specified, multiple input_col
        default_dp = tuple_source_dp.flatmap(input_col=(0, 2))
        self.assertEqual(list(itertools.chain(*[(n - 1, n + 1) for n in range(20)])), list(default_dp))

        # flatmap with no fn specified, some special input
        tuple_source_dp = IterableWrapper([[1, 2, [3, 4]], [5, 6, [7, 8]]])
        default_dp = tuple_source_dp.flatmap(input_col=(0, 2))
        self.assertEqual([1, [3, 4], 5, [7, 8]], list(default_dp))

        # Reset Test: reset the DataPipe after reading part of it
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(flatmapped_dp, n_elements_before_reset)

        self.assertEqual(expected_list[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_list, res_after_reset)

        # __len__ Test: length should be len(source_dp)*len(fn->out_shape) which we can't know
        with self.assertRaisesRegex(TypeError, "length relies on the output of its function."):
            len(flatmapped_dp)

    def test_round_robin_demux_iterdatapipe(self):
        source_dp = IterableWrapper(list(range(23)))
        with self.assertRaisesRegex(ValueError, "Expected `num_instaces`"):
            _ = source_dp.round_robin_demux(0)

        # Funtional Test
        dp1, dp2, dp3 = source_dp.round_robin_demux(3)
        self.assertEqual(list(range(0, 23, 3)), list(dp1))
        self.assertEqual(list(range(1, 23, 3)), list(dp2))
        self.assertEqual(list(range(2, 23, 3)), list(dp3))

        # __len__ Test
        self.assertEqual(len(dp1), 8)
        self.assertEqual(len(dp2), 8)
        self.assertEqual(len(dp3), 7)

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

        (dp2,) = source_dp.unzip(sequence_length=3, columns_to_skip=[0, 2], buffer_size=0)
        self.assertEqual(list(range(10, 20)), list(dp2))

        source_dp = IterableWrapper([(i, i + 10, i + 20, i + 30) for i in range(10)])
        dp2, dp3 = source_dp.unzip(sequence_length=4, columns_to_skip=[0, 3])
        self.assertEqual(list(range(10, 20)), list(dp2))
        self.assertEqual(list(range(20, 30)), list(dp3))

        # Functional Test: one child DataPipe yields all value first, but buffer_size = 5 being too small, raises error
        source_dp = IterableWrapper([(i, i + 10) for i in range(10)])
        dp1, dp2 = source_dp.unzip(sequence_length=2, buffer_size=4)
        it1 = iter(dp1)
        for _ in range(4):
            next(it1)
        with self.assertRaises(BufferError):
            next(it1)
        with self.assertRaises(BufferError):
            list(dp2)

        dp1, dp2 = source_dp.unzip(sequence_length=2, buffer_size=4)
        with self.assertRaises(BufferError):
            list(dp2)

        # Reset Test: DataPipe resets when a new iterator is created, even if this datapipe hasn't been read
        dp1, dp2 = source_dp.unzip(sequence_length=2)
        _ = iter(dp1)
        output2 = []
        with self.assertRaisesRegex(RuntimeError, r"iterator has been invalidated"):
            for i, n2 in enumerate(dp2):
                output2.append(n2)
                if i == 4:
                    _ = iter(dp1)  # This will reset all child DataPipes
        self.assertEqual(list(range(10, 15)), output2)

        # Reset Test: DataPipe reset when some of it have been read
        dp1, dp2 = source_dp.unzip(sequence_length=2)
        output1, output2 = [], []
        for i, (n1, n2) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    _ = iter(dp1)  # Reset both all child DataPipe
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(str(wa[0].message), r"Some child DataPipes are not exhausted")
                break
        for n1, n2 in zip(dp1, dp2):
            output1.append(n1)
            output2.append(n2)
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

    def test_shard_expand(self):

        # Functional Test: ensure expansion generates the right outputs
        def testexpand(s):
            stage1 = IterableWrapper([s])
            stage2 = ShardExpander(stage1)
            return list(iter(stage2))

        def myexpand(lo, hi, fmt):
            return [fmt.format(i) for i in range(lo, hi)]

        self.assertEqual(testexpand("ds-{000000..000009}.tar"), myexpand(0, 10, "ds-{:06d}.tar"))
        self.assertEqual(testexpand("{0..9}"), myexpand(0, 10, "{}"))
        self.assertEqual(testexpand("{0..999}"), myexpand(0, 1000, "{}"))
        self.assertEqual(testexpand("{123..999}"), myexpand(123, 1000, "{}"))
        self.assertEqual(testexpand("{000..999}"), myexpand(0, 1000, "{:03d}"))
        with self.assertRaisesRegex(ValueError, r"must not start with 0"):
            testexpand("{01..999}")
        with self.assertRaisesRegex(ValueError, r"must be shorter"):
            testexpand("{0000..999}")
        with self.assertRaisesRegex(ValueError, r"bad range"):
            testexpand("{999..123}")
        self.assertEqual(testexpand("{0..1}{0..1}"), "00 01 10 11".split())

    def test_combining_infinite_iterdatapipe(self):
        r"""
        Test combining DataPipe can properly exit at the end of iteration
        with an infinite DataPipe as the input.
        """

        def _get_dp(length=10):
            source_dp = IterableWrapper(list(range(length)))
            inf_dp = IterableWrapper(list(range(length))).cycle()
            return source_dp, inf_dp

        # zip
        noinf_dp, inf_dp = _get_dp(10)
        dp = inf_dp.zip(noinf_dp)
        res = list(dp)
        self.assertEqual(res, [(i, i) for i in range(10)])

        # mux
        noinf_dp, inf_dp = _get_dp(10)
        dp = inf_dp.mux(noinf_dp)
        res = list(dp)
        self.assertEqual(res, [i for i in range(10) for _ in range(2)])

        # zip_with_iter
        noinf_dp, inf_dp = _get_dp(10)
        dp = noinf_dp.zip_with_iter(inf_dp, key_fn=lambda x: x)
        res = list(dp)
        self.assertEqual(res, [(i, i) for i in range(10)])

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

    def test_drop_iterdatapipe(self):
        # tuple tests
        input_dp = IterableWrapper([(0, 1, 2), (3, 4, 5), (6, 7, 8)])

        # Functional Test: single index drop for tuple elements
        drop_dp = input_dp.drop(1)
        self.assertEqual([(0, 2), (3, 5), (6, 8)], list(drop_dp))

        # Functional Test: multiple indices drop for tuple elements
        drop_dp = input_dp.drop([0, 2])
        self.assertEqual([(1,), (4,), (7,)], list(drop_dp))

        # dict tests
        input_dp = IterableWrapper([{"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 4, "c": 5}, {"a": 5, "b": 6, "c": 7}])

        # Functional Test: single key drop for dict elements
        drop_dp = input_dp.drop("a")
        self.assertEqual([{"b": 2, "c": 3}, {"b": 4, "c": 5}, {"b": 6, "c": 7}], list(drop_dp))

        # Functional Test: multiple key drop for dict elements
        drop_dp = input_dp.drop(["a", "b"])
        self.assertEqual([{"c": 3}, {"c": 5}, {"c": 7}], list(drop_dp))

        # list tests
        input_dp = IterableWrapper([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        # Functional Test: single key drop for list elements
        drop_dp = input_dp.drop(2)
        self.assertEqual([[0, 1], [3, 4], [6, 7]], list(drop_dp))

        # Functional Test: multiple key drop for list elements
        drop_dp = input_dp.drop([0, 1])
        self.assertEqual([[2], [5], [8]], list(drop_dp))

        # Reset Test:
        n_elements_before_reset = 2
        input_dp = IterableWrapper([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        drop_dp = input_dp.drop([0, 1])
        expected_res = [[2], [5], [8]]
        res_before_reset, res_after_reset = reset_after_n_next_calls(drop_dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

        # __len__ Test:
        input_dp = IterableWrapper([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        drop_dp = input_dp.drop([0, 1])
        self.assertEqual(3, len(drop_dp))

    def test_slice_iterdatapipe(self):
        # tuple tests
        input_dp = IterableWrapper([(0, 1, 2), (3, 4, 5), (6, 7, 8)])

        # Functional Test: slice with no stop and no step for tuple
        slice_dp = input_dp.slice(1)
        self.assertEqual([(1, 2), (4, 5), (7, 8)], list(slice_dp))

        # Functional Test: slice with no step for tuple
        slice_dp = input_dp.slice(0, 2)
        self.assertEqual([(0, 1), (3, 4), (6, 7)], list(slice_dp))

        # Functional Test: slice with step for tuple
        slice_dp = input_dp.slice(0, 2, 2)
        self.assertEqual([(0,), (3,), (6,)], list(slice_dp))

        # Functional Test: filter with list of indices for tuple
        slice_dp = input_dp.slice([0, 1])
        self.assertEqual([(0, 1), (3, 4), (6, 7)], list(slice_dp))

        # list tests
        input_dp = IterableWrapper([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        # Functional Test: slice with no stop and no step for list
        slice_dp = input_dp.slice(1)
        self.assertEqual([[1, 2], [4, 5], [7, 8]], list(slice_dp))

        # Functional Test: slice with no step for list
        slice_dp = input_dp.slice(0, 2)
        self.assertEqual([[0, 1], [3, 4], [6, 7]], list(slice_dp))

        # Functional Test: filter with list of indices for list
        slice_dp = input_dp.slice(0, 2)
        self.assertEqual([[0, 1], [3, 4], [6, 7]], list(slice_dp))

        # dict tests
        input_dp = IterableWrapper([{"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 4, "c": 5}, {"a": 5, "b": 6, "c": 7}])

        # Functional Test: filter with list of indices for dict
        slice_dp = input_dp.slice(["a", "b"])
        self.assertEqual([{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}], list(slice_dp))

        # __len__ Test:
        input_dp = IterableWrapper([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        slice_dp = input_dp.slice(0, 2)
        self.assertEqual(3, len(slice_dp))

        # Reset Test:
        n_elements_before_reset = 2
        input_dp = IterableWrapper([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        slice_dp = input_dp.slice([2])
        expected_res = [[2], [5], [8]]
        res_before_reset, res_after_reset = reset_after_n_next_calls(slice_dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

    def test_flatten_iterdatapipe(self):
        # tuple tests

        # Functional Test: flatten for an index
        input_dp = IterableWrapper([(0, 1, (2, 3)), (4, 5, (6, 7)), (8, 9, (10, 11))])
        flatten_dp = input_dp.flatten(2)
        self.assertEqual([(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)], list(flatten_dp))

        # Functional Test: flatten for list of indices
        input_dp = IterableWrapper([((0, 10), 1, (2, 3)), ((4, 14), 5, (6, 7)), ((8, 18), 9, (10, 11))])
        flatten_dp = input_dp.flatten([0, 2])
        self.assertEqual([(0, 10, 1, 2, 3), (4, 14, 5, 6, 7), (8, 18, 9, 10, 11)], list(flatten_dp))

        # Functional Test: flatten all iters in the datapipe one level (no argument)
        input_dp = IterableWrapper([(0, (1, 2)), (3, (4, 5)), (6, (7, 8))])
        flatten_dp = input_dp.flatten()
        self.assertEqual([(0, 1, 2), (3, 4, 5), (6, 7, 8)], list(flatten_dp))

        # list tests

        # Functional Test: flatten for an index
        input_dp = IterableWrapper([[0, 1, [2, 3]], [4, 5, [6, 7]], [8, 9, [10, 11]]])
        flatten_dp = input_dp.flatten(2)
        self.assertEqual([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], list(flatten_dp))

        # Functional Test: flatten for list of indices
        input_dp = IterableWrapper([[[0, 10], 1, [2, 3]], [[4, 14], 5, [6, 7]], [[8, 18], 9, [10, 11]]])
        flatten_dp = input_dp.flatten([0, 2])
        self.assertEqual([[0, 10, 1, 2, 3], [4, 14, 5, 6, 7], [8, 18, 9, 10, 11]], list(flatten_dp))

        # Functional Test: flatten all iters in the datapipe one level (no argument)
        input_dp = IterableWrapper([[0, [1, 2]], [3, [4, 5]], [6, [7, 8]]])
        flatten_dp = input_dp.flatten()
        self.assertEqual([[0, 1, 2], [3, 4, 5], [6, 7, 8]], list(flatten_dp))

        # Functional Test: string test, flatten all iters in the datapipe one level (no argument)
        input_dp = IterableWrapper([["zero", ["one", "2"]], ["3", ["4", "5"]], ["6", ["7", "8"]]])
        flatten_dp = input_dp.flatten()
        self.assertEqual([["zero", "one", "2"], ["3", "4", "5"], ["6", "7", "8"]], list(flatten_dp))

        # dict tests

        # Functional Test: flatten for an index
        input_dp = IterableWrapper([{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}, {"a": 5, "b": 6, "c": {"d": 7, "e": 8}}])
        flatten_dp = input_dp.flatten("c")
        self.assertEqual([{"a": 1, "b": 2, "d": 3, "e": 4}, {"a": 5, "b": 6, "d": 7, "e": 8}], list(flatten_dp))

        # Functional Test: flatten for an index already flat
        input_dp = IterableWrapper([{"a": 1, "b": 2, "c": {"d": 9, "e": 10}}, {"a": 5, "b": 6, "c": {"d": 7, "e": 8}}])
        flatten_dp = input_dp.flatten("a")
        self.assertEqual(
            [{"a": 1, "b": 2, "c": {"d": 9, "e": 10}}, {"a": 5, "b": 6, "c": {"d": 7, "e": 8}}], list(flatten_dp)
        )

        # Functional Test: flatten for list of indices
        input_dp = IterableWrapper(
            [
                {"a": {"f": 10, "g": 11}, "b": 2, "c": {"d": 3, "e": 4}},
                {"a": {"f": 10, "g": 11}, "b": 6, "c": {"d": 7, "e": 8}},
            ]
        )
        flatten_dp = input_dp.flatten(["a", "c"])
        self.assertEqual(
            [{"f": 10, "g": 11, "b": 2, "d": 3, "e": 4}, {"f": 10, "g": 11, "b": 6, "d": 7, "e": 8}], list(flatten_dp)
        )

        # Functional Test: flatten all iters in the datapipe one level (no argument)
        input_dp = IterableWrapper([{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}, {"a": 5, "b": 6, "c": {"d": 7, "e": 8}}])
        flatten_dp = input_dp.flatten()
        self.assertEqual([{"a": 1, "b": 2, "d": 3, "e": 4}, {"a": 5, "b": 6, "d": 7, "e": 8}], list(flatten_dp))

        # Functional Test: flatten all iters one level, multiple iters
        input_dp = IterableWrapper(
            [
                {"a": {"f": 10, "g": 11}, "b": 2, "c": {"d": 3, "e": 4}},
                {"a": {"f": 10, "g": 11}, "b": 6, "c": {"d": 7, "e": 8}},
            ]
        )
        flatten_dp = input_dp.flatten()
        self.assertEqual(
            [{"f": 10, "g": 11, "b": 2, "d": 3, "e": 4}, {"f": 10, "g": 11, "b": 6, "d": 7, "e": 8}], list(flatten_dp)
        )

        # __len__ Test:
        input_dp = IterableWrapper([(0, 1, (2, 3)), (4, 5, (6, 7)), (8, 9, (10, 11))])
        flatten_dp = input_dp.flatten(2)
        self.assertEqual(3, len(flatten_dp))

        # Reset Test:
        n_elements_before_reset = 2
        input_dp = IterableWrapper([(0, 1, (2, 3)), (4, 5, (6, 7)), (8, 9, (10, 11))])
        flatten_dp = input_dp.flatten(2)
        expected_res = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)]
        res_before_reset, res_after_reset = reset_after_n_next_calls(flatten_dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

    def test_length_setter_iterdatapipe(self):
        input_dp = IterableWrapper(range(10))

        # Functional Test: Setting length doesn't change the content of the DataPipe
        dp: IterDataPipe = input_dp.set_length(3)
        self.assertEqual(list(range(10)), list(dp))

        with self.assertRaises(AssertionError):
            input_dp.set_length(-1)

        # __len__ Test: Length is as specified and propagates through
        dp = input_dp.set_length(3).map(lambda x: x + 1)
        self.assertEqual(3, len(dp))

        # Reset Test:
        n_elements_before_reset = 2
        dp = input_dp.set_length(3)
        expected_res = list(range(10))
        res_before_reset, res_after_reset = reset_after_n_next_calls(dp, n_elements_before_reset)
        self.assertEqual(expected_res[:n_elements_before_reset], res_before_reset)
        self.assertEqual(expected_res, res_after_reset)

    def test_random_splitter_iterdatapipe(self):

        n_epoch = 2

        # Functional Test: Split results are the same across epochs
        dp = IterableWrapper(range(10))
        train, valid = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0)
        results = []
        for _ in range(n_epoch):
            res = list(train)
            self.assertEqual(5, len(res))
            results.append(res)
        self.assertEqual(results[0], results[1])
        valid_res = list(valid)
        self.assertEqual(5, len(valid_res))
        self.assertEqual(list(range(10)), sorted(results[0] + valid_res))

        # Functional Test: lengths can be known in advance because it splits evenly into integers.
        self.assertEqual(5, len(train))
        self.assertEqual(5, len(valid))

        # Functional Test: DataPipe can split into 3 DataPipes, and infer `total_length` when not given
        dp = IterableWrapper(range(10))
        train, valid, test = dp.random_split(weights={"train": 0.6, "valid": 0.2, "test": 0.2}, seed=0)
        results = []
        for _ in range(n_epoch):
            res = list(train)
            self.assertEqual(6, len(res))
            results.append(res)
        self.assertEqual(results[0], results[1])
        valid_res = list(valid)
        self.assertEqual(2, len(valid_res))
        test_res = list(test)
        self.assertEqual(2, len(test_res))
        self.assertEqual(list(range(10)), sorted(results[0] + valid_res + test_res))

        # Functional Test: lengths can be known in advance because it splits evenly into integers.
        self.assertEqual(6, len(train))
        self.assertEqual(2, len(valid))
        self.assertEqual(2, len(test))

        # Functional Test: Split can work even when weights do not split evenly into integers.
        dp = IterableWrapper(range(13))
        train, valid, test = dp.random_split(weights={"train": 0.6, "valid": 0.2, "test": 0.2}, seed=0)
        res = list(train) + list(valid) + list(test)
        self.assertEqual(list(range(13)), sorted(res))

        # Functional Test: lengths can be known in advance because it splits evenly into integers.
        with self.assertRaisesRegex(TypeError, "Lengths of the split cannot be known in advance"):
            len(train)

        # Functional Test: Error when `total_length` cannot be inferred
        nolen_dp = IDP_NoLen(range(10))
        with self.assertRaisesRegex(TypeError, "needs `total_length`"):
            _, __ = nolen_dp.random_split(weights={"train": 0.5, "valid": 0.5}, seed=0)  # type: ignore[call-arg]

        # Functional Test: `target` must match a key in the `weights` dict
        dp = IterableWrapper(range(10))
        with self.assertRaisesRegex(KeyError, "does not match any key"):
            _ = dp.random_split(
                total_length=10, weights={"train": 0.5, "valid": 0.2, "test": 0.2}, seed=0, target="NOTINDICT"
            )

        # Functional Test: `target` is specified, and match the results from before
        dp = IterableWrapper(range(10))
        train = dp.random_split(
            total_length=10, weights={"train": 0.6, "valid": 0.2, "test": 0.2}, seed=0, target="train"
        )
        results2 = []
        for _ in range(n_epoch):
            res = list(train)
            self.assertEqual(6, len(res))
            results2.append(res)
        self.assertEqual(results2[0], results2[1])
        self.assertEqual(results, results2)

        # Functional Test: `override_seed` works and change split result
        train.override_seed(1)
        seed_1_res = list(train)
        self.assertNotEqual(results2[0], seed_1_res)

        # Functional Test: `override_seed` doesn't impact the current iteration, only the next one
        temp_res = []
        for i, x in enumerate(train):
            temp_res.append(x)
            if i == 3:
                train.override_seed(0)
        self.assertEqual(seed_1_res, temp_res)  # The current iteration should equal seed 1 result
        self.assertEqual(results2[0], list(train))  # The next iteration should equal seed 0 result

        # Functional Test: Raise exception if both children are used at the same time
        dp = IterableWrapper(range(10))
        train, valid = dp.random_split(total_length=10, weights={"train": 0.5, "valid": 0.5}, seed=0)
        it_train = iter(train)
        next(it_train)
        it_valid = iter(valid)  # This resets the DataPipe and invalidates the other iterator
        next(it_valid)
        with self.assertRaisesRegex(RuntimeError, "iterator has been invalidated"):
            next(it_train)
        next(it_valid)  # No error, can keep going

    @skipIfNoCUDA
    def test_pin_memory(self):
        # Tensor
        dp = IterableWrapper([(i, i + 1) for i in range(10)]).map(_convert_to_tensor).pin_memory()
        self.assertTrue(all(d.is_pinned() for d in dp))

        # List of Tensors
        dp = IterableWrapper([[(i - 1, i), (i, i + 1)] for i in range(10)]).map(_convert_to_tensor).pin_memory()
        self.assertTrue(all(d0.is_pinned() and d1.is_pinned() for d0, d1 in dp))

        # Dict of Tensors
        dp = IterableWrapper([{str(i): (i, i + 1)} for i in range(10)]).map(_convert_to_tensor).pin_memory()
        self.assertTrue(all(v.is_pinned() for d in dp for v in d.values()))

        # Dict of List of Tensors
        dp = (
            IterableWrapper([{str(i): [(i - 1, i), (i, i + 1)]} for i in range(10)])
            .map(_convert_to_tensor)
            .pin_memory()
        )
        self.assertTrue(all(v.is_pinned() for d in dp for batch in d.values() for v in batch))

        # List of Dict of Tensors
        dp = IterableWrapper([{str(i): (i, i + 1)} for i in range(10)]).map(_convert_to_tensor).batch(2).pin_memory()
        self.assertTrue(all(v.is_pinned() for batch in dp for d in batch for v in d.values()))

        # List of List of Tensors
        dp = (
            IterableWrapper([[(i - 1, i), (i, i + 1)] for i in range(10)]).map(_convert_to_tensor).batch(2).pin_memory()
        )
        self.assertTrue(all(d0.is_pinned() and d1.is_pinned() for batch in dp for d0, d1 in batch))


if __name__ == "__main__":
    unittest.main()
