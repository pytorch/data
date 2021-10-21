# Copyright (c) Facebook, Inc. and its affiliates.
import io
import expecttest
import unittest
import warnings

from collections import defaultdict
from typing import Dict

import torchdata
import torch.utils.data.datapipes.iter

from torchdata.datapipes.iter import (
    IterDataPipe,
    IterableWrapper,
    InMemoryCacheHolder,
    KeyZipper,
    Cycler,
    Header,
    IndexAdder,
    LineReader,
    ParagraphAggregator,
    Rows2Columnar,
    SampleMultiplexer,
    BucketBatcher,
)

from _utils._common_utils_for_test import (
    IDP_NoLen,
    reset_after_n_next_calls,
)


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
    if any(missing_datapipes):
        msg = (
            "The following datapipes are exposed under `torch.utils.data.datapipes.iter`, "
            "but not under `torchdata.datapipes.iter`:\n"
        )
        raise AssertionError(msg + "\n".join(sorted(missing_datapipes)))


class TestDataPipe(expecttest.TestCase):
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

    def test_keyzipper_iterdatapipe(self) -> None:

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
        self.assertEqual([(i, (i, i)) for i in range(10)], list(zip_dp_w_key))

        # Functional Test: using a different merge function
        def merge_to_string(item1, item2):
            return f"{item1},{item2}"

        zip_dp_w_str_merge = source_dp.zip_by_key(
            ref_datapipe=ref_dp, key_fn=lambda x: x, ref_key_fn=lambda x: x, buffer_size=10, merge_fn=merge_to_string
        )
        self.assertEqual([f"{i},{i}" for i in range(10)], list(zip_dp_w_str_merge))

        # Functional Test: using a different merge function and keep_key=True
        zip_dp_w_key_str_merge = source_dp.zip_by_key(
            ref_datapipe=ref_dp,
            key_fn=lambda x: x,
            ref_key_fn=lambda x: x,
            keep_key=True,
            buffer_size=10,
            merge_fn=merge_to_string,
        )
        self.assertEqual([(i, f"{i},{i}") for i in range(10)], list(zip_dp_w_key_str_merge))

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

        # TODO: __len__ Test: returns the length of source when it is less than the limit
        # header_dp = source_dp.header(30)
        # self.assertEqual(20, len(header_dp))

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

    def test_bucket_batcher_iterdatapipe(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
