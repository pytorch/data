# Copyright (c) Facebook, Inc. and its affiliates.
import expecttest
import hashlib
import itertools
import lzma
import os
import tarfile
import unittest
import warnings
import zipfile

from json.decoder import JSONDecodeError

from torch.utils.data.datapipes.map import SequenceWrapper
from torchdata.datapipes.iter import (
    FileLister,
    FileLoader,
    IterableWrapper,
    MapZipper,
    IoPathFileLister,
    IoPathFileLoader,
    CSVParser,
    CSVDictParser,
    HashChecker,
    JsonParser,
    Saver,
    TarArchiveReader,
    ZipArchiveReader,
    XzFileReader,
)

from _utils._common_utils_for_test import (
    create_temp_dir_and_files,
    get_name,
    reset_after_n_next_calls,
)

try:
    import iopath  # type: ignore[import]  # noqa: F401

    HAS_IOPATH = True
except ImportError:
    HAS_IOPATH = False
skipIfNoIOPath = unittest.skipIf(not HAS_IOPATH, "no iopath")


class TestDataPipeLocalIO(expecttest.TestCase):
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
            warnings.warn(f"TestDataPipeLocalIO was not able to cleanup temp dir due to {e}")

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


if __name__ == "__main__":
    unittest.main()
