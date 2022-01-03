# Copyright (c) Facebook, Inc. and its affiliates.
import os
import unittest
import warnings

import expecttest

from _utils._common_utils_for_test import create_temp_dir, create_temp_files, reset_after_n_next_calls

from torchdata.datapipes.iter import FileLister, FSSpecFileLister, FSSpecFileOpener, FSSpecSaver, IterableWrapper

try:
    import fsspec

    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False
skipIfNoFSSpec = unittest.skipIf(not HAS_FSSPEC, "no fsspec")


class TestDataPipeFSSpec(expecttest.TestCase):
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
            warnings.warn(f"TestDataPipeLocalIO was not able to cleanup temp dir due to {e}")

    def _write_text_files(self):
        def filepath_fn(name: str) -> str:
            return os.path.join(self.temp_dir.name, os.path.basename(name))

        name_to_data = {"1.text": b"DATA", "2.text": b"DATA", "3.text": b"DATA"}
        source_dp = IterableWrapper(sorted(name_to_data.items()))
        saver_dp = source_dp.save_to_disk(filepath_fn=filepath_fn, mode="wb")
        list(saver_dp)

    @skipIfNoFSSpec
    def test_fsspec_file_lister_iterdatapipe(self):
        datapipe = FSSpecFileLister(root="file://" + self.temp_sub_dir.name)

        # check all file paths within sub_folder are listed
        for path in datapipe:
            self.assertIn(
                path.split("://")[1],
                {fsspec.implementations.local.make_path_posix(file) for file in self.temp_sub_files},
            )

    @skipIfNoFSSpec
    def test_fsspec_file_loader_iterdatapipe(self):
        datapipe1 = FSSpecFileLister(root="file://" + self.temp_sub_dir.name)
        datapipe2 = FSSpecFileOpener(datapipe1)

        # check contents of file match
        for _, f in datapipe2:
            self.assertEqual(f.read(), "0123456789abcdef")

        # Reset Test: Ensure the resulting streams are still readable after the DataPipe is reset/exhausted
        self._write_text_files()
        lister_dp = FileLister(self.temp_dir.name, "*.text")
        fsspec_file_loader_dp = FSSpecFileOpener(lister_dp, mode="rb")

        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(fsspec_file_loader_dp, n_elements_before_reset)
        self.assertEqual(2, len(res_before_reset))
        self.assertEqual(3, len(res_after_reset))
        for _name, stream in res_before_reset:
            self.assertEqual(b"DATA", stream.read())
        for _name, stream in res_after_reset:
            self.assertEqual(b"DATA", stream.read())

    @skipIfNoFSSpec
    def test_fsspec_saver_iterdatapipe(self):
        def filepath_fn(name: str) -> str:
            return "file://" + os.path.join(self.temp_dir.name, os.path.basename(name))

        # Functional Test: Saving some data
        name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2", "3.txt": b"DATA3"}
        source_dp = IterableWrapper(sorted(name_to_data.items()))
        saver_dp = source_dp.save_by_fsspec(filepath_fn=filepath_fn, mode="wb")
        res_file_paths = list(saver_dp)
        expected_paths = [filepath_fn(name) for name in name_to_data.keys()]
        self.assertEqual(expected_paths, res_file_paths)
        for name in name_to_data.keys():
            p = filepath_fn(name).split("://")[1]
            with open(p, "r") as f:
                self.assertEqual(name_to_data[name], f.read().encode())

        # Reset Test:
        saver_dp = FSSpecSaver(source_dp, filepath_fn=filepath_fn, mode="wb")
        n_elements_before_reset = 2
        res_before_reset, res_after_reset = reset_after_n_next_calls(saver_dp, n_elements_before_reset)
        self.assertEqual([filepath_fn("1.txt"), filepath_fn("2.txt")], res_before_reset)
        self.assertEqual(expected_paths, res_after_reset)
        for name in name_to_data.keys():
            p = filepath_fn(name).split("://")[1]
            with open(p, "r") as f:
                self.assertEqual(name_to_data[name], f.read().encode())

        # __len__ Test: returns the length of source DataPipe
        self.assertEqual(3, len(saver_dp))

    @skipIfNoFSSpec
    def test_fsspec_memory_list(self):
        fs = fsspec.filesystem("memory")
        fs.mkdir("foo")
        fs.touch("foo/bar1")
        fs.touch("foo/bar2")

        datapipe = FSSpecFileLister(root="memory://foo")
        self.assertEqual(set(datapipe), {"memory:///foo/bar1", "memory:///foo/bar2"})

        datapipe = FSSpecFileLister(root="memory://foo/bar1")
        self.assertEqual(set(datapipe), {"memory://foo/bar1"})

    @skipIfNoFSSpec
    def test_fsspec_memory_load(self):
        fs = fsspec.filesystem("memory")
        with fs.open("file", "w") as f:
            f.write("hello")
        with fs.open("file2", "w") as f:
            f.write("hello2")

        files = ["memory://file", "memory://file2"]
        datapipe = FSSpecFileOpener(files)
        self.assertEqual([f.read() for _, f in datapipe], ["hello", "hello2"])

    @skipIfNoFSSpec
    def test_fsspec_memory_save(self):
        def filepath_fn(name: str) -> str:
            return "memory://" + name

        name_to_data = {"1.txt": b"DATA1", "2.txt": b"DATA2"}
        source_dp = IterableWrapper(sorted(name_to_data.items()))
        saver_dp = FSSpecSaver(source_dp, filepath_fn=filepath_fn, mode="wb")

        self.assertEqual(set(saver_dp), {"memory://1.txt", "memory://2.txt"})


if __name__ == "__main__":
    unittest.main()
