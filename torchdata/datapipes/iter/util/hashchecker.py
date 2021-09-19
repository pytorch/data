# Copyright (c) Facebook, Inc. and its affiliates.
import hashlib
import os

from torch.utils.data import IterDataPipe, functional_datapipe


@functional_datapipe('check_hash')
class HashCheckerIterDataPipe(IterDataPipe):
    """
        Usage: dp = dp.check_hash({'train.py':'0d8b94d9fa9fb1ad89b9e3da9e1521495dca558fc5213b0fd7fd7b71c23f9921'})
    """

    def __init__(self, source_datapipe, hash_dict, hash_type="sha256"):
        self.source_datapipe = source_datapipe
        self.hash_dict = hash_dict
        self.hash_type = hash_type

        if self.hash_type not in ["sha256", "md5"]:
            raise ValueError(
                "Invalid hash_type requested, should be one of {}".format(["sha256", "md5"]))

    def __iter__(self):

        for file_name, stream in self.source_datapipe:
            if self.hash_type == "sha256":
                hash_func = hashlib.sha256()
            else:
                hash_func = hashlib.md5()

            while True:
                # Read by chunk to avoid filling memory
                chunk = stream.read(1024 ** 2)
                if not chunk:
                    break
                hash_func.update(chunk)

            # Rewind steam back (if possible)
            # TODO(VitalyFedyunin): this will not work (or work crappy for non-seekable steams like http)
            stream.seek(0)

            if file_name not in self.hash_dict:
                raise RuntimeError(
                    "Unspecified hash for file {}".format(file_name))

            if hash_func.hexdigest() != self.hash_dict[file_name]:
                raise RuntimeError("The hash {} of {} does not match. Delete the file manually and retry.".format(
                    hash_func.hexdigest(), file_name))

            yield (file_name, stream)

    def __len__(self):
        return len(self.source_datapipe)
