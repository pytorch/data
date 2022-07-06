# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import string
import tempfile
import unittest

from torchdata.datapipes.iter import AISFileLister, AISFileLoader

try:
    from aistore.client.api import Client
    from aistore.client.errors import AISError, ErrBckNotFound

    AIS_CLUSTER_ENDPT = "http://localhost:8080"

    HAS_AIS = Client(AIS_CLUSTER_ENDPT).is_aistore_running()
except (ImportError, ConnectionError):
    HAS_AIS = False
skipIfNoAIS = unittest.skipIf(not HAS_AIS, "AIS not running or library not installed")


@skipIfNoAIS
class TestDataPipeLocalIO(unittest.TestCase):
    def setUp(self):
        # initialize client and create new bucket
        self.client = Client(AIS_CLUSTER_ENDPT)
        letters = string.ascii_lowercase
        self.bck_name = "".join(random.choice(letters) for _ in range(10))
        self.client.create_bucket(self.bck_name)
        # create temp files
        num_objs = 10

        # create 10 objects in the `/temp` dir
        for i in range(num_objs):
            object_body = "test string" * random.randrange(1, 10)
            content = object_body.encode("utf-8")
            obj_name = f"temp/obj{ i }"
            with tempfile.NamedTemporaryFile() as file:
                file.write(content)
                file.flush()
                self.client.put_object(self.bck_name, obj_name, file.name)

        # create 10 objects in the `/`dir
        for i in range(num_objs):
            object_body = "test string" * random.randrange(1, 10)
            content = object_body.encode("utf-8")
            obj_name = f"obj{ i }"
            with tempfile.NamedTemporaryFile() as file:
                file.write(content)
                file.flush()
                self.client.put_object(self.bck_name, obj_name, file.name)

    def tearDown(self):
        # Try to destroy bucket and its items
        try:
            self.client.destroy_bucket(self.bck_name)
        except ErrBckNotFound:
            pass

    def test_ais_io_iterdatapipe(self):

        prefixes = [
            ["ais://" + self.bck_name],
            ["ais://" + self.bck_name + "/"],
            ["ais://" + self.bck_name + "/temp/", "ais://" + self.bck_name + "/obj"],
        ]

        # check if the created files exist
        for prefix in prefixes:
            urls = AISFileLister(url=AIS_CLUSTER_ENDPT, source_datapipe=prefix)
            ais_loader = AISFileLoader(url=AIS_CLUSTER_ENDPT, source_datapipe=urls)
            with self.assertRaises(TypeError):
                len(urls)
            self.assertEqual(len(list(urls)), 20)
            self.assertEqual(sum(1 for _ in ais_loader), 20)

        # check for incorrect prefixes
        prefixes = ["ais://asdasd"]

        # AISFileLister: Bucket not found
        try:
            list(AISFileLister(url=AIS_CLUSTER_ENDPT, source_datapipe=prefixes))
        except ErrBckNotFound as err:
            self.assertEqual(err.status_code, 404)

        # AISFileLoader: incorrect inputs
        url_list = [[""], ["ais:"], ["ais://"], ["s3:///unkown-bucket"]]

        for url in url_list:
            with self.assertRaises(AISError):
                file_loader = AISFileLoader(url=AIS_CLUSTER_ENDPT, source_datapipe=url)
                for _ in file_loader:
                    pass


if __name__ == "__main__":
    unittest.main()
