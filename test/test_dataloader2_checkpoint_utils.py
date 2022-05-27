# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from typing import Any, Dict, Optional, Tuple
from unittest import TestCase

from torchdata.dataloader2.dataloader2 import READING_SERVICE_STATE_KEY_NAME, SERIALIZED_DATAPIPE_KEY_NAME
from torchdata.dataloader2.dataloader2_checkpoint_utils import try_deserialize_as_dlv2_checkpoint


class DataLoader2CheckpointUtilTest(TestCase):
    def _test_try_deserialize_as_dlv2_checkpoint(
        self, checkpoint_bytes: Optional[bytes], expected: Tuple[bool, Optional[Dict[str, Any]]]
    ) -> None:
        ans = try_deserialize_as_dlv2_checkpoint(checkpoint_bytes)
        self.assertEqual(ans, expected)

    def test_try_deserialize_as_dlv2_checkpoint(self) -> None:
        self._test_try_deserialize_as_dlv2_checkpoint(None, (False, None))

        self._test_try_deserialize_as_dlv2_checkpoint(b"123", (False, None))

        dlv2_state_dict1 = {
            SERIALIZED_DATAPIPE_KEY_NAME: "",
            READING_SERVICE_STATE_KEY_NAME: "",
        }
        self._test_try_deserialize_as_dlv2_checkpoint(pickle.dumps(dlv2_state_dict1), (True, dlv2_state_dict1))

        dlv2_state_dict2 = {
            SERIALIZED_DATAPIPE_KEY_NAME: b"123",
            READING_SERVICE_STATE_KEY_NAME: b"456",
        }

        self._test_try_deserialize_as_dlv2_checkpoint(pickle.dumps(dlv2_state_dict2), (True, dlv2_state_dict2))
