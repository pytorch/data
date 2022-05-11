# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
from typing import Any, Dict, Optional, Tuple

from torchdata.dataloader2.dataloader2 import READING_SERVICE_STATE_KEY_NAME, SERIALIZED_DATAPIPE_KEY_NAME

logger: logging.Logger = logging.getLogger()


def try_deserialize_as_dlv2_checkpoint(checkpoint_bytes: Optional[bytes]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Handling the checkpoint conversion from bytes to DataLoader V2 format.
    Model store checkpoint agent only store ByteIO/Tensor/ShardedTensor.

    Args:
        checkpoint_bytes: checkpoint in bytes format.

    Returns:
        Tuple[succeeded, dataloader2_checkpoint]: if we succeeded in converting
        to dataloader v2 checkpoint, and the checkpoint as in dataloader2
        checkpoint format.
    """
    if checkpoint_bytes is None:
        logger.info("Empty reader checkpoint.")
        return False, None
    try:
        deserialized_state_dict = pickle.loads(checkpoint_bytes)
        if isinstance(deserialized_state_dict, Dict):
            if (
                SERIALIZED_DATAPIPE_KEY_NAME in deserialized_state_dict.keys()
                and READING_SERVICE_STATE_KEY_NAME in deserialized_state_dict.keys()
            ):
                logger.info("Checkpoint deserialized as dataloader v2 checkpoing.")
                return True, deserialized_state_dict
    except pickle.UnpicklingError:
        logger.info("Reader checkpoint UnpicklingError. Proceed as dataloader v1 checkpoint.")
    except Exception:
        logger.warn("Exception when deserializing reader checkpoint as dataloader v2 checkpoint")
        raise
    return False, None


def serialize_dlv2_checkpoint(dlv2_checkpoint: Dict[str, Any]) -> bytes:
    """
    Handling the checkpoint conversion from DataLoader V2 format to bytes format.
    """
    return pickle.dumps(dlv2_checkpoint)
