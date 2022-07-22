# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pickle
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar, Union

from torch.utils.data.graph import DataPipe

from torchdata.dataloader2.adapter import Adapter

from .error import PauseIteration
from .reading_service import (
    CheckpointableReadingServiceInterface,
    ReadingServiceInterface,
)

T_co = TypeVar("T_co", covariant=True)
SERIALIZED_DATAPIPE_KEY_NAME = "serialized_datapipe"
READING_SERVICE_STATE_KEY_NAME = "reading_service_state"


def serialize_datapipe(datapipe: DataPipe) -> bytes:
    try:
        return pickle.dumps(datapipe)
    except pickle.PickleError as e:
        raise NotImplementedError(
            f"Prototype only support pickle-able datapipes for checkpoint: {e}"
        )


def deserialize_datapipe(serialized_state: bytes) -> DataPipe:
    try:
        return pickle.loads(serialized_state)
    except pickle.PickleError as e:
        raise NotImplementedError(
            f"Prototype only support pickle-able datapipes for checkpoint: {e}"
        )


@dataclass
class ConcurrencySpec:
    num_workers: int
    timeout: Optional[int] = None
    prefetch_factor: int = 2
    persistent_workers: bool = False


class DataLoader2(Generic[T_co]):
    def __init__(
        self,
        datapipe: DataPipe,
        datapipe_adapter_fn: Optional[Union[Iterable[Adapter], Adapter]] = None,
        reading_service: Optional[ReadingServiceInterface] = None,
    ) -> None:
        self.datapipe = datapipe
        self._adapted: bool = False
        self._datapipe_iter: Optional[Iterator[T_co]] = None
        self._reset_iter: bool = True
        # TODO(630): Some ReadingServices might want to validate adapters, we can add this feature
        if datapipe_adapter_fn is None:
            self.datapipe_adapter_fns = None
        elif isinstance(datapipe_adapter_fn, Iterable):
            self.datapipe_adapter_fns = datapipe_adapter_fn
        else:
            self.datapipe_adapter_fns = [datapipe_adapter_fn]
        self.reading_service = reading_service
        self.reading_service_state: Optional[bytes] = None
        self._terminated: bool = False

        if self.datapipe_adapter_fns is not None:
            for adapter_fn in self.datapipe_adapter_fns:
                self.datapipe = adapter_fn(self.datapipe)
        self._datapipe_before_reading_service_adapt: DataPipe = self.datapipe

    def __iter__(self) -> Iterator[T_co]:
        if self._terminated:
            raise Exception(
                "Cannot iterate over the DataLoader as it has already been shut down"
            )

        if self._reset_iter:
            if not self._adapted and self.reading_service is not None:
                if self.reading_service_state is None:
                    self.datapipe = self.reading_service.initialize(self.datapipe)
                else:
                    if not isinstance(
                        self.reading_service, CheckpointableReadingServiceInterface
                    ):
                        raise TypeError(
                            "Cannot restore from non-checkpointable reading service"
                        )
                    self.datapipe = self.reading_service.restore(
                        self.datapipe, self.reading_service_state
                    )
                self._adapted = True

            if self.reading_service is not None:
                self.reading_service.initialize_iteration()

            self._datapipe_iter = iter(self.datapipe)

            self._reset_iter = False

        return self

    def __next__(self) -> T_co:
        if self._reset_iter:
            raise StopIteration
        try:
            return next(self._datapipe_iter)  # type: ignore[arg-type]
        except PauseIteration:
            raise StopIteration
        except StopIteration:
            if self.reading_service is not None:
                self.reading_service.finalize_iteration()
            self._reset_iter = True
            raise

    def __getattr__(self, name: str) -> Any:
        if self._datapipe_iter is None:
            raise AttributeError
        return getattr(self._datapipe_iter, name)

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if not self._reset_iter:
            self._reset_iter = True
            self._datapipe_iter = None
        if not self._terminated:
            if self.reading_service is not None:
                self.reading_service.finalize()
            self._terminated = True

    def __enter__(self) -> "DataLoader2[T_co]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return: Dict[str, Any]
        {"serialized_datapipe": bytes, "reading_service_state": bytes}

        Serialized DataPipe: Use datapipe before reading service adaption.
        Reading Service State: Reading Service checkpoint information.
        """
        reading_service_state = None
        if self.reading_service is not None and isinstance(
            self.reading_service, CheckpointableReadingServiceInterface
        ):
            reading_service_state = self.reading_service.checkpoint()

        # Serialize datapipe after applying adapters and before reading service adaption
        serialized_datapipe = serialize_datapipe(
            self._datapipe_before_reading_service_adapt
        )

        return {
            SERIALIZED_DATAPIPE_KEY_NAME: serialized_datapipe,
            READING_SERVICE_STATE_KEY_NAME: reading_service_state,
        }

    @classmethod
    def from_state(
        cls,
        state: Dict[str, Any],
        reading_service: CheckpointableReadingServiceInterface,
    ) -> "DataLoader2[T_co]":
        """
        Create new DataLoader with deserialized datapipe and reading service
        Set reading_service_state to new DataLoader for Reading Service to restore.
        """
        serialized_datapipe = state[SERIALIZED_DATAPIPE_KEY_NAME]
        reading_service_state = state[READING_SERVICE_STATE_KEY_NAME]

        data_loader: "DataLoader2[T_co]" = DataLoader2(
            datapipe=deserialize_datapipe(serialized_datapipe),
            datapipe_adapter_fn=None,
            reading_service=reading_service,
        )
        data_loader.reading_service_state = reading_service_state
        return data_loader

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # edge case checking
        # iterator has already been created: 1) iterator is just created 2) iterator is created and iter is exhausted
        if self._datapipe_iter is not None:
            raise RuntimeError(
                "DataLoaderV2 iterator has already been created, `load_state_dict()` can’t be called. Please create a new dataloader in order to use load state dict."
            )

        serialized_datapipe = state[SERIALIZED_DATAPIPE_KEY_NAME]
        reading_service_state = state[READING_SERVICE_STATE_KEY_NAME]

        # deserialize datapipe
        deserialized_datapipe = deserialize_datapipe(serialized_datapipe)

        # override existing datapipe and reading service state
        self.datapipe = deserialized_datapipe
        self.reading_service_state = reading_service_state

        # re-initialize datapipe_adapter_fn and _datapipe_before_reading_service_adapt
        if self.datapipe_adapter_fns is not None:
            for adapter_fn in self.datapipe_adapter_fns:
                self.datapipe = adapter_fn(self.datapipe)
        self._datapipe_before_reading_service_adapt = self.datapipe
