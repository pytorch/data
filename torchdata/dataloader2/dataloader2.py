# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pickle
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar, Union

from torch.utils.data.datapipes.datapipe import (
    _DataPipeSerializationWrapper,
    _IterDataPipeSerializationWrapper,
    _MapDataPipeSerializationWrapper,
)

from torch.utils.data.graph import DataPipe
from torchdata.dataloader2.adapter import Adapter

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.map import MapDataPipe

from .error import PauseIteration
from .reading_service import CheckpointableReadingServiceInterface, ReadingServiceInterface

T_co = TypeVar("T_co", covariant=True)
SERIALIZED_DATAPIPE_KEY_NAME = "serialized_datapipe"
READING_SERVICE_STATE_KEY_NAME = "reading_service_state"


def serialize_datapipe(datapipe: DataPipe) -> bytes:
    try:
        return pickle.dumps(datapipe)
    except pickle.PickleError as e:
        raise NotImplementedError(f"Prototype only support pickle-able datapipes for checkpoint: {e}")


def deserialize_datapipe(serialized_state: bytes) -> DataPipe:
    try:
        return pickle.loads(serialized_state)
    except pickle.PickleError as e:
        raise NotImplementedError(f"Prototype only support pickle-able datapipes for checkpoint: {e}")


@dataclass
class ConcurrencySpec:
    num_workers: int
    timeout: Optional[int] = None
    prefetch_factor: int = 2
    persistent_workers: bool = False


class DataLoader2Iterator(Iterator[T_co]):
    def __init__(self, dataloader: "DataLoader2", iterator_id: int):
        self.dataloader = dataloader
        self.iterator_id = iterator_id

    def __next__(self) -> T_co:
        if self.iterator_id == self.dataloader.valid_iterator_id:
            self.dataloader._reset_iter = True
            try:
                return next(self.dataloader._datapipe_iter)  # type: ignore[arg-type]
            except PauseIteration:
                raise StopIteration
            except StopIteration:
                if self.dataloader.reading_service is not None:
                    self.dataloader.reading_service.finalize_iteration()
                raise
            except Exception:
                if self.dataloader:
                    self.dataloader.shutdown()
                raise
        else:
            if self.dataloader.reading_service is not None:
                self.dataloader.reading_service.finalize_iteration()
            raise RuntimeError(
                "This iterator has been invalidated because another iterator has been created "
                "from the same DataLoader2.\n"
                "This may be caused multiple references to the same DataLoader2. "
                "For feedback regarding this single iterator per DataLoader2 constraint, feel free "
                "to comment on this issue: https://github.com/pytorch/data/issues/45."
            )

    def __getattr__(self, name):
        """
        To delegate operations to ``dataloader._datapipe_iter``.
        """
        if self.dataloader._datapipe_iter is None:
            raise AttributeError
        return getattr(self.dataloader._datapipe_iter, name)


class DataLoader2(Generic[T_co]):
    r"""
    ``DataLoader2`` is used to optimize and execute the given ``DataPipe`` graph
    based on ``ReadingService`` and ``Adapter`` functions, with support for

    - Dynamic sharding for multi-process and distributed data loading
    - Multiple backend ``ReadingServices``
    - ``DataPipe`` graph in-place modification like shuffle control, memory pinning, etc.
    - Snapshot the state of data-preprocessing pipeline (WIP)

    Args:
        datapipe (``IterDataPipe`` or ``MapDataPipe``): ``DataPipe`` from which to load the data. A deepcopy of this will be made during
            initialization, allowing the input to be re-used in a different ``DataLoader2`` without sharing states.
            ``None`` can only be used together with ``load_state_dict`` right after DataLoader creation.
        datapipe_adapter_fn (``Iterable[Adapter]`` or ``Adapter``, optional): ``Adapter`` function(s) that will be applied
            to the DataPipe (default: ``None``).
        reading_service (ReadingServiceInterface, optional): defines how ``DataLoader2`` should execute operations over
            the ``DataPipe``, e.g. multiprocessing/distributed (default: ``None``). A deepcopy of this will be made during
            initialization, allowing the input to be re-used in a different ``DataLoader2`` without sharing states.
    """

    def __init__(
        self,
        datapipe: Optional[DataPipe],
        datapipe_adapter_fn: Optional[Union[Iterable[Adapter], Adapter]] = None,
        reading_service: Optional[ReadingServiceInterface] = None,
    ) -> None:
        self.datapipe = self._wrap_and_copy_dp(datapipe) if datapipe is not None else None
        self._adapted: bool = False
        self._datapipe_iter: Optional[Iterator[T_co]] = None
        self._reset_iter: bool = True  # Sets to `False` when __iter__ starts, and `True` when `StopIteration``
        # TODO(630): Some ReadingServices might want to validate adapters, we can add this feature
        if datapipe_adapter_fn is None:
            self.datapipe_adapter_fns = None
        elif isinstance(datapipe_adapter_fn, Iterable):
            self.datapipe_adapter_fns = datapipe_adapter_fn
        else:
            self.datapipe_adapter_fns = [datapipe_adapter_fn]
        self.reading_service = reading_service
        self.reading_service_state: Optional[bytes] = None  # is not `None` when `load_state_dict` is called
        self._terminated: bool = False
        self.valid_iterator_id: Optional[int] = None

        if self.datapipe is not None and self.datapipe_adapter_fns is not None:
            for adapter_fn in self.datapipe_adapter_fns:
                self.datapipe = adapter_fn(self.datapipe)
        self._datapipe_before_reading_service_adapt: DataPipe = self._copy(self.datapipe)

    def __iter__(self) -> DataLoader2Iterator[T_co]:
        r"""
        Return a singleton iterator from the ``DataPipe`` graph adapted by ``ReadingService``.
        ``DataPipe`` will be restored if the serialized state is provided to construct
        ``DataLoader2``. And, ``initialize_iteration`` and ``finalize_iterator`` will be
        invoked at the beginning and end of the iteration correspondingly.
        """
        if self.datapipe is None:
            raise RuntimeError("Please provide datapipe or use load_state_dict to load datapipe from state")

        if self._terminated:
            raise RuntimeError("Cannot iterate over the DataLoader as it has already been shut down")

        if self._reset_iter:
            if not self._adapted and self.reading_service is not None:
                if self.reading_service_state is None:
                    self.datapipe = self.reading_service.initialize(self.datapipe)
                else:
                    if not isinstance(self.reading_service, CheckpointableReadingServiceInterface):
                        raise TypeError("Cannot restore from non-checkpointable reading service")
                    self.datapipe = self.reading_service.restore(self.datapipe, self.reading_service_state)
                self._adapted = True

            if self.reading_service is not None:
                self.reading_service.initialize_iteration()

            self._datapipe_iter = iter(self.datapipe)
            self._reset_iter = False

        self.valid_iterator_id = 0 if self.valid_iterator_id is None else self.valid_iterator_id + 1
        return DataLoader2Iterator(self, self.valid_iterator_id)

    def __del__(self) -> None:
        self.shutdown()

    @staticmethod
    def _copy(obj):
        r"""
        Standardized way for DataLoader2 to copy an object when needed, such as for DataPipe/ReadingService.
        This uses `pickle` to serialize/deserialize to create the copy.
        """
        return pickle.loads(pickle.dumps(obj))

    @staticmethod
    def _wrap_and_copy_dp(datapipe: DataPipe):
        r"""
        Wraps the ``DataPipe`` with the corresponding serialization wrapper.
        Then, creates a copy with the class's static copy method.

        """
        wrapped_dp: DataPipe = datapipe
        if not isinstance(datapipe, _DataPipeSerializationWrapper):
            if isinstance(datapipe, IterDataPipe):
                wrapped_dp = _IterDataPipeSerializationWrapper(datapipe)
            elif isinstance(datapipe, MapDataPipe):
                wrapped_dp = _MapDataPipeSerializationWrapper(datapipe)
        return DataLoader2._copy(wrapped_dp)

    def shutdown(self) -> None:
        r"""
        Shuts down ``ReadingService`` and clean up iterator.
        """
        if not self._reset_iter:
            self._reset_iter = True
            self._datapipe_iter = None
        if not self._terminated:
            if self.reading_service is not None:
                self.reading_service.finalize_iteration()
                self.reading_service.finalize()
            self._terminated = True

    def __enter__(self) -> "DataLoader2[T_co]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()

    def state_dict(self) -> Dict[str, Any]:
        r"""
        Return a dictionary to represent the state of data-processing pipeline with keys:

        - ``serialized_datapipe``:Serialized ``DataPipe`` before ``ReadingService`` adaption.
        - ``reading_service_state``: The state of ``ReadingService`` and adapted ``DataPipe``.
        """
        reading_service_state = None
        if self.reading_service is not None and isinstance(self.reading_service, CheckpointableReadingServiceInterface):
            reading_service_state = self.reading_service.checkpoint()

        # Serialize datapipe after applying adapters and before reading service adaption
        serialized_datapipe = serialize_datapipe(self._datapipe_before_reading_service_adapt)

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
        Create new ``DataLoader2`` with ``DataPipe`` graph and ``ReadingService`` restored
        from the serialized state.
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

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        For the existing ``DataLoader2``, load serialized state to restore ``DataPipe`` graph
        and reset the internal state of ``ReadingService``.
        """
        # edge case checking
        # iterator has already been created: 1) iterator is just created 2) iterator is created and iter is exhausted
        if self._datapipe_iter is not None:
            raise RuntimeError(
                "DataLoaderV2 iterator has already been created, `load_state_dict()` can’t be called. "
                "Please create a new dataloader in order to use load state dict."
            )

        serialized_datapipe = state_dict[SERIALIZED_DATAPIPE_KEY_NAME]
        reading_service_state = state_dict[READING_SERVICE_STATE_KEY_NAME]

        # deserialize datapipe
        deserialized_datapipe = deserialize_datapipe(serialized_datapipe)
        assert deserialized_datapipe is not None

        # override existing datapipe and reading service state
        self.datapipe = deserialized_datapipe
        self.reading_service_state = reading_service_state

        # re-initialize datapipe_adapter_fn and _datapipe_before_reading_service_adapt
        if self.datapipe_adapter_fns is not None:
            for adapter_fn in self.datapipe_adapter_fns:
                self.datapipe = adapter_fn(self.datapipe)
        self._datapipe_before_reading_service_adapt = self._copy(self.datapipe)
