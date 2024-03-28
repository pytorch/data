r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import random
import copy
import os
import pickle
import queue
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

from torch._utils import ExceptionWrapper
from typing import Optional, Union, TYPE_CHECKING
from torch.utils.data._utils import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS, HAS_NUMPY
from torch.utils.data._utils.worker import WorkerInfo, _generate_state
if TYPE_CHECKING:
    from torch.utils.data import Dataset

if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog:
        def __init__(self):
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
else:
    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


# def get_worker_info() -> Optional[WorkerInfo]:
#     r"""Returns the information about the current
#     :class:`~torch.utils.data.DataLoader` iterator worker process.

#     When called in a worker, this returns an object guaranteed to have the
#     following attributes:

#     * :attr:`id`: the current worker id.
#     * :attr:`num_workers`: the total number of workers.
#     * :attr:`seed`: the random seed set for the current worker. This value is
#       determined by main process RNG and the worker id. See
#       :class:`~torch.utils.data.DataLoader`'s documentation for more details.
#     * :attr:`dataset`: the copy of the dataset object in **this** process. Note
#       that this will be a different object in a different process than the one
#       in the main process.

#     When called in the main process, this returns ``None``.

#     .. note::
#        When used in a :attr:`worker_init_fn` passed over to
#        :class:`~torch.utils.data.DataLoader`, this method can be useful to
#        set up each worker process differently, for instance, using ``worker_id``
#        to configure the ``dataset`` object to only read a specific fraction of a
#        sharded dataset, or use ``seed`` to seed other libraries used in dataset
#        code.
#     """
#     return _worker_info


# r"""Dummy class used to signal the end of an IterableDataset"""
# @dataclass(frozen=True)
# class _IterableDatasetStopIteration:
#     worker_id: int

# r"""Dummy class used to resume the fetching when worker reuse is enabled"""
# @dataclass(frozen=True)
# class _ResumeIteration:
#     seed: Optional[int] = None

from torch.utils.data._utils.worker import _IterableDatasetStopIteration, _ResumeIteration

r"""Dummy class used to signal StateRequest"""
@dataclass(frozen=True)
class _StateRequest:
    uuid: str
    worker_id: int

r"""Dummy class used to signal StateRequestComplete"""
@dataclass(frozen=True)
class _StateRequestComplete:
    uuid: str

r"""Dummy class used to signal LoadStateRequest"""
@dataclass(frozen=True)
class _LoadStateRequest:
    uuid: str
    worker_id: int
    state_dict: Dict[str, Any]

r"""Dummy class used to signal LoadStateRequest"""
@dataclass(frozen=True)
class _LoadStateRequestComplete:
    uuid: str
    worker_id: int


def try_to_serialize(obj: Any, pickle: bool) -> Union[dict, bytes, None]:
    if hasattr(obj, "state_dict") and hasattr(obj, "load_state_dict"):
        obj_state = obj.state_dict()
    elif pickle:
        try:
            obj_state = pickle.dumps(obj)
        except Exception:
            # If pickle fails, we fall back to copying the state
            # dict. This is not ideal, but it's better than
            # failing entirely.
            obj_state = None
    else:
        obj_state = None

    return obj_state


def try_to_deserialize(obj: Any, serialized: Union[dict, bytes, None]) -> Union[Any, None]:
    if hasattr(obj, "load_state_dict"):
        obj.load_state_dict(serialized)
        return obj
    else:
        try:
            obj = pickle.loads(serialized)
        except:
            obj = None
    return obj


def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                 num_workers, persistent_workers, shared_seed):
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    try:
        # Initialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
        # module's handlers are executed after Python returns from C low-level
        # handlers, likely when the same fatal signal had already happened
        # again.
        # https://docs.python.org/3/library/signal.html#execution-of-python-signal-handlers
        signal_handling._set_worker_signal_handlers()

        torch.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        if HAS_NUMPY:
            np_seed = _generate_state(base_seed, worker_id)
            import numpy as np
            np.random.seed(np_seed)

        from torch.utils.data import IterDataPipe
        from torch.utils.data.graph_settings import apply_random_seed

        shared_rng = torch.Generator()
        if isinstance(dataset, IterDataPipe):
            assert shared_seed is not None
            shared_rng.manual_seed(shared_seed)
            dataset = apply_random_seed(dataset, shared_rng)

        # global _worker_info
        torch.utils.data._utils.worker._worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset)

        from torch.utils.data import _DatasetKind

        init_exception = None

        fetcher = None
        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)

            if HAS_NUMPY:
                numpy_state = np.random.get_state()
            else:
                numpy_state = None

            if dataset_kind == _DatasetKind.Iterable:
                fetcher_state = {
                    "dataset_iter": try_to_serialize(fetcher.dataset_iter, False),
                    "ended": fetcher.ended,
                }
            else:
                fetcher_state = None
            # Pick up any user-defined dataset state, for both map/iterable style datasets
            dataset_state = try_to_serialize(dataset, False)
            initial_state_dict = copy.deepcopy({
                "worker_id": worker_id,
                "fetcher_state": fetcher_state,
                "dataset_state": dataset_state,
                "shared_rng": shared_rng.get_state(),
                "random_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": numpy_state,
                "iteration_end": False,
            })
        except Exception:
            init_exception = ExceptionWrapper(
                where=f"in DataLoader worker process {worker_id}")

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _LoadStateRequest):
                # Restore state from r.state_dict
                # index_buffer.clear()
                if r.state_dict == "skip":  # TODO: is this used?
                    state_dict = copy.deepcopy(initial_state_dict)
                else:
                    state_dict = r.state_dict

                if state_dict["fetcher_state"] is not None:
                    fetcher.dataset_iter = try_to_deserialize(fetcher.dataset_iter, state_dict["fetcher_state"]["dataset_iter"])
                    fetcher.ended = state_dict["fetcher_state"]["ended"]
                if state_dict["dataset_state"] is not None:
                    dataset = try_to_deserialize(dataset, state_dict["dataset_state"])
                shared_rng.set_state(state_dict["shared_rng"])
                random.setstate(state_dict["random_rng_state"])
                torch.set_rng_state(state_dict["torch_rng_state"])
                if HAS_NUMPY:
                    np.random.set_state(state_dict["numpy_rng_state"])
                # iteration_end = state_dict["iteration_end"]
                iteration_end = False

                data_queue.put((_LoadStateRequestComplete(r.uuid, r.worker_id), None))
                del state_dict
                continue
            elif isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    assert r.seed is not None
                    shared_rng.manual_seed(r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                # Recreate the fetcher for worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last)
                continue
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, (index, snapshot) = r
            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            state_dict = None
            if init_exception is not None:
                data = init_exception
                # init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
                except Exception as e:
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(
                            where=f"in DataLoader worker process {worker_id}")
                if snapshot or iteration_end:
                    # always generate snapshot when Iterable raises StopIteration.
                    if HAS_NUMPY:
                        numpy_state = np.random.get_state()
                    else:
                        numpy_state = None

                    if dataset_kind == _DatasetKind.Iterable:
                        fetcher_state = {
                            "dataset_iter": try_to_serialize(fetcher.dataset_iter, False),
                            "ended": fetcher.ended,
                        }
                    else:
                        fetcher_state = None
                    # Pick up any user-defined dataset state, for both map/iterable style datasets
                    dataset_state = try_to_serialize(dataset, False)
                    state_dict = {
                        "worker_id": worker_id,
                        "fetcher_state": fetcher_state,
                        "dataset_state": dataset_state,
                        "shared_rng": shared_rng.get_state(),
                        "random_rng_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(),
                        "numpy_rng_state": numpy_state,
                    }
            data_queue.put((idx, (data, worker_id, state_dict)))
            del data, idx, index, r, state_dict  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


torch.utils.data._utils.worker._worker_loop = _worker_loop
