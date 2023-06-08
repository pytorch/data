# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as py_mp
import pickle
import warnings

from abc import ABC, abstractmethod
from datetime import timedelta
from functools import partial
from multiprocessing.queues import Queue
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from torchdata._constants import default_dl2_worker_join_timeout_in_s, default_timeout_in_s
from torchdata.dataloader2 import communication
from torchdata.dataloader2.graph import DataPipe, list_dps, replace_dp, set_graph_random_seed, traverse_dps
from torchdata.dataloader2.graph._serialization import attach_wrapper
from torchdata.dataloader2.graph.utils import _find_replicable_branches
from torchdata.dataloader2.random import dist_share_seed, SeedGenerator
from torchdata.dataloader2.utils import process_init_fn, WorkerInfo
from torchdata.dataloader2.utils.dispatch import _DummyIterDataPipe, find_lca_round_robin_sharding_dp
from torchdata.datapipes.iter import FullSync


class ReadingServiceInterface(ABC):
    r"""
    Interface for ``ReadingService``. Please extend custom ``ReadingService`` based on this interface class.

    ReadingService must be picklable prior to ``initialize`` being called. This is because a copy of it will be
    created by ``DataLoader2`` to avoid the situation where the same ReadingService object is used by
    multiple ``DataLoader2``, and its internal state will be modifiable by each of them.

    As a result of this constraint, certain initialization steps may need to take place within the
    ``initialize`` method rather than ``__init__`` of the ReadingService class.
    """

    @abstractmethod
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        r"""
        ``ReadingService`` takes a ``DataPipe`` graph, adapts it into a new ``DataPipe`` graph based on the custom need.
        Called once in creating ``DataLoader2`` iterator at first time. Prior to calling this method,
        the ``ReadingService`` object must be picklable.

        Args:
            datapipe: Original ``DataPipe`` graph.

        Return:
            An adapted or a new ``DataPipe`` graph.
        """
        pass

    def finalize(self) -> None:
        r"""
        ``ReadingService`` cleans up internal states and fully shuts down the service.
        Called in ``DataLoader2``'s ``shutdown`` and ``__del__``.
        """
        pass

    def initialize_iteration(
        self, seed_generator: SeedGenerator, iter_reset_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        r"""
        ``ReadingService`` spins up service for an epoch. Called at the beginning
        of every time getting ``DataLoader2`` iterator.

        Args:
            seed_generator: SeedGenerator object created and managed by DataLoader2. As the single
                source of randomness, it will govern the determinism for all of random operations
                with the graph of DataPipes.
            iter_reset_fn: Optional reset function from the prior ``ReadingServcie``
                when ``SequentialReadingService`` chains multiple ``ReadingServices``

        Returns:
            A new ``iter_reset_fn`` to be used by subseqeuent ``ReadingService``

        Example:
            MultiProcessingReadingService starts setting worker seeds per process and prefetching
            items from the graph.
        """
        pass

    def finalize_iteration(self) -> None:
        r"""
        ``ReadingService`` ends service after an epoch is finished. Called when
        the iterator of ``DataLoader2`` is depleted.
        """
        pass

    def __del__(self):
        # Due to non-deterministic order of destruction, by the time `finalize` is called,
        # some objects may already be `None`.
        try:
            self.finalize()
        except AttributeError:
            pass


class CheckpointableReadingServiceInterface(ReadingServiceInterface):
    r"""
    Extend ``ReadingServiceInterface`` with two additional methods to save/restore the state of the data-processing graph.
    """

    @abstractmethod
    def checkpoint(self) -> bytes:
        """
        ``ReadingService`` serializes the internal states. Called in ``DataLoader2.state_dict``.
        """
        pass

    @abstractmethod
    def restore(self, datapipe: DataPipe, serialized_state: bytes) -> DataPipe:
        """
        ``ReadingService`` adapts ``DataPipe`` graph based on the serialized state.
        Called once in creating ``DataLoader2`` iterator at first time.
        Counterpart of ``initialize``, which adapt ``DataPipe`` graph from scratch.

        Args:
            datapipe: original ``DataPipe`` graph before adapted by ``ReadingService``
            serialized_state: The serialized state of internal state used to restore the state
                of the adapted ``DataPipe`` graph.

        Returns:
            Adapted ``DataPipe`` generated from the serialized state.
        """
        pass


def _collate_no_op(batch):
    return batch[0]


class PrototypeMultiProcessingReadingService(ReadingServiceInterface):
    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "`PrototypeMultiProcessingReadingService` is deprecated and will be removed in TorchData 0.8. "
            "Please use `MultiProcessingReadingService`."
        )
        return MultiProcessingReadingService(*args, **kwargs)


class InProcessReadingService(ReadingServiceInterface):
    r"""
    Default ReadingService to serve the ``DataPipe` graph in the main process,
    and apply graph settings like determinism control to the graph.

    Args:
        prefetch_cnt: (int, 0 by default): Number of data will be prefetched in the main process.
        init_fn: (Callable, optional): Custom function to be called when the main
            process starts to iterate over ``DataPipe`` graph.
        reset_fn: (Callable, optional): Custom function to be called at the beginning
            of each epoch with ``DataPipe``, ``WorkerInfo`` and ``SeedGenerator``
            as the expected arguments.
    """
    _prefetch_cnt: int
    _init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]]
    _reset_fn: Optional[Callable[[DataPipe, WorkerInfo, SeedGenerator], DataPipe]]
    _end_datapipe: Optional[DataPipe]

    def __init__(
        self,
        prefetch_cnt: int = 0,
        init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]] = None,
        reset_fn: Optional[Callable[[DataPipe, WorkerInfo, SeedGenerator], DataPipe]] = None,
    ) -> None:
        self._prefetch_cnt = prefetch_cnt
        self._init_fn = init_fn
        self._reset_fn = reset_fn
        self._end_datapipe = None

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        worker_info = WorkerInfo(1, 0)
        datapipe = process_init_fn(datapipe, worker_info, self._init_fn)
        self._end_datapipe = datapipe
        return datapipe

    def initialize_iteration(
        self, seed_generator: SeedGenerator, iter_reset_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        assert self._end_datapipe is not None

        # Set random seeds for DataPipe that are in the main process (NOT those in worker processes)
        # Worker seeds are set in `process_reset_fn`
        set_graph_random_seed(self._end_datapipe, seed_generator)

        return None

    def _pause(
        self, pause_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        """
        Pauses DataPipes' activities in the main process in order to collect state.
        """
        assert self._end_datapipe is not None

        dp_list = list_dps(traverse_dps(self._end_datapipe))
        for dp in dp_list:
            if hasattr(dp, "pause") and callable(dp.pause):
                dp.pause()
        return None

    def _resume(
        self, resume_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        """
        Resumes DataPipes' activities. This is required to be called after `_pause` before
        the DataLoader can keep yielding elements.
        """
        assert self._end_datapipe is not None

        dp_list = list_dps(traverse_dps(self._end_datapipe))
        # Reversed order
        for dp in dp_list[::-1]:
            if hasattr(dp, "resume") and callable(dp.resume):
                dp.resume()
        return None

    def _limit(
        self, num_batches: Optional[int], limit_fn: Optional[Callable[[DataPipe, Optional[int]], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe, Optional[int]], DataPipe]]:
        r"""
        Apply limit_fn to the DataPipe graph.
        """
        if limit_fn is not None:
            # TODO: Remove when flexible checkpoint is supported
            limit_fn(self._end_datapipe, num_batches)  # type: ignore[arg-type]
        return None


class MultiProcessingReadingService(ReadingServiceInterface):
    r"""
    Spawns multiple worker processes to load data from the ``DataPipe`` graph.
    If any non-replicable ``DataPipe`` (``sharding_round_robin_dispatch``) is presented in the graph,
    a separate dispatching process will be created to load data from the lowest common ancestor
    of all non-replicable ``DataPipes`` and distributes data to each worker process in the round-robin manner
    Then, the subsequent ``DataPipe`` graph in each worker process will process the data from the dispatching
    process and eventually return the result to the main process.

    Args:
        num_workers (int): How many subprocesses to use for data loading.
        multiprocessing_context (str, optional): Multiprocessing starting method.
            If method is None then the default context is returned.
            Otherwise, method should be 'fork', 'spawn'.
        worker_prefetch_cnt: (int, 10 by default): Number of data will be prefetched at
            the end of each worker process.
        main_prefetch_cnt: (int, 10 by default): Number of data will be prefetched
            at the end of the whole pipeline in the main process.
        worker_init_fn: (Callable, optional): Function to be called when each worker
            process launches with ``DataPipe`` and ``WorkerInfo`` as the expected arguments.
        worker_reset_fn: (Callable, optional): Function to be called at the beginning
            of each epoch in each worker process with ``DataPipe``, ``WorkerInfo``
            and ``SeedGenerator`` as the expected arguments.
    """
    num_workers: int
    multiprocessing_context: Optional[str]
    worker_prefetch_cnt: int
    main_prefetch_cnt: int
    worker_init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]]
    worker_reset_fn: Optional[Callable[[DataPipe, WorkerInfo, SeedGenerator], DataPipe]]
    _worker_processes: List[Tuple[py_mp.process.BaseProcess, Queue, Queue]]
    _dispatch_process: Optional[Tuple[py_mp.process.BaseProcess, List[Queue], List[Queue]]]
    _worker_datapipes: List[DataPipe]
    _worker_consumer_datapipe: Optional[DataPipe]
    _main_prefetch_datapipe: Optional[DataPipe]
    _end_datapipe: Optional[DataPipe]
    _mp: bool
    _finalized: bool = False

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context: Optional[str] = None,
        worker_prefetch_cnt: int = 10,
        main_prefetch_cnt: int = 10,
        worker_init_fn: Optional[Callable[[DataPipe, WorkerInfo], DataPipe]] = None,
        worker_reset_fn: Optional[Callable[[DataPipe, WorkerInfo, SeedGenerator], DataPipe]] = None,
    ) -> None:
        if num_workers == 0:
            warnings.warn("Please use `InProcessReadingService` for num_workers=0")
        self.num_workers = num_workers

        if multiprocessing_context is not None:
            _all_start_methods = mp.get_all_start_methods()
            assert (
                multiprocessing_context in _all_start_methods
            ), f"Please choose one available multiprocessing context from {_all_start_methods}"
        self.multiprocessing_context = multiprocessing_context
        self.worker_prefetch_cnt = worker_prefetch_cnt
        self.main_prefetch_cnt = main_prefetch_cnt
        self.worker_init_fn = worker_init_fn
        self.worker_reset_fn = worker_reset_fn
        self._worker_processes = []
        self._dispatch_process = None
        self._worker_datapipes = []
        self._worker_consumer_datapipe = None
        self._main_prefetch_datapipe = None
        self._end_datapipe = None
        self._mp = num_workers > 0

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        r"""
        ``MultiProcessingReadingService`` finds information about sharding,
        separates graph by multiple pieces and reconnects it using queues.
        creates subprocesses.
        """
        if not self._mp:
            # TODO(616): Warn and recommend usage of InProcessReadingService
            worker_info = WorkerInfo(1, 0)
            datapipe = process_init_fn(datapipe, worker_info, self.worker_init_fn)
            self._end_datapipe = datapipe
            return datapipe

        ctx = mp.get_context(self.multiprocessing_context)

        # Launch dispatching process for the lowest common ancestor of non-replicable DataPipes
        graph = traverse_dps(datapipe)
        dispatching_dp = find_lca_round_robin_sharding_dp(graph)
        # TODO(ejguan): When the last DataPipe is round_robin_sharding, use InPrcoessReadingService
        if dispatching_dp is not None:
            dummy_dp = _DummyIterDataPipe()
            graph = replace_dp(graph, dispatching_dp, dummy_dp)  # type: ignore[arg-type]
            datapipe = list(graph.values())[0][0]
            # TODO(ejguan): Determine buffer_size at runtime or use unlimited buffer
            round_robin_dps = dispatching_dp.round_robin_demux(num_instances=self.num_workers)
            # TODO(ejguan): Benchmark if we need to prefetch in dispatching process
            worker_info = WorkerInfo(self.num_workers, 0)
            process, req_queues, res_queues = communication.eventloop.CreateProcessForMultipleDataPipelines(
                ctx,
                round_robin_dps,
                process_name="dispatching process",
                worker_info=worker_info,
                custom_reset_fn=self.worker_reset_fn,
            )
            assert len(req_queues) == self.num_workers and len(res_queues) == self.num_workers
            for req_queue in req_queues:
                req_queue.cancel_join_thread()
            for res_queue in res_queues:
                res_queue.cancel_join_thread()
            process.daemon = True
            process.start()
            self._dispatch_process = (process, req_queues, res_queues)

        # Find replicable branches for worker processes
        # The rest of non-replicable DataPipes will remain in the main process
        replicable_dps = _find_replicable_branches(graph)
        assert (
            len(replicable_dps) == 1
        ), "MultiProcessingReadingService only supports single replicable branch currently"
        replicable_dp = replicable_dps[0]
        replicable_dp = attach_wrapper(replicable_dp)

        for worker_id in range(self.num_workers):
            worker_info = WorkerInfo(self.num_workers, worker_id)
            # Dispatching process for non-replicable DataPipes exists
            dispatching_req_queue = None if self._dispatch_process is None else self._dispatch_process[1][worker_id]
            dispatching_res_queue = None if self._dispatch_process is None else self._dispatch_process[2][worker_id]
            call_on_process_init = partial(
                process_init_fn,
                worker_info=worker_info,
                custom_init_fn=self.worker_init_fn,
                worker_prefetch_cnt=self.worker_prefetch_cnt,
                dispatching_req_queue=dispatching_req_queue,
                dispatching_res_queue=dispatching_res_queue,
            )
            (process, req_queue, res_queue) = communication.eventloop.CreateProcessForDataPipeline(
                ctx,
                replicable_dp,
                process_name="worker process",
                worker_info=worker_info,
                call_on_process_init=call_on_process_init,
                custom_reset_fn=self.worker_reset_fn,
            )
            req_queue.cancel_join_thread()
            process.daemon = True
            process.start()
            self._worker_processes.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
            )
            self._worker_datapipes.append(local_datapipe)

        end_datapipe = communication.iter._IterateQueueDataPipes(self._worker_datapipes)  # type: ignore[assignment]
        self._worker_consumer_datapipe = end_datapipe

        if self.main_prefetch_cnt > 0:
            end_datapipe = self._worker_consumer_datapipe.prefetch(self.main_prefetch_cnt)  # type: ignore[union-attr]
            self._main_prefetch_datapipe = end_datapipe

        # Attach non-replicable DataPipes
        if replicable_dps[0] is not datapipe:
            graph = replace_dp(graph, replicable_dps[0], end_datapipe)
            end_datapipe = datapipe  # type: ignore[assignment]

        self._end_datapipe = end_datapipe
        assert self._end_datapipe is not None

        return self._end_datapipe  # type: ignore[return-value]

    def initialize_iteration(
        self, seed_generator: SeedGenerator, iter_reset_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        assert self._end_datapipe is not None

        # Set random seeds for DataPipe that are in the main process (NOT those in worker processes)
        # Worker seeds are set in `process_reset_fn`
        set_graph_random_seed(self._end_datapipe, seed_generator)

        if self._mp:
            if self.main_prefetch_cnt > 0:
                # Stop prefetching first
                self._main_prefetch_datapipe.reset()  # type: ignore[union-attr]
            # Send the shared seed to subprocesses
            assert self._worker_consumer_datapipe is not None
            self._worker_consumer_datapipe.reset_epoch(seed_generator, iter_reset_fn)
        # In-process (num_workers == 0)
        else:
            # Technically speaking, we should call `_process_reset_fn` to reset global RNGs
            # for data-related operations. However, it would pollute the state of global RNGs
            # (random, torch and numpy), if users have already seeded them in the main process
            # TODO(ejguan): This should be fixed by adding a method to isolate global RNGs
            pass
        return None

    def finalize(self) -> None:
        r"""
        ``MultiProcessingReadingService`` invalidate states & properly exits all subprocesses.
        """
        if self._finalized:
            return
        self._finalized = True

        # TODO(618): Check if anyone stuck with messages
        # Clean up worker processes
        if self.num_workers > 0:
            self._worker_consumer_datapipe.request_terminate()  # type: ignore[union-attr]
        for process, req_queue, _ in self._worker_processes:
            try:
                process.join(default_dl2_worker_join_timeout_in_s)
            except TimeoutError:
                pass
            req_queue.close()

        # Clean up dispatching process
        if self._dispatch_process is not None:
            try:
                self._dispatch_process[0].join(default_dl2_worker_join_timeout_in_s)
            except TimeoutError:
                pass
            for req_queue in self._dispatch_process[1]:
                req_queue.close()

        self._worker_processes = []
        self._dispatch_process = None

    def _pause(
        self, pause_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        r"""
        Pauses DataPipes' activities such as prefetching within main/worker/dispatching processes,
        in order to collect state. The provided ``pause_fn`` will be executed in
        worker/dispatching processes.
        """
        if self.num_workers == 0:
            raise RuntimeError(
                "If you would like to use `pause` with `MultiProcessingReadingService`, "
                "please use more than 0 worker."
            )
        assert self._end_datapipe is not None
        # Call pause for DataPipes in the main process (e.g. prefetch, fullsync)
        dp_list = list_dps(traverse_dps(self._end_datapipe))
        for dp in dp_list:
            if hasattr(dp, "pause") and callable(dp.pause):
                dp.pause()
        self._worker_consumer_datapipe.request_pause(pause_fn)  # type: ignore[union-attr]
        return None

    def _resume(
        self, resume_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        r"""
        Resumes DataPipes' activities. This is required to be called after `_pause` before
        the DataLoader can keep yielding elements.
        """
        if self.num_workers > 0:
            self._worker_consumer_datapipe.request_resume(resume_fn)  # type: ignore[union-attr]
        else:
            raise RuntimeError(
                "If you would like to use `resume` with `MultiProcessingReadingService`, "
                "please use more than 0 worker."
            )
        assert self._end_datapipe is not None
        # Call resume for DataPipes in the main process (e.g. prefetch, fullsync)
        dp_list = list_dps(traverse_dps(self._end_datapipe))
        for dp in dp_list[::-1]:
            if hasattr(dp, "resume") and callable(dp.resume):
                dp.resume()
        return None

    def _limit(
        self, num_batches: Optional[int], limit_fn: Optional[Callable[[DataPipe, Optional[int]], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe, Optional[int]], DataPipe]]:
        r"""
        Send limit_fn to worker/dispatching process to set the limit number to the specified DataPipes.
        """
        if limit_fn is not None:
            # Only propogate limit when dispatching process exists
            num_batches = None if self._dispatch_process is None else num_batches
            self._worker_consumer_datapipe.request_limit(num_batches, limit_fn)  # type: ignore[union-attr]
            # TODO: Remove when flexible checkpoint is supported
            limit_fn(self._end_datapipe, num_batches)  # type: ignore[arg-type]
        return None


class DistributedReadingService(ReadingServiceInterface):
    r"""
    ``DistributedReadingSerivce`` handles distributed sharding on the graph of ``DataPipe`` and
    guarantee the randomness by sharing the same seed across the distributed processes.

    Args:
        timeout: Timeout for operations executed against the process group in seconds.
            Default value equals 30 minutes.
    """

    def __init__(self, timeout: int = default_timeout_in_s):
        if not dist.is_available():
            raise RuntimeError("Torch Distributed is required to be available")
        self._world_size: int = 1
        self._rank: int = 0
        self._datapipe: Optional[DataPipe] = None
        self._timeout: int = timeout
        self._pg: Optional[dist.ProcessGroup] = None

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        r"""
        Launches the ``gloo``-backend distributed process group. Carries out distributed sharding
        on the graph of ``DataPipe`` and returns the graph attached with a ``FullSyncIterDataPipe``
        at the end.
        """
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Torch Distributed is required to be initialized")
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._pg = dist.new_group(backend="gloo", timeout=timedelta(seconds=self._timeout))
        torch.utils.data.graph_settings.apply_sharding(
            datapipe, self._world_size, self._rank, SHARDING_PRIORITIES.DISTRIBUTED
        )
        # Only append FullSyncIterDataPipe if it's not presented at the end of the pipeline
        if not isinstance(datapipe, FullSync):
            datapipe = datapipe.fullsync(self._timeout)
        self._datapipe = datapipe
        return datapipe

    def initialize_iteration(
        self, seed_generator: SeedGenerator, iter_reset_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        r"""
        Shares the same seed from rank 0 to other ranks across the distributed processes
        and apply the random seed to the ``DataPipe`` graph.
        """
        assert self._datapipe is not None

        shared_seed = dist_share_seed(seed_generator.generate_shared_seed(), self._pg)
        seed_generator.seed(shared_seed)
        seed_generator = seed_generator.spawn(self._rank, inplace=True)
        set_graph_random_seed(self._datapipe, seed_generator)
        return None

    def finalize(self) -> None:
        r"""
        Clean up the distributed process group.
        """
        if self._pg is not None:
            dist.destroy_process_group(self._pg)
            self._pg = None


class SequentialReadingService(CheckpointableReadingServiceInterface):
    def __init__(self, *reading_services):
        self.reading_services = reading_services

    # Sequential Order
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        for rs in self.reading_services:
            datapipe = rs.initialize(datapipe)
        return datapipe

    # Reversed Order
    def finalize(self) -> None:
        for rs in reversed(self.reading_services):
            rs.finalize()

    # Sequential Order
    def initialize_iteration(
        self, seed_generator: SeedGenerator, iter_reset_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        chained_iter_reset_fn = iter_reset_fn
        for rs in self.reading_services:
            chained_iter_reset_fn = rs.initialize_iteration(
                seed_generator=seed_generator, iter_reset_fn=chained_iter_reset_fn
            )
        return chained_iter_reset_fn

    # Reversed Order
    def finalize_iteration(self) -> None:
        for rs in reversed(self.reading_services):
            rs.finalize_iteration()

    # Sequential Order
    def checkpoint(self) -> bytes:
        states = []
        for rs in self.reading_services:
            if hasattr(rs, "checkpoint") and callable(rs.checkpoint):
                states.append(rs.checkpoint())
            else:
                warnings.warn(f"{rs} doesn't support `checkpoint`, skipping...")
                states.append(b"")
        return pickle.dumps(states)

    # Sequential Order, to align with initialize
    def restore(self, datapipe, serialized_state: bytes) -> DataPipe:
        states = pickle.loads(serialized_state)
        assert len(states) == len(self.reading_services)
        for rs, state in zip(self.reading_services, states):
            if hasattr(rs, "restore") and callable(rs.restore):
                datapipe = rs.restore(datapipe, state)
            else:
                warnings.warn(f"{rs} doesn't support `restore` from state, initialize from scratch")
                datapipe = rs.initialize(datapipe)
        return datapipe

    def _pause(
        self, pause_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        r"""
        Pause the ``DataPipe`` graph defined in all ``ReadingServices``. For example of
        ``MultiProcessingReadingService`` would accept a ``pause_fn`` from a prior ``ReadingService``
        to execute custom pause logic within worker/dispatching processes.
        """
        for rs in self.reading_services:
            if hasattr(rs, "_pause"):
                pause_fn = rs._pause(pause_fn)
        return pause_fn

    def _resume(
        self, resume_fn: Optional[Callable[[DataPipe], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe], DataPipe]]:
        r"""
        Resume the ``DataPipe`` graph defined in all ``ReadingServices``. For example of
        ``MultiProcessingReadingService`` would accept a ``resume_fn`` from a prior ``ReadingService``
        to execute custom resume logic within worker/dispatching processes.
        """
        for rs in self.reading_services:
            if hasattr(rs, "_resume"):
                resume_fn = rs._resume(resume_fn)
        return resume_fn

    def _limit(
        self, num_batches: Optional[int], limit_fn: Optional[Callable[[DataPipe, Optional[int]], DataPipe]] = None
    ) -> Optional[Callable[[DataPipe, Optional[int]], DataPipe]]:
        r"""
        Limit the ``DataPipe`` graph defined in all ``ReadingServices``. For example of
        ``MultiProcessingReadingService`` would accept a ``limit_fn`` from a prior ``ReadingService``
        to set limit to ``DataPipes` within worker/dispatching processes.
        """
        for rs in self.reading_services:
            if hasattr(rs, "_limit"):
                limit_fn = rs._limit(num_batches, limit_fn)
        return limit_fn
