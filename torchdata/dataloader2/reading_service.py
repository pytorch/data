# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import multiprocessing as mp

# import os
import time
from abc import ABC, abstractmethod

from collections import deque

from datetime import timedelta
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
import torchdata.dataloader2.graph as graph
from torch.utils.data import DataLoader
from torch.utils.data.graph import DataPipe

from torchdata._constants import default_timeout_in_s

from torchdata.dataloader2 import communication
from torchdata.datapipes.iter import FullSync, IterableWrapper, ShardingFilter


class ReadingServiceInterface(ABC):
    @abstractmethod
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        """
        ReadingService traverses datapipe graph, finds executable part,
        adapts into its own datapipe, and replaces in datapipe graph.

        Called once in creating DataLoader iterator at first time.

        Args:
            datapipe: DataPipe. Original datapipe.

        Return:
            Adapted DataPipe.

        Example:
            MultiProcessingReadingService finds information about sharding,
            separates graph by multiple pieces and reconnects it using queues.
            Spawns processes/threads.
        """
        pass

    def finalize(self) -> None:
        """
        ReadingService cleanup states.
        Called in DataLoader shutdown and __del__

        Example:
            MultiProcessingReadingService invalidate states & handle persistent worker.
        """
        pass

    def initialize_iteration(self) -> None:
        """
        ReadingService spin up service.
        Called at the beginning of every time getting DataLoader iterator.

        Example:
            MultiProcessingReadingService starts prefetching items from the graph.
        """
        pass

    def finalize_iteration(self) -> None:
        """
        ReadingService end service.

        Example:
            MultiprocessingReadingService cleans up processes.
        """
        pass


class CheckpointableReadingServiceInterface(ReadingServiceInterface):
    @abstractmethod
    def checkpoint(self) -> bytes:
        """
        ReadingService serialize backend states.
        Called in DataLoader checkpoint.
        """

        pass

    @abstractmethod
    def restore(self, datapipe: DataPipe, serialized_state: bytes) -> DataPipe:
        """
        ReadingService adapts datapipe and consume serialized state.

        Called once in creating DataLoader iterator at first time.
        Counterpart of `initialize`, which adapt datapipe from scratch.

        Returns:
            Adapted IterDataPipe.
        """
        pass


def _collate_no_op(batch):
    return batch[0]


class _IterateQueueDataPipes_OLD:
    def __init__(self, datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        self.reset()
        # TODO(612): This is slow as it does not sends data requests ahead.
        exclude_datapipes: List[Any] = []
        while len(exclude_datapipes) < len(self.datapipes):
            for dp in self.datapipes:
                if dp not in exclude_datapipes:
                    forever = True
                    while forever:
                        try:
                            value = dp.nonblocking_next()
                            yield value
                            forever = False
                        except StopIteration:
                            exclude_datapipes.append(dp)
                            forever = False
                        except communication.iter.NotAvailable:
                            time.sleep(0.00001)

    def reset(self):
        # NonBlocking DataPipes do not reset automatically, have to do it manually
        for dp in self.datapipes:
            dp.reset_iterator()


class _IterateQueueDataPipes:
    def __init__(self, datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        self.reset()
        # TODO(612): This is hacky as it access .protocol field of the pipes also doesn't check protocols status (yet)
        total_pipes = len(self.datapipes)
        disabled_pipe = [False] * len(self.datapipes)
        cnt_disabled_pipes = 0

        for idx in range(total_pipes):
            self.datapipes[idx].protocol.request_next()

        while cnt_disabled_pipes < total_pipes:
            for idx in range(total_pipes):
                if not disabled_pipe[idx]:
                    response = self.datapipes[idx].protocol.get_response_next(block=True)
                    if isinstance(response, communication.messages.StopIterationResponse):
                        disabled_pipe[idx] = True
                        cnt_disabled_pipes += 1
                        break
                    if isinstance(response, communication.messages.InvalidStateResponse):
                        raise communication.iter.InvalidStateResetRequired
                    if isinstance(response, communication.messages.TerminateResponse):
                        raise communication.iter.TerminateRequired
                    self.datapipes[idx].protocol.request_next()
                    yield response.value

    def reset(self):
        # print(os.getpid(), "ressreting owned non blcked datapipes")
        # NonBlocking DataPipes do not reset automatically, have to do it manually
        for dp in self.datapipes:
            dp.reset_iterator()


class _IterateQueueDataPipes_New:
    def __init__(self, datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        self.reset()
        # TODO(612): This is slow as it does not sends data requests ahead.

        datapipes_count = len(self.datapipes)
        pre_pipe_buffer_len = 10
        buffers = [deque([], pre_pipe_buffer_len) for _ in range(datapipes_count)]
        idx = 0
        completed_datapipes = [False] * datapipes_count
        active_datapipes = datapipes_count
        buffer_elements = 0

        # print('initial buffer', len(buffers[0]))

        while active_datapipes or buffer_elements > 0:
            if len(buffers[idx]):
                value = buffers[idx].popleft()
                yield value
                idx = (idx + 1) % datapipes_count
            else:
                if completed_datapipes[idx]:
                    idx = (idx + 1) % datapipes_count
                else:
                    prefetched = 0
                    for i in range(datapipes_count * 2):
                        new_idx = (idx + i) % datapipes_count
                        if not completed_datapipes[new_idx]:
                            if len(buffers[new_idx]) < pre_pipe_buffer_len:

                                try:
                                    value = self.datapipes[new_idx].nonblocking_next()
                                    buffers[new_idx].append(value)
                                    if new_idx == idx:
                                        break
                                    prefetched += 1
                                except StopIteration:
                                    completed_datapipes[new_idx] = True
                                    active_datapipes -= 1
                                except communication.iter.NotAvailable:
                                    pass

                    # if prefetched:
                    #     print('prefetched', prefetched)
                    time.sleep(0.0000001)

    def reset(self):
        # print(os.getpid(), "ressreting owned non blcked datapipes")
        # NonBlocking DataPipes do not reset automatically, have to do it manually
        for dp in self.datapipes:
            dp.reset_iterator()


class PrototypeMultiProcessingReadingService(ReadingServiceInterface):
    num_workers: int
    processes: List
    datapipes: List

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context=None,
        post_adapter_fn=None,
    ) -> None:
        self.num_workers = num_workers
        # TODO(613): Should be one of 'fork', 'spawn'
        self.multiprocessing_context = multiprocessing_context
        self.processes = []
        self.datapipes = []
        self.post_adapter_fn = post_adapter_fn

    @staticmethod
    def init_datapipe_process(num_workers, worker_id, datapipe):
        # TODO(614): Add distributed support
        # TODO(615): Add shuffle determinism support
        torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)
        return datapipe

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        if self.num_workers == 0:
            # TODO(616): Warn and recommend usage of InProcessReadingService
            return datapipe
        for worker_id in range(self.num_workers):
            # TODO(617): Separate into function, because we also need to apply distributed seed
            #            and call it inside process
            call_inside_process = functools.partial(self.init_datapipe_process, self.num_workers, worker_id)
            ctx = mp.get_context(self.multiprocessing_context)
            (process, req_queue, res_queue) = communication.eventloop.SpawnProcessForDataPipeline(
                ctx, datapipe, call_inside_process
            )
            process.start()
            self.processes.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
            )
            self.datapipes.append(local_datapipe)

        datapipe = IterableWrapper(_IterateQueueDataPipes(self.datapipes), deepcopy=False)  # type: ignore[return-value]
        if self.post_adapter_fn is not None:
            datapipe = self.post_adapter_fn(datapipe)
        return datapipe

    def initialize_iteration(self) -> None:
        for dp in self.datapipes:
            dp.reset_iterator()

    def __del__(self):
        self.finalize()

    def finalize(self) -> None:
        # TODO(618): Check if anyone stuck with messages
        def clean_me(process, req_queue, res_queue):
            # TODO(619): Can send terminations simultaneously
            # TODO(620): Make termination a function of QueueWrapperDataPipe (similar to reset)
            req_queue.put(communication.messages.TerminateRequest())
            _ = res_queue.get()
            process.join()

        for process, req_queue, res_queue in self.processes:
            clean_me(process, req_queue, res_queue)

        self.processes = []


class Prototype2MultiProcessingReadingService(ReadingServiceInterface):
    num_workers: int
    #   pin_memory: bool
    #   timeout: float
    #   worker_init_fn: Optional[Callable[[int], None]]
    #   prefetch_factor: int
    #   persistent_workers: bool

    processes: List
    datapipes: List

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context=None,
        post_adapter_fn=None,
    ) -> None:
        self.num_workers = num_workers
        # TODO(VitalyFedyunin): Should be one of 'fork', 'spawn'
        self.multiprocessing_context = multiprocessing_context
        self.processes_t0 = []
        self.processes_t1 = []
        self.datapipes = []
        self.post_adapter_fn = post_adapter_fn

    @staticmethod
    def init_datapipe_process(num_workers, worker_id, datapipe):
        # TODO(VitalyFedyunin): Add distributed support
        # TODO(VitalyFedyunin): Add shuffle determinism support
        # torch.utils.data.graph_settings.apply_sharding(datapipe, num_workers, worker_id)

        # DO nothing as we shard in separate process now
        return datapipe

    @staticmethod
    def connect_sharding_datapipe_to_queue(req_queue, res_queue, call_after_fn, datapipe):
        shard_dp = Prototype2MultiProcessingReadingService.get_sharding_datapipe(datapipe)
        queue_consume_datapipe = communication.iter.QueueWrapper(
            communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
        )
        graph_traverse = graph.traverse_dps(datapipe)
        datapipe_graph = graph.replace_dp(graph_traverse, shard_dp, queue_consume_datapipe)
        datapipe = list(datapipe_graph.values())[0][0]
        datapipe = call_after_fn(datapipe)
        return datapipe

    @staticmethod
    def get_sharding_datapipe(datapipe):
        graph_traverse = graph.traverse_dps(datapipe)
        shard_dps = graph.find_dps(graph_traverse, ShardingFilter)
        assert len(shard_dps) == 1
        return shard_dps[0]

    def initialize(self, datapipe: DataPipe) -> DataPipe:
        # TODO(VitalyFedyunin): Must have one and only one sharding pipe.
        # The separate process should only have one pipe connected to other parts of the graph.
        # If one of the conditions is not met, roll back to parallel processes.
        if self.num_workers == 0:
            # TODO(VitalyFedyunin): Warn and recommend usage of InPorcessReadingService
            return datapipe

        ctx = mp.get_context(self.multiprocessing_context)

        pre_shard_graph = graph.clone_datapipe(datapipe)
        shard_dp = self.get_sharding_datapipe(pre_shard_graph)
        pre_shard_dp = shard_dp.source_datapipe
        forked_dps = pre_shard_dp.fork(self.num_workers)
        sharded_forked_dps = []
        # Manually add sharding filters (per forked pipe), and apply sharding
        for pipe_id, pipe in enumerate(forked_dps):
            sharded_dp = pipe.sharding_filter()
            sharded_dp.apply_sharding(self.num_workers, pipe_id)
            sharded_forked_dps.append(sharded_dp)
        call_inside_process = None  # functools.partial(self.init_datapipe_process, 1, 0)
        process, pipes_and_queues = communication.eventloop.SpawnProcessForMultipleDataPipelines(
            ctx, sharded_forked_dps, call_inside_process
        )
        process.start()
        # Take care about termination of the separate process
        for _, req_queue, res_queue in pipes_and_queues:
            self.processes_t0.append((process, req_queue, res_queue))

        for worker_id, pipe_and_queues in zip(range(self.num_workers), pipes_and_queues):
            # TODO(VitalyFedyunin): Separate into function, because we also need to apply distributed seed and call it inside process
            call_inside_process = functools.partial(self.init_datapipe_process, self.num_workers, worker_id)

            call_inside_process = functools.partial(
                self.connect_sharding_datapipe_to_queue, pipe_and_queues[1], pipe_and_queues[2], call_inside_process
            )

            (process, req_queue, res_queue) = communication.eventloop.SpawnProcessForDataPipeline(
                ctx, datapipe, call_inside_process
            )
            process.start()
            self.processes_t1.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
            )
            self.datapipes.append(local_datapipe)

        datapipe = IterableWrapper(_IterateQueueDataPipes(self.datapipes), deepcopy=False)
        if self.post_adapter_fn is not None:
            datapipe = self.post_adapter_fn(datapipe)
        return datapipe  # type: ignore[return-value]

    def initialize_iteration(self) -> None:
        # for dp in self.datapipes:
        #     dp.reset_iterator()
        pass

    def __del__(self):
        self.finalize()

    def finalize(self) -> None:
        def terminate(processes):
            join_processes = set()
            # TODO(VitalyFedyunin): Check if anyone stuck with messages
            for process, req_queue, _res_queue in processes:
                # print(os.getpid(), " putting Terminate into queue of", process.pid)
                req_queue.put(communication.messages.TerminateRequest())
                join_processes.add(process)
            # TODO(VitalyFedyunin): Collect termination responses (if needed)
            for process in join_processes:
                # print("joining ", process.pid)
                process.join()
            # TODO(VitalyFedyunin): Add join timeouts and force-kills

        terminate(self.processes_t1)
        self.processes_t1 = []
        terminate(self.processes_t0)
        self.processes_t0 = []


class MultiProcessingReadingService(ReadingServiceInterface):
    num_workers: int
    pin_memory: bool
    timeout: float
    worker_init_fn: Optional[Callable[[int], None]]
    prefetch_factor: int
    persistent_workers: bool

    def __init__(
        self,
        num_workers: int = 0,
        pin_memory: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.dl_: Optional[DataLoader] = None

    # Wrap the DataLoader with IterableWrapper to respect type annotation
    def initialize(self, datapipe: DataPipe) -> DataPipe:
        self.dl_ = DataLoader(
            datapipe,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            multiprocessing_context=self.multiprocessing_context,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            # TODO(621): `collate_fn` is necessary until we stop using DLv1 https://github.com/pytorch/data/issues/530
            collate_fn=_collate_no_op,
            batch_size=1,  # This reading service assume batching is done via DataPipe
        )
        return IterableWrapper(self.dl_)  # type: ignore[return-value]

    def finalize(self) -> None:
        if self.persistent_workers and self.dl_ is not None and self.dl_._iterator is not None:
            self.dl_._iterator._shutdown_workers()  # type: ignore[attr-defined]
            self.dl_._iterator = None


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
        on the graph of ``DataPipe`` and returnes the graph attached with a ``FullSyncIterDataPipe``
        at the end.
        """
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Torch Distributed is required to be initialized")
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._pg = dist.new_group(backend="gloo", timeout=timedelta(seconds=self._timeout))
        torch.utils.data.graph_settings.apply_sharding(
            datapipe,
            self._world_size,
            self._rank,
        )
        # Only append FullSyncIterDataPipe if it's not presented at the end of the pipeline
        if not isinstance(datapipe, FullSync):
            datapipe = datapipe.fullsync(self._timeout)
        self._datapipe = datapipe
        return datapipe

    def initialize_iteration(self) -> None:
        r"""
        Shares the same seed from rank 0 to other ranks across the distributed processes
        and apply the random seed to the graph of ``DataPipe``.
        """
        # TODO: Seed Generator should be moved to DataLoader2 after the API
        #       change of initialize_iteration is landed.
        seed = self._share_seed()
        _seed_generator = torch.Generator()
        _seed_generator.manual_seed(seed)
        assert self._datapipe is not None
        self._datapipe = torch.utils.data.graph_settings.apply_random_seed(
            self._datapipe,
            _seed_generator,
        )

    def _share_seed(self):
        shared_seed = torch.empty((), dtype=torch.int64).random_()
        dist.broadcast(shared_seed, src=0, group=self._pg)
        return shared_seed.item()

    def finalize(self) -> None:
        r"""
        Clean up the distributed process group.
        """
        self._pg = None
