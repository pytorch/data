# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import torch
import torchdata.dataloader2.graph as graph
from torch.utils.data import DataLoader
from torch.utils.data.graph import DataPipe

from torchdata.dataloader2 import communication
from torchdata.datapipes.iter import IterableWrapper, ShardingFilter


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


class _IterateQueueDataPipes:
    def __init__(self, datapipes):
        self.datapipes = datapipes

    def __iter__(self):
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
                            time.sleep(0.001)


class PrototypeMultiProcessingReadingService(ReadingServiceInterface):
    num_workers: int
    processes: List
    datapipes: List

    def __init__(
        self,
        num_workers: int = 0,
        multiprocessing_context=None,
    ) -> None:
        self.num_workers = num_workers
        # TODO(613): Should be one of 'fork', 'spawn'
        self.multiprocessing_context = multiprocessing_context
        self.processes = []
        self.datapipes = []

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

        return IterableWrapper(_IterateQueueDataPipes(self.datapipes), deepcopy=False)  # type: ignore[return-value]

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
    ) -> None:
        self.num_workers = num_workers
        # TODO(VitalyFedyunin): Should be one of 'fork', 'spawn'
        self.multiprocessing_context = multiprocessing_context
        self.processes = []
        self.datapipes = []

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
        graph_traverse = graph.traverse(datapipe)
        datapipe_graph = graph.replace_dp(graph_traverse, shard_dp, queue_consume_datapipe)
        datapipe = list(datapipe_graph.values())[0][0]
        datapipe = call_after_fn(datapipe)
        return datapipe

    @staticmethod
    def get_sharding_datapipe(datapipe):
        graph_traverse = graph.traverse(datapipe)
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
            self.processes.append((process, req_queue, res_queue))

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
            self.processes.append((process, req_queue, res_queue))  # These queues are independent
            local_datapipe = communication.iter.QueueWrapper(
                communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
            )
            self.datapipes.append(local_datapipe)

        return IterableWrapper(_IterateQueueDataPipes(self.datapipes), deepcopy=False)  # type: ignore[return-value]

    def initialize_iteration(self) -> None:
        for dp in self.datapipes:
            dp.reset_iterator()

    def __del__(self):
        self.finalize()

    def finalize(self) -> None:
        join_processes = set()
        # TODO(VitalyFedyunin): Check if anyone stuck with messages
        for process, req_queue, res_queue in self.processes:
            req_queue.put(communication.messages.TerminateRequest())
            join_processes.add(process)
        # TODO(VitalyFedyunin): Collect termination responses (if needed)
        for process in join_processes:
            process.join()
        self.processes = []


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
