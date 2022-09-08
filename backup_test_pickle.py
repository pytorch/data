# Backing up unused functions here

# class UnpickleDP(IterDataPipe):
#     def __init__(self, source_datapipe):
#         self.source_datapipe = source_datapipe
#         # self.prefetch = prefetch
#         # self.thread = None
#
#     def __iter__(self):
#         for filename in self.source_datapipe:
#             with open(filename, "rb") as handle:
#                 b = pickle.load(handle)
#             yield b
#
#
# # Create pickle files
# def main_():
#     dp = IterableWrapper(["tar_images.tar"]).open_files(mode="b").load_from_tar()
#     all = list(dp)
#     # print(all)
#     all_data = []
#     for path, stream in all:
#         data = stream.read()
#         # print(len(data))
#         all_data.append((path, data))
#     # i = pickle.dumps(all_data)
#
#     with open("images.pickle", "wb") as handle:
#         pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print(len(i) / 1024 / 1024)

# ITEMS_NUM = 20  # 20, now defined in main
# PREFETCH_ITEMS = 200


# def tar_dp():
#     # tar_files = [f"tar_files/tar_images{i}.tar" for i in range(ITEMS_NUM)]
#     tar_files = [f"/home/ubuntu/source_data/large_images_tars/images{i}.tar" for i in range(ITEMS_NUM)]
#     dp = IterableWrapper(tar_files).shuffle().sharding_filter()
#     dp = dp.open_files(mode="b").load_from_tar(mode="r:")
#     dp = dp.map(map_read)
#     dp = dp.map(map_calculate_md5)
#     # dp = PrefetcherIterDataPipe(dp, PREFETCH_ITEMS)
#     return dp


# def pickle_dp():
#     pickle_files = [f"pickle_files/images{i}.pickle" for i in range(ITEMS_NUM)]
#     dp = IterableWrapper(pickle_files).shuffle().sharding_filter()
#     dp = UnpickleDP(dp).flatmap(noop)
#     dp = dp.map(map_calculate_md5)
#     # dp = PrefetcherIterDataPipe(dp, PREFETCH_ITEMS)
#     return dp

# def main():
    # rs = PrototypeMultiProcessingReadingService(num_workers=5, post_adapter_fn=post_adapter_fn)
    # check_and_output_speed(f"[prefetch is True, 5 workers]", tar_dp, rs, prefetch = 50)

    # rs = Prototype2MultiProcessingReadingService(num_workers=5, post_adapter_fn=post_adapter_fn)
    # check_and_output_speed(f"[prefetch is True, 5 workers]", tar_dp, rs, prefetch = 50)

    # rs = MultiProcessingReadingService(num_workers=5)
    # check_and_output_speed(f"[5 workers]", tar_dp, rs, prefetch = 50)

    # rs = PrototypeMultiProcessingReadingService(num_workers=1, post_adapter_fn=post_adapter_fn)
    # check_and_output_speed(f"[prefetch is True, 1 worker]", tar_dp, rs, prefetch = 50)

    # rs = MultiProcessingReadingService(num_workers=1)
    # check_and_output_speed(f"[1 worker]", tar_dp, rs, prefetch = 50)

    # check_and_output_speed(f"[1 inprocess]", tar_dp, None, prefetch = 50)

    # check_and_output_speed(f"[1 inprocess]", tar_dp, None, prefetch = None)


# def main_full():
#
#     for proto in [Prototype2MultiProcessingReadingService, PrototypeMultiProcessingReadingService]:
#         # for proto in [Prototype2MultiProcessingReadingService]:
#         for workers in [2, 5, 10, 15]:
#             for use_post_adapter_fn in [post_adapter_fn]:  # Not working with None(?)
#                 for function in [pickle_dp, tar_dp]:
#                     for use_prefetch in [None, 50, 200]:
#                         rs = proto(num_workers=workers, post_adapter_fn=use_post_adapter_fn)
#                         use_global_prefetch = use_post_adapter_fn is not None
#                         check_and_output_speed(
#                             f"[prefetch is {use_global_prefetch}, {workers} workers]",
#                             function,
#                             rs,
#                             prefetch=use_prefetch,
#                         )
#
#     for workers in [2, 5, 10, 15]:
#         for use_prefetch in [None, 50]:
#             for function in [pickle_dp, tar_dp]:
#                 rs = MultiProcessingReadingService(num_workers=workers)
#                 check_and_output_speed(f"[{workers} workers]", function, rs, prefetch=use_prefetch)

# def slow_map(x):
#     return x
#
#
# def main_test():
#     items = 1000
#     dp = IterableWrapper(RangeDebug(items)).map(slow_map).filter(is_odd).sharding_filter()
#     dp = PrefetcherIterDataPipe(dp, 20)
#     dp = dp.map(slow_map)
#     for i in iter(dp):
#         print(i)
#
#
# def main_mp_test():
#     items = 1000
#     datapipe = IterableWrapper(RangeDebug(items))
#
#     ctx = mp.get_context("fork")
#     num_workers = 2
#     forked_dps = datapipe.fork(num_workers)
#
#     sharded_forked_dps = []
#     # Manually add sharding filters (per forked pipe), and apply sharding
#     for pipe_id, pipe in enumerate(forked_dps):
#         sharded_dp = pipe.sharding_filter()
#         sharded_dp.apply_sharding(num_workers, pipe_id)
#         sharded_forked_dps.append(sharded_dp)
#     call_inside_process = None  # functools.partial(self.init_datapipe_process, 1, 0)
#     process, pipes_and_queues = communication.eventloop.SpawnProcessForMultipleDataPipelines(
#         ctx, sharded_forked_dps, call_inside_process
#     )
#     process.start()
#
#     processes = []
#     datapipes = []
#
#     # Take care about termination of the separate process
#     for _, req_queue, res_queue in pipes_and_queues:
#         dp = communication.iter.QueueWrapper(
#             communication.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue)
#         )
#         datapipes.append(dp)
#         processes.append((process, req_queue, res_queue))
#
#     # print(datapipes)
#
#     dp0, dp1 = datapipes
#     print(os.getpid(), "creating iterator 0")
#     it0 = iter(dp0)
#     print(os.getpid(), "creating iterator 1")
#     it1 = iter(dp1)
#     print(os.getpid(), "next(it0)")
#     item0 = next(it0)
#     print(item0)
#     item1 = next(it1)
#     print(item1)
#     item0 = next(it0)
#     item0 = next(it0)
#     item1 = next(it1)
#     print(os.getpid(), "resetting it1")
#     it1 = iter(dp1)
#     print(os.getpid(), "getting from it1")
#     item1 = next(it1)
#     print(os.getpid(), "getting from it0")
#     try:
#         item0 = next(it0)
#     except communication.iter.InvalidStateResetRequired:
#         print(os.getpid(), "invalid iterator confirmed")
#     print(os.getpid(), "creating new iterator")
#     it0 = iter(dp0)
#     print(os.getpid(), "getting new item")
#     item0 = next(it0)
#
#     for process, req_queue, res_queue in processes:
#         req_queue.put(communication.messages.TerminateRequest())
#
#     process.join()
