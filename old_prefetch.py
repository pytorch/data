import hashlib
import os

from typing import Optional
from torchdata.datapipes import functional_datapipe
import threading
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from functools import partial
import sys

from torchdata.dataloader2 import (
    communication,
    DataLoader2,
    MultiProcessingReadingService,
    Prototype2MultiProcessingReadingService,
    PrototypeMultiProcessingReadingService,
)
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
import pandas as pd
# import matplotlib.pyplot as plt

import psutil


def inc(x):
    return x + 1


def is_odd(x):
    return bool(x % 2)


PRODUCER_SLEEP_INTERVAL = 0.0001  # Interval between buffer fullfilment checks
CONSUMER_SLEEP_INTERVAL = 0.0001  # Interval between checking items availablitity in buffer


class _PrefetchData:
    def __init__(self, source_datapipe, buffer_size):
        self.run_prefetcher = True
        # TODO: Potential optimization is changing buffer from list to dequeue
        self.prefetch_buffer = []
        self.buffer_size = buffer_size
        self.source_datapipe = source_datapipe


@functional_datapipe("prefetch")
class PrefetcherIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, buffer_size: int = 10):
        self.source_datapipe = source_datapipe
        if buffer_size <= 0:
            raise ValueError("'buffer_size' is required to be a positive integer.")
        self.buffer_size = buffer_size
        self.thread: Optional[threading.Thread] = None

    @staticmethod
    def thread_worker(prefetch_data):
        itr = iter(prefetch_data.source_datapipe)
        stop_iteration = False
        while prefetch_data.run_prefetcher:
            if len(prefetch_data.prefetch_buffer) < prefetch_data.buffer_size and not stop_iteration:
                try:
                    item = next(itr)
                    prefetch_data.prefetch_buffer.append(item)
                except StopIteration:
                    stop_iteration = True
                except communication.iter.InvalidStateResetRequired:
                    stop_iteration = True
                except communication.iter.TerminateRequired:
                    prefetch_data.run_prefetcher = False
            elif stop_iteration and len(prefetch_data.prefetch_buffer) == 0:
                prefetch_data.run_prefetcher = False
            else:  # Buffer is full, waiting for main thread to consume items
                # TODO: Calculate sleep interval based on previous consumption speed
                time.sleep(PRODUCER_SLEEP_INTERVAL)

    def __iter__(self):
        if self.buffer_size < 1:
            yield from self.source_datapipe
        else:
            try:
                prefetch_data = _PrefetchData(self.source_datapipe, self.buffer_size)
                self.prefetch_data = prefetch_data
                self.thread = threading.Thread(
                    target=PrefetcherIterDataPipe.thread_worker, args=(prefetch_data,), daemon=True
                )
                self.thread.start()
                while prefetch_data.run_prefetcher:
                    if len(prefetch_data.prefetch_buffer) > 0:
                        yield prefetch_data.prefetch_buffer[0]
                        prefetch_data.prefetch_buffer = prefetch_data.prefetch_buffer[1:]
                    else:
                        # TODO: Calculate sleep interval based on previous availability speed
                        time.sleep(CONSUMER_SLEEP_INTERVAL)
            finally:
                prefetch_data.run_prefetcher = False
                if self.thread is not None:
                    self.thread.join()
                    self.thread = None

    def __getstate__(self):
        """
        Getting state in threading enviroment requires next operations:
            1) Stopping of the producer thread.
            2) Saving buffer.
            3) Adding lazy restart of producer thread when __next__ is called again
              (this will guarantee that you only change state of the source_datapipe
               after entire state of the graph is saved).
        """
        # TODO: Update __getstate__ and __setstate__ to support snapshotting and restoration
        return dict(source_datapipe=self.source_datapipe)

    def __setstate__(self, state):
        self.source_datapipe = state["source_datapipe"]

    def reset(self):
        if self.thread is not None:
            self.prefetch_data.run_prefetcher = False
            self.thread.join()

    def reset_iterator(self):
        self.reset()


class RangeDebug:
    """
    `__iter__` Creates an iterator of range(x)
    """

    def __init__(self, x):
        self.x = x

    def __iter__(self):
        for i in range(self.x):
            print(os.getpid(), f">>>>>>>> returning {i}")
            yield i


def post_adapter_fn(dp):
    return PrefetcherIterDataPipe(dp, 10)


def map_read(x):
    """
    Read stream and close. Used for tar files.
    """
    data = x[1].read()
    x[1].close()
    return x[0], data


def noop(x):
    return x


def image_loader(path):
    with open(path, "rb") as image:
        data = image.read()
    return data


def transform_md5(x, n_md5):
    long_str = ""
    for i in range(n_md5):  # 8 times use about 10 workers at 100%, need more workers to hit IO bandwidth
        long_str += str(hashlib.md5(x).hexdigest())
    result = hashlib.md5(long_str.encode()).hexdigest()
    size = len(x)
    return "", str(result), size


def get_sample_collate(x):
    return x[0][0]


def map_calculate_md5(x, n_md5):
    """
    Calculate MD5 hash of x[1]. Used by both DataPipes. This is like doing a transform.
    Increasing the number of md5 calculation will determine how much CPU you eat up
    (this is approximate for complexity of transforms).
    Balancing between IO and CPU bound.
    """
    long_str = ""
    for i in range(n_md5):
        long_str += str(hashlib.md5(x[1]).hexdigest())
    result = hashlib.md5(long_str.encode()).hexdigest()
    size = len(x[1])
    return x[0], str(result), size


def check_and_output_speed(prefix: str, function, rs, prefetch=None, dlv1=None, n_md5=None):
    """
    Benchmark the speed of the prefetching setup and prints the results.
    Args:
        prefix: String indicating what is being executed
        function: function that returns a DataPipe
        rs: ReadingService for testing
        prefetch: number of batches to prefetch

    Returns:
        TODO: Return data and charts if possible
    """
    initial_memory_usage = psutil.virtual_memory().used
    max_memory_usage = initial_memory_usage

    # /dev/nvme1n1p1:
    # Timing cached reads:       33676 MB in  2.00 seconds = 16867.31 MB/sec
    # Timing buffered disk reads: 2020 MB in  3.00 seconds = 673.32 MB/sec
    # [5 workers] tar_dp and MultiProcessingReadingService with prefetch None results are: total time 30 sec, with 200000 items 6533 per/sec, which is 133% of best. 21813 Mbytes with io speed at 712 MBps
    if dlv1 is not None:
        dl = dlv1
        rs_type = "Old DL w/ ImageFolder"
    else:
        dp = function()

        if prefetch is not None:
            dp = PrefetcherIterDataPipe(dp, prefetch)

        # Setup DataLoader, otherwise just use DataPipe
        if rs is not None:
            rs_type = rs.__class__.__name__
            if "Prototype" in rs_type:
                rs_type = "DataLoader2 w/ tar archives"
            else:
                rs_type = "Old DL w/ tar archives"
            dl = DataLoader2(dp, reading_service=rs)
        else:
            dl = dp
            rs_type = "[Pure DataPipe]"

    start = time.time()
    # report = start  # Time since last report, create one every 60s
    items_len = 0  # Number of items processed
    total_size = 0  # Number of bytes processed
    time_to_first = None
    # print('starting iterations')
    for _name, _md5, size in dl:
        if items_len > 10 and time_to_first is None:
            time_to_first = time.time() - start
        total_size += size
        items_len += 1
        if psutil.virtual_memory().used > max_memory_usage:
            max_memory_usage = psutil.virtual_memory().used

    total = time.time() - start
    speed = int(items_len / total)  # item per sec
    function_name = "ImageFolder" if dlv1 else function.__name__

    io_speed = int(total_size / total / 1024 / 1024)  # size MiBs per sec
    total_size = int(total_size / 1024 / 1024)  # total size in MiBs
    total = int(total)
    print(
        f"{prefix} {function_name} and {rs_type} with prefetch {prefetch} | n_md5 {n_md5} results are: total time {total} sec, with {items_len} items at {speed} files per/sec. {total_size} MiB with io speed at {io_speed} MiBps"
    )
    change_in_memory_usage = (max_memory_usage - initial_memory_usage) / 1024 / 1024
    print(f"initial_memory_usage: {initial_memory_usage / 1024 / 1024:0.1f} MiBs")
    print(f"change_in_memory_usage: {change_in_memory_usage:0.1f} MiBs\n")
    return prefix, function_name, rs_type, prefetch, total, items_len, speed, total_size, io_speed, int(change_in_memory_usage)


def append_result(df, workers, n_tar_files, n_md5, fs, iteration,
                  columns, _prefix, fn_name, rs_type, prefetch, total, items_len, speed, total_size, io_speed, change_in_memory_usage):
    return pd.concat(
        [
            df,
            pd.DataFrame(
                data=[[workers, fn_name, rs_type, prefetch, n_md5, total, n_tar_files, items_len, total_size, speed,
                       io_speed, fs, iteration, change_in_memory_usage]],
                columns=columns,
            ),
        ]
    )


def save_result(df, csv_name, path=""):

    # Save CSV, you can scp for the file afterwards
    df.to_csv(os.path.join(path, f"{csv_name}.csv"))

    # # Save Plot - we can plot it locally
    # df.set_index("n_workers", inplace=True)
    # df.groupby("RS Type")["io_speed (MB/s)"].plot(legend=True)
    #
    # plt.ylabel("IO Speed (MB/s)")
    # plt.xticks(range(0, max_worker))
    # plt.savefig(os.path.join(path, f"{img_name}.jpg"), dpi=300)


def main(args):
    args_fs_str = args.fs.lower()

    def tar_dp_n(path, n_items, n_md5):
        tar_files = [f"{path}/images{i}.tar" for i in range(n_items)]
        dp = IterableWrapper(tar_files).shuffle().sharding_filter()
        dp = dp.open_files(mode="b").load_from_tar(mode="r:")
        dp = dp.map(map_read)
        dp = dp.map(partial(map_calculate_md5, n_md5=n_md5))
        return dp

    def s3_dp(n_items, n_md5):

        if args_fs_str in ("s3_4x", "s3_10x"):
            s3_path = f"s3://torchdatabenchmarkdatasets/{args_fs_str[3:]}images0.tar"
            print(s3_path)
        else:
            s3_path = "s3://torchdatabenchmarkdatasets/images0.tar"
            print(s3_path)

        dp = IterableWrapper([s3_path] * n_items).shuffle().sharding_filter()
        # dp = dp.load_files_by_s3(region="us-east-1").load_from_tar(mode="r:")  # non-Streaming
        dp = dp.open_files_by_fsspec(mode="rb", anon=True).load_from_tar(mode="r|")  # Streaming version
        # The same as tar_dp_n after
        dp = dp.map(map_read)
        dp = dp.map(partial(map_calculate_md5, n_md5=n_md5))
        return dp

    n_tar_files = args.n_tar_files  # Each tar files is ~100MB
    n_prefetch = args.n_prefetch  # 100 by default
    n_md5 = args.n_md5  # 4 by default
    n_runs = args.n_epochs

    if args_fs_str == "local":
        path = "/home/ubuntu"
    elif args_fs_str in ("4x", "10x"):
        path = f"/{args_fs_str}tar"
    elif args_fs_str in ("io2", "gp2", "sc1", "st1", "ssd"):
        path = f"/{args_fs_str}_data"
    elif args_fs_str == "fsx_non_iso":
        path = "/fsx/users/ktse"
    elif args_fs_str in ("ontap", "fsx"):
        path = f"/{args_fs_str}_isolated/ktse"
    elif "s3" in args_fs_str:
        path = ""
    else:
        raise RuntimeError(f"Bad args.fs, was given {args.fs}")

    args_file_size_str = args.file_size.lower()
    if args_file_size_str in ("l", "large"):
        path += "/source_data/large_images_tars"
    elif args_file_size_str in ("xl", "xxl"):
        path += f"/source_data/{args_file_size_str}_images_tars"
    else:
        raise RuntimeError(f"Bad args.file_size, was given {args.file_size}")

    if args_fs_str == "local":
        image_folder_path = "/home/ubuntu/source_data/image_folder"
    elif args_fs_str == "fsx_non_iso":
        image_folder_path = "/fsx/users/ktse/source_data/image_folder"
    elif args_fs_str in ("ontap", "fsx"):
        image_folder_path = f"/{args_fs_str}_isolated/ktse/source_data/image_folder"
    else:
        image_folder_path = f"/{args_fs_str}_data/source_data/image_folder"


    columns = ["n_workers", "file_type", "RS Type", "n_prefetch", "n_md5", "total_time", "n_tar_files",
               "n_items", "total_size (MB)", "speed (file/s)", "io_speed (MB/s)", "fs", "iteration", "change_in_memory_usage"]

    df = pd.DataFrame(columns=columns)

    if args.use_s3:
        print("Loading data from S3...")
        dp_fn = partial(s3_dp, n_items=n_tar_files, n_md5=n_md5)
        dp_fn.__name__ = "S3_Tar"
        args_fs_str = "s3" if "s3" not in args_fs_str else args_fs_str
    else:
        print("Loading data from disk...")
        print(f"{path = }")
        print(f"{image_folder_path = }")
        dp_fn = partial(tar_dp_n, path=path, n_items=n_tar_files, n_md5=n_md5)
        dp_fn.__name__ = "Tar"

    for n_workers in [8, 12]:  # range(40, 88, 8):
        for n_prefetch in [n_prefetch]:  # The number of `n_prefetch` doesn't seem to influence the speed

            # Old DataLoader
            # for i in range(1 + n_runs):  # 1 warm-up + n runs
            #     old_rs = MultiProcessingReadingService(num_workers=n_workers, prefetch_factor=n_prefetch)
            #     params = check_and_output_speed(f"[{n_workers} workers]",
            #                                     dp_fn, old_rs, prefetch=n_prefetch, n_md5=n_md5)
            #     df = append_result(df, n_workers, n_tar_files, n_md5, args_fs_str, i, columns, *params)

            # New Prototype RS DataLoader2
            for i in range(1 + n_runs):  # 1 warm-up + n runs
                new_rs = PrototypeMultiProcessingReadingService(num_workers=n_workers, post_adapter_fn=post_adapter_fn)
                params = check_and_output_speed(f"[prefetch is True, {n_workers} workers]",
                                                dp_fn, new_rs, prefetch=n_prefetch, n_md5=n_md5)

                df = append_result(df, n_workers, n_tar_files, n_md5, args_fs_str, i, columns, *params)

            # DLv1 with ImageFolder
            # TODO: Improvement - I can add a function to filter out paths that are not relevant
            # n_folders = 1200
            # if n_tar_files != n_folders:
            #     warnings.warn(f"ImageFolder version always read all {n_folders} folder,"
            #                   f"but n_tar_files is {n_tar_files} != {n_folders}.")
            # image_folder = ImageFolder(root=image_folder_path,
            #                            transform=partial(transform_md5, n_md5=n_md5), loader=image_loader)
            # dlv1 = DataLoader(dataset=image_folder, num_workers=n_workers,
            #                   prefetch_factor=n_prefetch, collate_fn=get_sample_collate)
            #
            # for i in range(1 + n_runs):  # 1 warm-up + n runs
            #     params = check_and_output_speed(f"[DLv1 ImageFolder {n_workers} workers]",
            #                                     None, None, prefetch=n_prefetch, dlv1=dlv1, n_md5=n_md5)
            #     df = append_result(df, n_workers, n_tar_files, n_md5, args_fs_str, i, columns, *params)

    # Save CSV
    print(df)
    save_result(df, csv_name=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=str,
                        help="FileSystem (e.g. local, io2, gp2, sc1) storing tar files named 'images{i}.tar'",
                        default="local")
    parser.add_argument("--n-epochs", default=3, type=int,
                        help="Number of times to benchmark per setup excluding warm up")
    parser.add_argument("--n-tar-files", default=160, type=int, help="Number of tar files")
    parser.add_argument("--n-prefetch", default=50, type=int, help="Number of batches to prefetch")
    parser.add_argument("--n-md5", default=4, type=int,
                        help="Number of times to compute MD5 hash per file, "
                             "a proxy for transformation complexity (Low: 22, Med: 54, High: 77)")
    parser.add_argument("--file-size", default="large", type=str,
                        help="image size pixels, large (256x256), XL (512x512), XXL (1024x1024)")
    parser.add_argument("--output-file", default="prefetch_result", type=str,
                        help="output csv file name")
    parser.add_argument("--use-s3", default=False, action="store_true", help="Load file from S3 instead of local")
    args = parser.parse_args()
    main(args)
