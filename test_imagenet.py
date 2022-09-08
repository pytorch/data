import os
from PIL import Image
from bench_transforms import ClassificationPresetTrain
from common import decode
from io import BytesIO
import time
import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from functools import partial

from torchdata.dataloader2 import (
    communication,
    DataLoader2,
    PrototypeMultiProcessingReadingService,
)
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
import pandas as pd

import psutil
from test_pickle import PrefetcherIterDataPipe

from ffcv.loader import Loader, OrderOption
from try_ffcv import pipelines as ffcv_pipelines


def post_adapter_fn(dp):
    return PrefetcherIterDataPipe(dp, 10)


def stream_to_image(x):
    """
    Read stream and close. Used for tar files.
    """
    buf = BytesIO(x[1].read())
    return Image.open(buf).convert("RGB")


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
    total_size = 33102845000  # 0  # Number of bytes processed
    time_to_first = None
    # print('starting iterations')
    for _tensor in dl:
        if items_len > 10 and time_to_first is None:
            time_to_first = time.time() - start
        # total_size += size
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

def check_and_output_speed_ffcv(prefix: str, ffcv_loader, batch_size):

    prefetch = 50

    rs_type = "FFCV"
    initial_memory_usage = psutil.virtual_memory().used
    max_memory_usage = initial_memory_usage

    start = time.time()
    # report = start  # Time since last report, create one every 60s
    items_len = 0  # Number of items processed
    total_size = 33102845000  # Number of bytes processed
    time_to_first = None
    # print('starting iterations')
    for _tensor in ffcv_loader:
        if items_len > 10 and time_to_first is None:
            time_to_first = time.time() - start
        # size = 94525  # Unless I compute size of tensor here
        # total_size += size
        items_len += batch_size
        if psutil.virtual_memory().used > max_memory_usage:
            max_memory_usage = psutil.virtual_memory().used

    total = time.time() - start
    speed = int(items_len / total)  # item per sec
    function_name = "FFCV"

    io_speed = int(total_size / total / 1024 / 1024)  # size MiBs per sec
    total_size = int(total_size / 1024 / 1024)  # total size in MiBs
    total = int(total)
    n_md5 = 0
    print(
        f"{prefix} {function_name} and {rs_type} with prefetch {prefetch} | n_md5 {n_md5} results are: total time {total} sec, with {items_len} items at {speed} files per/sec. {total_size} MiB with io speed at {io_speed} MiBps",
        flush=True
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


def main(args):
    args_fs_str = args.fs.lower()

    def tar_dp_n(path, n_items, use_source_prefetch):
        print(f"{use_source_prefetch = }")

        tar_files = IterableWrapper([path]).list_files()
        dp = tar_files.shuffle().sharding_filter()

        if use_source_prefetch:
            dp = dp.open_files(mode="b").prefetch(5).load_from_tar(mode="r:")
        else:
            dp = dp.open_files(mode="b").load_from_tar(mode="r:")
        dp = dp.map(stream_to_image)  #.map(ClassificationPresetTrain(on="pil"))

        return dp

    def s3_dp(n_items, n_md5, use_source_prefetch):
        print(f"{use_source_prefetch = }")
        if args_fs_str in ("s3_4x", "s3_10x"):
            s3_path = f"s3://torchdatabenchmarkdatasets/{args_fs_str[3:]}images0.tar"
            print(s3_path)
        else:
            s3_path = "s3://torchdatabenchmarkdatasets/images0.tar"
            print(s3_path)

        dp = IterableWrapper([s3_path] * n_items).shuffle().sharding_filter()
        # dp = dp.load_files_by_s3(region="us-east-1").load_from_tar(mode="r:")  # non-Streaming
        # Streaming version
        if use_source_prefetch:
            dp = dp.open_files_by_fsspec(mode="rb", anon=True).prefetch(5).load_from_tar(mode="r|")
        else:
            dp = dp.open_files_by_fsspec(mode="rb", anon=True).load_from_tar(mode="r|")
        # The same as tar_dp_n after
        dp = dp.map(map_read)
        dp = dp.map(partial(map_calculate_md5, n_md5=n_md5))
        return dp

    n_tar_files = args.n_tar_files  # Each tar files is ~100MB
    n_prefetch = 50
    n_runs = args.n_epochs


    if args_fs_str in ("ontap", "fsx"):
        path = f"/{args_fs_str}_isolated/ktse"
    else:
        raise RuntimeError(f"Bad args.fs, was given {args.fs}")


    columns = ["n_workers", "file_type", "RS Type", "n_prefetch", "n_md5", "total_time", "n_tar_files",
               "n_items", "total_size (MB)", "speed (file/s)", "io_speed (MB/s)", "fs", "iteration", "change_in_memory_usage"]

    df = pd.DataFrame(columns=columns)

    print("Loading data from disk...")
    ffcv_path = f"{path}/imagenet_beton/200.beton"
    tar_path = f"{path}/imagenet_tars/"
    print(f"{tar_path = }")
    print(f"{ffcv_path = }")

    dp_fn = partial(tar_dp_n, path=tar_path, n_items=n_tar_files,
                    use_source_prefetch=args.use_source_prefetch)
    dp_fn.__name__ = "Tar"

    for n_workers in [8, 12]:

        print(f"{n_workers = }")

        # New Prototype RS DataLoader2

        if not args.use_ffcv:
            print("Using DL2")


            for i in range(1 + n_runs):  # 1 warm-up + n runs

                print(f"Run {i}:")
                new_rs = PrototypeMultiProcessingReadingService(num_workers=n_workers, post_adapter_fn=post_adapter_fn)
                params = check_and_output_speed(f"[prefetch is True, {n_workers} workers]",
                                                dp_fn, new_rs, prefetch=n_prefetch)

                df = append_result(df, n_workers, n_tar_files, 0, args_fs_str, i, columns, *params)

        # FFCV
        else:
            print("Using FFCV", flush=True)

            for batch_size in [8, 64]:
                for i in range(1 + n_runs):  # 1 warm-up + n runs
                    print(f"Run {i}:", flush=True)
                    ffcv_dl = Loader(ffcv_path, batch_size=batch_size, num_workers=n_workers,
                                    order=OrderOption.QUASI_RANDOM, pipelines=ffcv_pipelines, os_cache=False, drop_last=False,
                                    recompile=True, batches_ahead=5)
                    params = check_and_output_speed_ffcv(f"[FFCV {n_workers} workers]", ffcv_loader=ffcv_dl, batch_size=batch_size)
                    df = append_result(df, n_workers, n_tar_files, 0, args_fs_str, i, columns, *params)
                    del ffcv_dl

    # Save CSV
    print(df)
    save_result(df, csv_name=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=str,
                        help="FileSystem (fsx or ontap)'",
                        default="fsx")
    parser.add_argument("--n-epochs", default=1, type=int,
                        help="Number of times to benchmark per setup excluding warm up")
    parser.add_argument("--n-tar-files", default=160, type=int, help="Number of tar files")
    parser.add_argument("--output-file", default="prefetch_result", type=str,
                        help="output csv file name")
    parser.add_argument("--use-source-prefetch", default=False, action="store_true", help="Use source prefetch")
    parser.add_argument("--use-ffcv", default=False, action="store_true", help="Use ffcv")
    args = parser.parse_args()
    main(args)
