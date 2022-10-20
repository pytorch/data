# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the


import argparse
import hashlib
import os
import time
from functools import partial
from typing import Callable

import pandas as pd
import psutil
from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper


def map_read(t):
    """
    Read stream and close. Used for tar files.

    Args:
        t: (path, data_stream) tuple
    """
    data = t[1].read()
    t[1].close()
    return t[0], data


def map_calculate_md5(t, n_md5):
    """
    Calculate MD5 hash of data for `n_md5` number of times. Increasing the number of md5 calculation will determine
    CPU usage (this is an approximate for the complexity of data transforms).

    Args:
        t: (path, data) tuple
        n_md5: number of times to compute hash of the data
    """
    path, data = t
    long_str = ""
    for _ in range(n_md5):
        long_str += str(hashlib.md5(data).hexdigest())
    result = hashlib.md5(long_str.encode()).hexdigest()
    size = len(data)
    return path, str(result), size


def check_and_output_speed(prefix: str, create_dp_fn: Callable, n_prefetch: int, n_md5: int, n_workers: int):
    """
    Benchmark the speed of the prefetching setup and prints the results.

    Args:
        prefix: String indicating what is being executed
        create_dp_fn: function that returns a DataPipe
        n_prefetch: number of batches to prefetch
        n_md5: number of times to compute hash of the data
    """
    initial_memory_usage = psutil.virtual_memory().used
    max_memory_usage = initial_memory_usage

    dp = create_dp_fn()

    rs_type = "DataLoader2 w/ tar archives"
    new_rs = PrototypeMultiProcessingReadingService(
        num_workers=n_workers, prefetch_worker=n_prefetch, prefetch_mainloop=n_prefetch
    )
    dl: DataLoader2 = DataLoader2(dp, reading_service=new_rs)

    start = time.time()
    items_len = 0  # Number of items processed
    total_size = 0  # Number of bytes processed
    time_to_first = None
    for _name, _md5, size in dl:
        if items_len > 10 and time_to_first is None:
            time_to_first = time.time() - start
        total_size += size
        items_len += 1
        if psutil.virtual_memory().used > max_memory_usage:
            max_memory_usage = psutil.virtual_memory().used

    total = time.time() - start
    speed = int(items_len / total)  # item per sec
    function_name = create_dp_fn.__name__

    io_speed = int(total_size / total / 1024 / 1024)  # size MiBs per sec
    total_size = int(total_size / 1024 / 1024)  # total size in MiBs
    total = int(total)
    print(
        f"{prefix} {function_name} and {rs_type} with n_prefetch {n_prefetch} | "
        f"n_md5 {n_md5} results are: total time {total} sec, with {items_len} items at {speed} files per/sec. "
        f"{total_size} MiB with io speed at {io_speed} MiBps"
    )
    change_in_memory_usage = (max_memory_usage - initial_memory_usage) / 1024 / 1024
    print(f"initial_memory_usage: {initial_memory_usage / 1024 / 1024:0.1f} MiBs")
    print(f"change_in_memory_usage: {change_in_memory_usage:0.1f} MiBs\n")
    return (
        function_name,
        rs_type,
        n_prefetch,
        total,
        items_len,
        speed,
        total_size,
        io_speed,
        int(change_in_memory_usage),
    )


def append_result(
    df,
    workers,
    n_tar_files,
    n_md5,
    fs,
    iteration,
    columns,
    fn_name,
    rs_type,
    prefetch,
    total,
    items_len,
    speed,
    total_size,
    io_speed,
    change_in_memory_usage,
):
    return pd.concat(
        [
            df,
            pd.DataFrame(
                data=[
                    [
                        workers,
                        fn_name,
                        rs_type,
                        prefetch,
                        n_md5,
                        total,
                        n_tar_files,
                        items_len,
                        total_size,
                        speed,
                        io_speed,
                        fs,
                        iteration,
                        change_in_memory_usage,
                    ]
                ],
                columns=columns,
            ),
        ]
    )


def save_result(df, csv_name: str, directory: str = ""):
    file_path = os.path.join(directory, f"{csv_name}.csv")
    df.to_csv(file_path, mode="a")  # Append result


def main(args):
    def get_datapipe(path, n_items, n_md5, use_source_prefetch, use_s3=False):
        if use_s3:
            dp = IterableWrapper([path] * n_items).shuffle().sharding_filter()
            dp = dp.open_files_by_fsspec(mode="rb", anon=True)
            if use_source_prefetch:
                dp = dp.prefetch(5)
            dp = dp.load_from_tar(mode="r|")
        else:
            tar_files = [f"{path}/images{i}.tar" for i in range(n_items)]
            dp = IterableWrapper(tar_files).shuffle().sharding_filter().open_files(mode="b")
            if use_source_prefetch:
                dp = dp.prefetch(5)
            dp = dp.load_from_tar(mode="r:")
        dp = dp.map(map_read)
        dp = dp.map(partial(map_calculate_md5, n_md5=n_md5))
        return dp

    columns = [
        "n_workers",
        "file_type",
        "RS Type",
        "n_prefetch",
        "n_md5",
        "total_time",
        "n_tar_files",
        "n_items",
        "total_size (MB)",
        "speed (file/s)",
        "io_speed (MB/s)",
        "fs",
        "iteration",
        "change_in_memory_usage",
    ]

    df = pd.DataFrame(columns=columns)

    if args.use_s3:
        print("Loading data from S3...")
        fs_str = "s3"
        path = "s3://torchdatabenchmarkdatasets/images0.tar"
        dp_fn = partial(get_datapipe, path, args.n_tar_files, args.n_md5, args.use_source_prefetch, args.use_s3)
        dp_fn.__name__ = "S3_Tar"  # type: ignore[attr-defined]

    else:
        print("Loading data from disk...")
        fs_str = "Local"
        path = "/home/ubuntu/source_data/large_images_tars"
        dp_fn = partial(get_datapipe, path, args.n_tar_files, args.n_md5, args.use_source_prefetch, args.use_s3)
        dp_fn.__name__ = "Tar"  # type: ignore[attr-defined]
    # print(f"{path = }")

    for n_workers in [4, 8, 12]:
        for i in range(1 + args.n_epochs):  # 1 warm-up + n runs
            params = check_and_output_speed(
                f"[prefetch is True, {n_workers} workers]",
                dp_fn,
                n_prefetch=args.n_prefetch,
                n_md5=args.n_md5,
                n_workers=n_workers,
            )
            df = append_result(df, n_workers, args.n_tar_files, args.n_md5, fs_str, i, columns, *params)

    # Save CSV
    print(df)
    save_result(df, csv_name=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-epochs", default=3, type=int, help="Number of times to benchmark per setup excluding warm up"
    )
    parser.add_argument("--n-tar-files", default=200, type=int, help="Number of tar files (~100MB each)")
    parser.add_argument("--n-prefetch", default=20, type=int, help="Number of batches to prefetch")
    parser.add_argument(
        "--n-md5",
        default=22,
        type=int,
        help="Number of times to compute MD5 hash per file, "
        "a proxy for transformation complexity "
        "(Low ~3ms: 22, Med ~7ms: 54, High ~10ms: 77)",
    )
    parser.add_argument("--output-file", default="benchmark_result", type=str, help="output csv file name")
    parser.add_argument("--use-s3", default=False, action="store_true", help="Load file from S3 instead of local")
    parser.add_argument("--use-source-prefetch", default=False, action="store_true", help="Use source prefetch")
    args = parser.parse_args()
    main(args)

# python ~/data/benchmarks/cloud/aws_s3.py --n-tar-files 500 --n-epoch 1 --n-md5 22 &&
# python ~/data/benchmarks/cloud/aws_s3.py --n-tar-files 500 --n-epoch 1 --n-md5 22 --use-s3 &&
# python ~/data/benchmarks/cloud/aws_s3.py --n-tar-files 500 --n-epoch 1 --n-md5 54 &&
# python ~/data/benchmarks/cloud/aws_s3.py --n-tar-files 500 --n-epoch 1 --n-md5 54 --use-s3 &&
# python ~/data/benchmarks/cloud/aws_s3.py --n-tar-files 500 --n-epoch 1 --n-md5 77 &&
# python ~/data/benchmarks/cloud/aws_s3.py --n-tar-files 500 --n-epoch 1 --n-md5 77 --use-s3
