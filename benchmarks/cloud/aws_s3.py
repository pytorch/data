import argparse
import hashlib
import os

import time

from functools import partial

import pandas as pd

# import matplotlib.pyplot as plt

import psutil

from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper


def map_read(x):
    """
    Read stream and close. Used for tar files.
    """
    data = x[1].read()
    x[1].close()
    return x[0], data


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


def check_and_output_speed(prefix: str, function, rs, n_prefetch, n_md5):
    """
    Benchmark the speed of the prefetching setup and prints the results.
    Args:
        prefix: String indicating what is being executed
        function: function that returns a DataPipe
        rs: ReadingService for testing
        n_prefetch: number of batches to prefetch
    """
    initial_memory_usage = psutil.virtual_memory().used
    max_memory_usage = initial_memory_usage

    dp = function()

    rs_type = "DataLoader2 w/ tar archives"
    dl = DataLoader2(dp, reading_service=rs)

    start = time.time()
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
    function_name = function.__name__

    io_speed = int(total_size / total / 1024 / 1024)  # size MiBs per sec
    total_size = int(total_size / 1024 / 1024)  # total size in MiBs
    total = int(total)
    print(
        f"{prefix} {function_name} and {rs_type} with n_prefetch {n_prefetch} | n_md5 {n_md5} results are: total time {total} sec, with {items_len} items at {speed} files per/sec. {total_size} MiB with io speed at {io_speed} MiBps"
    )
    change_in_memory_usage = (max_memory_usage - initial_memory_usage) / 1024 / 1024
    print(f"initial_memory_usage: {initial_memory_usage / 1024 / 1024:0.1f} MiBs")
    print(f"change_in_memory_usage: {change_in_memory_usage:0.1f} MiBs\n")
    return (
        prefix,
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
    _prefix,
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
    def get_datapipe(path, n_items, n_md5, use_source_prefetch, use_s3=False):
        if use_s3:
            dp = IterableWrapper([path] * n_items).shuffle().sharding_filter()
        else:
            tar_files = [f"{path}/images{i}.tar" for i in range(n_items)]
            dp = IterableWrapper(tar_files).shuffle().sharding_filter()
        if use_source_prefetch:
            dp = dp.open_files_by_fsspec(mode="rb", anon=True).prefetch(5).load_from_tar(mode="r|")
        else:
            dp = dp.open_files_by_fsspec(mode="rb", anon=True).load_from_tar(mode="r|")
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
        dp_fn.__name__ = "S3_Tar"

    else:
        print("Loading data from disk...")
        fs_str = "Local"
        path = "/home/ubuntu/source_data/large_images_tars"
        dp_fn = partial(get_datapipe, path, args.n_tar_files, args.n_md5, args.use_source_prefetch, args.use_s3)
        dp_fn.__name__ = "Tar"
    print(f"{path = }")

    for n_workers in [2, 4, 8, 12]:
        # New Prototype RS DataLoader2
        for i in range(1 + args.n_epochs):  # 1 warm-up + n runs
            new_rs = PrototypeMultiProcessingReadingService(
                num_workers=n_workers, prefetch_worker=args.n_prefetch, prefetch_mainloop=args.n_prefetch
            )
            params = check_and_output_speed(
                f"[prefetch is True, {n_workers} workers]", dp_fn, new_rs, n_prefetch=args.n_prefetch, n_md5=args.n_md5
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

# python benchmarks/cloud/aws_s3.py --n-tar-files 200 --n-epochs 3 --n-md5 22
