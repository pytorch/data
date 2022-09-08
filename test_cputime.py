import hashlib
import timeit


def md5_fn(x, n_md5):
    """
    Calculate MD5 hash of x[1]. Used by both DataPipes. This is like doing a transform.
    Increasing the number of md5 calculation will determine how much CPU you eat up
    (this is approximate for complexity of transforms).
    Balancing between IO and CPU bound.
    """
    long_str = ""
    for _ in range(n_md5):  # 8 times use about 10 workers at 100%, need more workers to hit IO bandwidth
        long_str += str(hashlib.md5(x).hexdigest())
    result = hashlib.md5(long_str.encode()).hexdigest()
    size = len(x)
    return str(result), size


with open("/home/ubuntu/source_data/large_images/0.JPEG", "rb") as img_file:
    data = img_file.read()
    file_size = len(data)
    print(f"{file_size = } bytes")

if __name__ == '__main__':

    n_times = 100
    ms_per_n_md5_res = []
    for n_md5 in range(4, 80, 8):
        res = timeit.timeit(f'md5_fn(data, {n_md5})', number=n_times, setup="from __main__ import md5_fn, data")
        total_time_in_ms = res / n_times * 1000
        ms_per_n_md5 = total_time_in_ms / n_md5
        ms_per_n_md5_res.append(ms_per_n_md5)
        print(f"{n_md5 = }: {total_time_in_ms:0.2f} ms | {ms_per_n_md5:0.2f} ms per n_md5")

    # Result: 0.13 ms per n_md5 for file_size of 94525 bytes
    # Need: 22, 53, 75
    average = sum(ms_per_n_md5_res) / len(ms_per_n_md5_res)
    print(f"\nAssuming average {average:0.2f} ms per n_md5, then:")
    for target in [3, 7, 10]:
        print(f"n_md5 = {target / average:0.2f} for target time of {target:0.1f} ms")

    # 1057 images / tar * 10 tars per GB * 120 GB * 10 ms / image
    # 10ms means 10.57s per tar, about 6 tar per minute
    # 1057 * 10 * 120 * 10 ms to minute ~211.4 minutes on a single process
    # It should be faster on multiprocessing.

    # 40 tars * 10.57s / 16 workers
