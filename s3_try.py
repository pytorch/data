from torchdata.datapipes.iter import IterableWrapper
from test_pickle import map_read, map_calculate_md5, PrefetcherIterDataPipe
from functools import partial
from torchdata.dataloader2 import (
    DataLoader2,
    PrototypeMultiProcessingReadingService,
)

# individual_images_dp = IterableWrapper(["s3://torchdatabenchmarkdatasets/256.jpeg"] * 1057)
# individual_images_dp = individual_images_dp.load_files_by_s3(region="us-east-1")

# tar_dp = IterableWrapper(["s3://torchdatabenchmarkdatasets/images0.tar"])
# tar_dp = tar_dp.load_files_by_s3(region="us-east-1").load_from_tar(mode="r:")
# for i, x in enumerate(tar_dp):
#     print(x)
#     if i == 10:
#         break
#
# print()
# dp = IterableWrapper(["/home/ubuntu/source_data/large_images_tars/images0.tar"])
# dp = dp.open_files(mode="b").load_from_tar(mode="r:")
# for i, x in enumerate(dp):
#     print(x)
#     if i == 10:
#         break



def s3_dp(n_items, n_md5):
    dp = IterableWrapper(["s3://torchdatabenchmarkdatasets/images0.tar"] * n_items)
    dp = dp.load_files_by_s3(region="us-east-1").load_from_tar(mode="r:")
    # The same as tar_dp_n after
    dp = dp.map(map_read)
    dp = dp.map(partial(map_calculate_md5, n_md5=n_md5))
    return dp


def post_adapter_fn(dp):
    return PrefetcherIterDataPipe(dp, 10)



if __name__ == "__main__":
    for i in range(2):
        print(f"{i = }")
        dp = s3_dp(1, 1)
        dp = PrefetcherIterDataPipe(dp, 50)
        new_rs = PrototypeMultiProcessingReadingService(num_workers=1, post_adapter_fn=post_adapter_fn)
        dl = DataLoader2(dp, reading_service=new_rs)
        ls = list(dl)
        print(len(ls))



