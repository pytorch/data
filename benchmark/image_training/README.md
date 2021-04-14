# image_training

After clone the repo, pls go into the `torchdata/benchmark/image_training` folder to start training for image classification. <br>
Add all the training and validation images into subfolders of `images_dataset_ds/train` and `images_dataset_ds/val`. Each subfolder is required to represent the corresponding class of images.

Run `python3 generate_tar_datasets.py` to generate tar data files. (Please see comment in the script) <br>
To train using dataset, run `python3 ./train.py -r ./images_dataset_ds -ds -bs=16 -f=./temp.pt -rmf`. <br> 
To train using datapipe, run `python3 ./train.py -r ./images_dataset_tar -dp -n=10 -bs=16 -f=./temp.pt -rmf`. <br>
To train using webdataset, run `python3 ./train.py -r ./images_dataset_tar -wds -n=10 -bs=16 -f=./temp.pt -rmf`.

- Add `-bs <batchsize>` and `-dl <droplast>` to control batching (default batch size is 1 and default drop last is False)
- Add `-s <shufflebuffersize>` to control the buffer size of shuffling for datapipe and webdataset
- Add `-nk <numberofworkers>` to control multiprocessing (default is 2)
- Add `-ep <numberofepochs>` to control the number of epochs for training (default is 5)
- Add `-rmf <logfilepath>` to enable resource monitoring for CPU+MEMORY usage (default is ./resource_usage.csv)

TODO:
1. The iterable datapipe has no way to tell how many labels within the datapipe, so need to specify explicitly at the moment.
