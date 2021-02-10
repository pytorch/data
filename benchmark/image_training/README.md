# image_training

After clone the repo, pls go into the `torchdata/benchmark/image_training` folder. <br>
Add all the training and validation images into `images_dataset_ds/train` and `images_dataset_ds/val` folder.

Run `python3 generate_tar_datasets.py` to generate tar data files. (Please see comment in the script) <br>
To train using dataset, run `python3 ./train.py -r ./images_dataset_ds -ds`. <br> 
To train using datapipe, run `python3 ./train.py -r ./images_dataset_tar -dp -n 10`.

TODO:
1. The iterable datapipe has no way to tell how many labels within the datapipe, so need to specify explicitly at the moment.
2. We currently shuffle the input files in preprocessing stage (when generating tar file). We will add a shuffle layer later.
3. We still keep the fast-forwarding logic when training with datapipe to get the size of the whole datapipe. 
   We need this to calculate the training accuracy, can not get rid of this logic when batch size is not 1.
   Another way is to explicitly assign the size (length) to the datapipe. Need to research on this later.
