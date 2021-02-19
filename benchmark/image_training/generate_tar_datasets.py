import os
from imagefolder import ImageFolder

if not os.path.exists('images_dataset_tar'):
    os.makedirs('images_dataset_tar')

# The following commented way works as well.
# Feel free to use this style if you don't want to tar the images of entire folder
# or you want to give a list of output tar file names
"""
im = ImageFolder([
    './images_dataset_ds/train/011k07',
    './images_dataset_ds/train/015x4r',
    './images_dataset_ds/train/01bqk0',
    './images_dataset_ds/train/01jfm_',
    './images_dataset_ds/train/01s105'])

im.to_tar('./images_dataset_tar/train_images_1.tar.gz')

im = ImageFolder([
    './images_dataset_ds/val/011k07',
    './images_dataset_ds/val/015x4r',
    './images_dataset_ds/val/01bqk0',
    './images_dataset_ds/val/01jfm_',
    './images_dataset_ds/val/01s105'])

im.to_tar('./images_dataset_tar/val_images_1.tar.gz')

im = ImageFolder([
    './images_dataset_ds/train/021sj1',
    './images_dataset_ds/train/02d9qx',
    './images_dataset_ds/train/02s195',
    './images_dataset_ds/train/034c16',
    './images_dataset_ds/train/03l9g'])

im.to_tar('./images_dataset_tar/train_images_2.tar.gz')

im = ImageFolder([
    './images_dataset_ds/val/021sj1',
    './images_dataset_ds/val/02d9qx',
    './images_dataset_ds/val/02s195',
    './images_dataset_ds/val/034c16',
    './images_dataset_ds/val/03l9g'])

im.to_tar(['./images_dataset_tar/val_images_2.tar.gz'])
"""

# collect all folders under image_dataset_ds/train
im = ImageFolder('./images_dataset_ds/train', recursive=True)
# generate 150 tar files that contain all the images in above folders, store in images_dataset_tar
# images are shuffled, and the size of each tar file is random from [0.5 * avg per file size, 1.5 * avg per file size]
# tar file name will be train_images_x.tar.gz (x from 0 to 149)
im.to_tar('./images_dataset_tar/train_images', num_of_tar=150)


# collect all folders under image_dataset_ds/val
im = ImageFolder('./images_dataset_ds/val', recursive=True)
# generate 50 tar files that contain all the images in above folders, store in images_dataset_tar
# images are shuffled, and the size of each tar file is random from [0.5 * avg per file size, 1.5 * avg per file size]
# tar file name will be val_images_x.tar.gz (x from 0 to 49)
im.to_tar('./images_dataset_tar/val_images', num_of_tar=50)
