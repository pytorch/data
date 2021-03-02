from __future__ import print_function, division

import argparse
from collections import defaultdict
import copy
import os
import time

import torch
import torch.optim as optim
import torch.multiprocessing as multiprocessing
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import IterDataPipe
from torchvision import datasets, models, transforms

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
    imagehandler as decoder_imagehandler)

# These 2 lines is to enable importing `datapipes` and `dataloader` from current path
# Feel free to update/remove this as needed
import sys
sys.path.insert(0, '../..')
import dataloader
import datapipes
from benchmark.utils import AccMeter, AverageMeter, ProgressMeter


class TransferDatapipe(IterDataPipe):
    def __init__(self, datapipe, phase, length=-1):
        super().__init__()
        self.datapipe = datapipe
        self.transform = get_transform_api()[phase]

    def __iter__(self):
        for item in self.datapipe:
            yield (self.transform(item[0][1]), item[1][1])


class ClassesDatapipe(IterDataPipe):
    def __init__(self, datapipe):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self):
        for image, category in self.datapipe:
            yield image, category["category_id"]


cleanups = []


def cleanup_calls():
    global cleanups
    for fn, *args in cleanups:
        fn(*args)


def add_cleanup(fn, *args):
    global cleanups
    cleanups.append((fn, *args))


def get_progress_meters():
    pmeters = dict()
    for phase in ('train', 'val'):
        pm = ProgressMeter(prefix=phase.upper())
        pm.add_meter(AverageMeter('EpochTime', ':6.3f'))
        pm.add_meter(AverageMeter('IterTime', ':6.3f'))
        pm.add_meter(AverageMeter('DataTime', ':6.3f'))
        pm.add_meter(AverageMeter('Loss', ':.4e'))
        pm.add_meter(AccMeter('Acc', (1, 5), ':6.2f'))
        pmeters[phase] = pm
    return pmeters


def train_model(model, criterion, optimizer, scheduler,
                dataloaders, device, num_epochs=25, log_interval=500):
    pms = get_progress_meters()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    start = time.time()
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            pm = pms[phase]
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_start = time.time()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):

                iter_end = time.time()
                # Iterate over data.
                for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                    pm.update('DataTime', time.time() - iter_end)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    pm.update('Loss', loss.item(), inputs.size(0))
                    pm.update('Acc', outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    pm.update('IterTime', time.time() - iter_end)

                    if phase == 'train' and idx % log_interval == 0:
                        pm.display(epoch, idx, exclude=['EpochTime'])

                    iter_end = time.time()

            pm.update('EpochTime', time.time() - epoch_start)
            print("=" * 10)
            pm.display(epoch, exclude=['IterTime', 'DataTime'])
            print("=" * 10)

            if phase == 'train':
                scheduler.step()

            # deep copy the model
            top1_avg = pm.get_meter('Acc').get_meter('Acc@1').avg
            if phase == 'val' and top1_avg > best_acc:
                best_acc = top1_avg
                best_model_wts = copy.deepcopy(model.state_dict())

            epoch_end = time.time()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, pms


def get_transform_api():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def prepare_datapipe(data_dir, num_workers, shuffle_buffer):
    all_pipes = []
    all_workers = []

    ctx = multiprocessing.get_context('spawn')

    for i in range(num_workers):
        datapipe1_t = dp.iter.ListDirFiles(data_dir, 'train*.tar.gz')
        wrapped_dp1_t = datapipes.iter.IterDatasetWrapper(datapipe1_t)
        shard_dp1_t = datapipes.iter.SimpleSharding(wrapped_dp1_t)
        shard_dp1_t.sharding_settings(num_workers, i)
        datapipe2_t = dp.iter.LoadFilesFromDisk(shard_dp1_t)
        datapipe3_t = dp.iter.ReadFilesFromTar(datapipe2_t)
        datapipe4_t = dp.iter.RoutedDecoder(
            datapipe3_t, handlers=[decoder_imagehandler('pilrgb'), decoder_basichandlers])
        datapipe5_t = dp.iter.GroupByKey(datapipe4_t, group_size=2)
        transfered_dp_t = TransferDatapipe(datapipe5_t, 'train')
        wrapped_trans_dp_t = datapipes.iter.IterDatasetWrapper(transfered_dp_t)
        (process, req_queue, res_queue) = dataloader.eventloop.SpawnProcessForDataPipeline(
            ctx, wrapped_trans_dp_t)
        process.start()

        all_workers.append(process)
        local_datapipe = datapipes.iter.QueueWrapper(req_queue, res_queue)
        all_pipes.append(local_datapipe)

        def clean_me(req_queue, res_queue, process):
            req_queue.put(datapipes.nonblocking.StopIteratorRequest())
            value = res_queue.get()
            process.join()

        add_cleanup(clean_me, req_queue, res_queue, process)

    joined_dp_t = datapipes.iter.GreedyJoin(*all_pipes)
    shuffled_dp_t = dp.iter.Shuffle(joined_dp_t, buffer_size=shuffle_buffer)
    final_dp_t = ClassesDatapipe(shuffled_dp_t)


    datapipe1_v = dp.iter.ListDirFiles(data_dir, 'val*.tar.gz')
    datapipe2_v = dp.iter.LoadFilesFromDisk(datapipe1_v)
    datapipe3_v = dp.iter.ReadFilesFromTar(datapipe2_v)
    datapipe4_v = dp.iter.RoutedDecoder(
        datapipe3_v, handlers=[decoder_imagehandler('pilrgb'), decoder_basichandlers])
    datapipe5_v = dp.iter.GroupByKey(datapipe4_v, group_size=2)
    datapipe6_v = TransferDatapipe(datapipe5_v, 'val')
    final_dp_v = ClassesDatapipe(datapipe6_v)

    return {'train': final_dp_t, 'val': final_dp_v}


def prepare_dataset(data_dir):
    data_transforms = get_transform_api()
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return image_datasets


def _get_categoryid(json_obj):
    return json_obj["category_id"]


def prepare_webdataset(data_dir, shuffle_buffer, decoder="pil"):
    try:
        import webdataset as wds
    except ImportError:
        raise RuntimeError("Webdataset is required to be installed for benchmark.")

    image_datasets = {}
    tar_files = defaultdict(list)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            for phase in ("train", "val"):
                if file.endswith(".tar.gz") and file.startswith(phase):
                    tar_files[phase].append(os.path.join(root, file))

    for phase in ["train", "val"]:
        # Default buffer size 8192 for each file, not shuffle in each shard
        ds = wds.WebDataset(tar_files[phase], shardshuffle=False)
        # Yield dict{"__key__": fname, "png": image_data, "json": json_data}

        if phase == 'train' and shuffle_buffer > 0:
            ds = ds.shuffle(shuffle_buffer)

        ds = ds.decode(decoder) \
               .to_tuple("jpg;png", "json") \
               .map_tuple(get_transform_api()[phase], _get_categoryid)
        image_datasets[phase] = ds

    return image_datasets


def main(args):
    root = args.root
    num_workers = args.num_of_workers if args.num_of_workers is not None else 2
    if args.dataset:
        image_datasets = prepare_dataset(root)
        num_of_classes = len(image_datasets['train'].classes)
        dl_shuffle = args.shuffle_buffer > 0
    elif args.datapipe:
        image_datasets = prepare_datapipe(root, num_workers, args.shuffle_buffer)
        # We want to compare classic DataSet with N workers with DataPipes
        # which use N separate processes (self managed, so DataLoader is not
        # allowed to spawn anything)
        num_workers = 0
        num_of_classes = args.num_of_labels
        assert num_of_classes
        dl_shuffle = False
    else:
        image_datasets = prepare_webdataset(root, args.shuffle_buffer)
        num_of_classes = args.num_of_labels
        assert num_of_classes
        dl_shuffle = False

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=1,
                                                        shuffle=dl_shuffle, num_workers=num_workers),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1,
                                                      shuffle=False, num_workers=num_workers)
                   }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_of_classes)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, pms = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                dataloaders, device, args.num_epochs, args.log_interval)
    if args.file_path is not None:
        torch.save({
            'state_dict': model_ft.state_dict(),
            'train_statistic': pms['train'].get_state(),
            'val_statistic': pms['val'].get_state(),
        }, args.file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Datapipe Benchmark Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-ds", "--dataset", action="store_true",
                       help="use dataset for training if set")
    group.add_argument("-dp", "--datapipe", action="store_true",
                       help="use datapipe for training if set")
    group.add_argument("-wds", "--webdataset", action="store_true",
                       help="use webdataset for training if set")
    parser.add_argument("-r", "--root", required=True, help="root dir of images")
    parser.add_argument("-n", "--num_of_labels", type=int,
                        help="required for datapipe or webdataset")
    parser.add_argument("-nk", "--num_of_workers", type=int, help="number of workers")
    parser.add_argument("-s", "--shuffle-buffer", type=int, default=100,
                        help="size of buffer for shuffle. shuffling is enabled as "
                        "default, and it can be disabled by setting buffer to 0")
    parser.add_argument("-ep", "--num-epochs", type=int, default=5,
                        help="number of epochs")
    parser.add_argument("--log-interval", type=int, default=500,
                        help="number of batches to wait before logging training status")
    parser.add_argument("-f", "--file-path", type=str,
                        help="file path to save the best model and statistics")

    main(parser.parse_args())

    cleanup_calls()
