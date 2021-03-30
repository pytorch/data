import io
import os
import time
import random
import math
import tarfile
import warnings
from typing import Union, List
from pathlib import Path

from torch.utils.data.datasets.common import get_file_pathnames_from_root


def is_img_ext(ext: str):
    return ext.lower() in [".png", ".jpg", ".jpeg", ".img", ".image", ".pbm", ".pgm", ".ppm"]


class ImageFolder:
    r""" :class:`ImageFolder`

    This is a class to do pre-processing for an image folder
    args:
        root: the root path of the image files. Can be either a single root string or a list of root strings
        recursive: if True, ImageFolder will collect all the subfolders for each input path in `root`

    """

    def __init__(self, root: Union[str, List[str]] = '.', recursive: bool = False):
        from os.path import abspath as absp
        from os.path import normpath as normp
        self.root_list: List[str] = [absp(normp(root))] if isinstance(root, str) else [absp(normp(x)) for x in root]

        self.classes = set()
        if recursive:
            n = len(self.root_list)
            for i in range(0, n):
                for rt, dirs, _ in os.walk(self.root_list[i]):
                    if len(dirs) > 0:
                        for subdir in dirs:
                            self.classes.add(subdir)
                            self.root_list.append(os.path.join(rt, subdir))
        # This is slow, but simple, ok for preprocessing
        self.root_list = list(set(self.root_list))
        self.class_to_idx = {c: i for i, c in enumerate(sorted(self.classes))}


    def to_tar(self,
               tar_pathname: Union[str, List[str]],
               *,
               create_label: bool = True,
               num_of_tar : int = 0,
               with_log : bool = True):

        assert tar_pathname

        # get all the files pathname from different input folders
        src_files_pathnames = []
        for root_path in self.root_list:
            for pathname in get_file_pathnames_from_root(root_path, ''):
                filename = os.path.basename(pathname)
                ext = os.path.splitext(filename)[1]

                # do not allow any non image file exist in the folder
                if not is_img_ext(ext):
                    warnings.warn("Image folder {} contains non image file {}, skip!".format(root_path, pathname))
                    continue

                src_files_pathnames.append((pathname, root_path))
        # shuffle pathnames
        random.shuffle(src_files_pathnames)

        # target file list
        if isinstance(tar_pathname, list):
            target_file_list = tar_pathname
        elif num_of_tar > 0:
            target_file_list = []
            for i in range(0, num_of_tar):
                target_file_list.append(tar_pathname + "_" + str(i) + ".tar.gz")
        else:
            target_file_list = [tar_pathname]

        num_source_files = len(src_files_pathnames)
        num_target_files = len(target_file_list)
        if num_source_files < num_target_files:
            # some generated files will be empty
            warnings.warn("The number of output files ({}) are bigger than the number of input files ({})".format(
                num_target_files, num_source_files))

        # generate output files
        source_file_count = 0
        avg_num_files_each = math.ceil(num_source_files / num_target_files)
        min_num_files_each = max(1, int(avg_num_files_each * 0.5))
        max_num_files_each = min(num_source_files, int(avg_num_files_each * 1.5))

        if with_log:
            print("Start creating a total num of {} tar file(s) with a total num of {} images".format(
                num_target_files, num_source_files))
            print("Avg num of image per file: {}, Min num of image per file: {}, Max num of image per file: {}".format(
                avg_num_files_each, min_num_files_each, max_num_files_each))
            print("NOTE that the last output tar file may have more image files than `Max num of image per file`")

        for i in range(0, num_target_files):
            tarstream = tarfile.open(target_file_list[i], mode="w:gz")

            prev_source_file_count = source_file_count
            if source_file_count < num_source_files:
                if i == num_target_files - 1:
                    # if this is the last output file, put the rest of files in it
                    num_files_each = num_source_files - source_file_count
                else:
                    # randomly pickup a number between [min_num_files_each, max_num_files_each]
                    num_files_each = random.randint(min_num_files_each, max_num_files_each)
                    # try to make sure each output file has at least `min_num_files_each` files in it.
                    num_src_files_needed = (num_target_files - i - 1) * min_num_files_each + num_files_each
                    num_src_files_left = num_source_files - source_file_count
                    # if we don't have enough src files left to maintain `min_num_files_each`, adjust `num_files_each`
                    if num_src_files_left < num_src_files_needed:
                        num_files_each = max(1, num_files_each - num_src_files_needed + num_src_files_left)

                while source_file_count < num_source_files and num_files_each > 0:
                    pathname = src_files_pathnames[source_file_count][0]
                    filename = os.path.basename(pathname)
                    basename = os.path.splitext(filename)[0]

                    # no encoding at the moment, store the raw image binary into tar file
                    tarstream.add(pathname, arcname=filename)

                    if create_label:
                        root_path = src_files_pathnames[source_file_count][1]
                        category = os.path.basename(os.path.normpath(root_path))
                        category_id = self.class_to_idx[category]
                        bio = io.BytesIO()
                        bio.write(str.encode('{{"category": "{c}", "category_id": {id}}}'
                                             .format(c=category, id=category_id)))
                        path_info = Path(pathname)
                        tinfo = tarfile.TarInfo(basename + ".json")
                        tinfo.size = bio.tell()
                        bio.seek(0)
                        tinfo.mtime = int(time.time())
                        tinfo.uname = path_info.owner()
                        tinfo.gname = path_info.group()
                        tinfo.mode = path_info.stat().st_mode & 0o0777
                        tarstream.addfile(tinfo, bio)

                    source_file_count = source_file_count + 1
                    num_files_each = num_files_each - 1

            tarstream.close()

            if with_log:
                print("Created {} with {} image(s)".format(target_file_list[i], source_file_count - prev_source_file_count))

        if with_log:
            print("Created a total of {} tar file(s) with a total of {} images".format(num_target_files, source_file_count))
