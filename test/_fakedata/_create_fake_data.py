# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tarfile


NUMBER_OF_FILES = 3
FILES = [
    ("bytes", "bt", "{fn}_0123456789abcdef\n", True),
    ("csv", "csv", "key,item\n0,{fn}_0\n1,{fn}_1\n"),
    ("json", "json", '{{"{fn}_0": [{{"{fn}_01": 1}}, {{"{fn}_02": 2}}], "{fn}_1": 1}}\n'),
    ("txt", "txt", "{fn}_0123456789abcdef\n"),
]


def create_files(folder, suffix, data, encoding=False):
    os.makedirs(folder, exist_ok=True)
    for i in range(NUMBER_OF_FILES):
        fn = str(i)
        d = data.format(fn=fn)
        mode = "wb" if encoding else "wt"
        if encoding:
            d = d.encode()
        with open(folder + "/" + fn + "." + suffix, mode) as f:
            f.write(d)

    with tarfile.open(folder + ".tar", mode="w") as archive:
        archive.add(folder)

    with tarfile.open(folder + ".tar.gz", mode="w:gz") as archive:
        archive.add(folder)


def create_tfrecord_files(path: str):
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not found!")
        print("We will not generate tfrecord files.")
        return

    os.makedirs(path, exist_ok=True)
    with tf.io.TFRecordWriter(os.path.join(path, "example.tfrecord")) as writer:
        for i in range(4):
            x = tf.range(i * 10, (i + 1) * 10)
            record_bytes = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "x_float": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                        "x_int": tf.train.Feature(int64_list=tf.train.Int64List(value=tf.cast(x * 10, "int64"))),
                        "x_byte": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test str"])),
                    }
                )
            ).SerializeToString()
            writer.write(record_bytes)

    with tf.io.TFRecordWriter(os.path.join(path, "sequence_example.tfrecord")) as writer:
        for i in range(4):
            x = tf.range(i * 10, (i + 1) * 10)
            rep = 2 * i + 3

            record_bytes = tf.train.SequenceExample(
                context=tf.train.Features(
                    feature={
                        "x_float": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                        "x_int": tf.train.Feature(int64_list=tf.train.Int64List(value=tf.cast(x * 10, "int64"))),
                        "x_byte": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test str"])),
                    }
                ),
                feature_lists=tf.train.FeatureLists(
                    feature_list={
                        "x_float_seq": tf.train.FeatureList(
                            feature=[tf.train.Feature(float_list=tf.train.FloatList(value=x))] * rep
                        ),
                        "x_int_seq": tf.train.FeatureList(
                            feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=tf.cast(x * 10, "int64")))]
                            * rep
                        ),
                        "x_byte_seq": tf.train.FeatureList(
                            feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test str"]))] * rep
                        ),
                    }
                ),
            ).SerializeToString()
            writer.write(record_bytes)


if __name__ == "__main__":
    for args in FILES:
        create_files(*args)
    create_tfrecord_files("tfrecord")
