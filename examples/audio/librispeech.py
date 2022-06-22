# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os

from pathlib import Path
from typing import Union

import torchaudio

from torchdata.datapipes.iter import FileOpener, HttpReader, IterableWrapper


URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"
BASE_URL = "http://www.openslr.org/resources/12/"
_CHECKSUMS = {
    "dev-clean.tar.gz": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "dev-other.tar.gz": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",
    "test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "test-other.tar.gz": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
    "train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    "train-clean-360.tar.gz": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
    "train-other-500.tar.gz": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",
}
AUDIO_EXT = ".flac"
TXT_EXT = ".trans.txt"


def decompress_filepath_fn(file_path, root_path):
    file_path = os.path.normpath(file_path)
    if file_path.endswith((AUDIO_EXT, TXT_EXT)):
        return os.path.join(root_path, *file_path.split(os.sep)[-4:])
    else:
        return os.path.join(root_path, os.path.basename(file_path))


def classify_file_fn(filepath):
    if filepath.endswith(AUDIO_EXT):
        return 0
    if filepath.endswith(TXT_EXT):
        return 1
    return None


def text_split_fn(line):
    fileid_text, transcript = line.strip().split(" ", 1)
    return (fileid_text, transcript)


def audio_key_fn(audio_file):
    audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
    return audio_filename


def load_librispeech_item(data):
    audio_file, transcript = data
    audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
    speaker_id, chapter_id, utterance_id = audio_filename.split("-")

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file)

    return (
        waveform,
        sample_rate,
        transcript,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


def LibriSpeech(root: Union[str, Path], url: str = URL, folder_in_archive: str = FOLDER_IN_ARCHIVE):
    if url in [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]:
        url = BASE_URL + url + ".tar.gz"

    # Get string representation of 'root' in case Path object is passed
    root = os.fspath(root)

    checksum_dict = {os.path.join(root, key): value for key, value in _CHECKSUMS.items()}

    url_dp = IterableWrapper([url])

    # Cache tar.gz archive
    cache_compressed_dp = url_dp.on_disk_cache(
        filepath_fn=lambda url: os.path.join(root, os.path.basename(url)),
        hash_dict=checksum_dict,
        hash_type="sha256",
    )
    cache_compressed_dp = HttpReader(cache_compressed_dp).end_caching(same_filepath_fn=True)

    # Cache decompressed archive into folder_in_archive
    cache_decompressed_dp = cache_compressed_dp.on_disk_cache(
        filepath_fn=lambda tar_path: os.path.join(root, folder_in_archive, tar_path.split(".")[0])
    )
    cache_decompressed_dp = FileOpener(cache_decompressed_dp, mode="b").load_from_tar()
    cache_decompressed_dp = cache_decompressed_dp.end_caching(
        filepath_fn=functools.partial(decompress_filepath_fn, root_path=os.path.join(root, folder_in_archive)),
    )

    audio_dp, txt_dp = cache_decompressed_dp.demux(2, classify_file_fn, drop_none=True, buffer_size=-1)

    txt_dp = FileOpener(txt_dp, mode="t").readlines(return_path=False).map(text_split_fn)
    transcript_map_dp = txt_dp.to_map_datapipe()

    audio_transcript_dp = audio_dp.zip_with_map(transcript_map_dp, key_fn=audio_key_fn)

    return audio_transcript_dp.map(load_librispeech_item)
