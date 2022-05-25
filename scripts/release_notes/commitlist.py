# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is adpated from PyTorch Core
# https://github.com/pytorch/pytorch/blob/master/scripts/release_notes/commitlist.py

import argparse
import csv
import os

from common import get_features, run


class Commit:
    def __init__(self, commit_hash, category, topic, title):
        self.commit_hash = commit_hash
        self.category = category
        self.topic = topic
        self.title = title

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.commit_hash == other.commit_hash
            and self.category == other.category
            and self.topic == other.topic
            and self.title == other.title
        )

    def __repr__(self):
        return f"Commit({self.commit_hash}, {self.category}, {self.topic}, {self.title})"


class CommitList:
    # NB: Private ctor. Use `from_existing` or `create_new`.
    def __init__(self, path, commits):
        self.path = path
        self.commits = commits

    @staticmethod
    def from_existing(path):
        commits = CommitList.read_from_disk(path)
        return CommitList(path, commits)

    @staticmethod
    def create_new(path, base_version, new_version):
        if os.path.exists(path):
            raise ValueError("Attempted to create a new commitlist but one exists already!")
        commits = CommitList.get_commits_between(base_version, new_version)
        return CommitList(path, commits)

    @staticmethod
    def read_from_disk(path):
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
        assert all(len(row) >= 4 for row in rows)
        return [Commit(*row[:4]) for row in rows]

    def write_to_disk(self):
        path = self.path
        rows = self.commits
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            for commit in rows:
                writer.writerow([commit.commit_hash, commit.category, commit.topic, commit.title])

    def keywordInFile(file, keywords):
        for key in keywords:
            if key in file:
                return True
        return False

    @staticmethod
    def categorize(commit_hash, title):
        features = get_features(commit_hash, return_dict=True)
        title = features["title"]
        labels = features["labels"]
        category = "Uncategorized"
        topic = "Untopiced"

        # We ask contributors to label their PR's appropriately
        # when they're first landed.
        # Check if the labels are there first.
        already_categorized = already_topiced = False
        for label in labels:
            if label.startswith("release notes: "):
                category = label.split("release notes: ", 1)[1]
                already_categorized = True
            if label.startswith("topic: "):
                topic = label.split("topic: ", 1)[1]
                already_topiced = True
        if already_categorized and already_topiced:
            return Commit(commit_hash, category, topic, title)

        if "deprecation" in title.lower():
            topic = "deprecations"

        files_changed = features["files_changed"]
        for file in files_changed:
            if CommitList.keywordInFile(file, ["docker/", ".github", "packaging/"]):
                category = "releng"
                break
            if CommitList.keywordInFile(
                file,
                [
                    "torchdata/dataloader2",
                ],
            ):
                category = "dataloader2"
                break
            if CommitList.keywordInFile(
                file,
                [
                    "torchdata/datapipes",
                ],
            ):
                category = "datapipe"
                break

        return Commit(commit_hash, category, topic, title)

    @staticmethod
    def get_commits_between(base_version, new_version):
        cmd = f"git merge-base {base_version} {new_version}"
        rc, merge_base, _ = run(cmd)
        assert rc == 0

        # Returns a list of something like
        # b33e38ec47 Allow a higher-precision step type for Vec256::arange (#34555)
        cmd = f"git log --reverse --oneline {merge_base}..{new_version}"
        rc, commits, _ = run(cmd)
        assert rc == 0

        log_lines = commits.split("\n")
        hashes, titles = zip(*[log_line.split(" ", 1) for log_line in log_lines])
        return [CommitList.categorize(commit_hash, title) for commit_hash, title in zip(hashes, titles)]

    def filter(self, *, category=None, topic=None):
        commits = self.commits
        if category is not None:
            commits = [commit for commit in commits if commit.category == category]
        if topic is not None:
            commits = [commit for commit in commits if commit.topic == topic]
        return commits

    def update_to(self, new_version):
        last_hash = self.commits[-1].commit_hash
        new_commits = CommitList.get_commits_between(last_hash, new_version)
        self.commits += new_commits


def create_new(path, base_version, new_version):
    commits = CommitList.create_new(path, base_version, new_version)
    commits.write_to_disk()


def update_existing(path, new_version):
    commits = CommitList.from_existing(path)
    commits.update_to(new_version)
    commits.write_to_disk()


def main():
    """
    Example Usages

    Create a new commitlist.
    Said commitlist contains commits between v1.5.0 and f5bc91f851.

        python commitlist.py --create_new tags/v1.5.0 f5bc91f851

    Update the existing commitlist to commit bfcb687b9c.

        python commitlist.py --update_to bfcb687b9c

    """
    parser = argparse.ArgumentParser(description="Tool to create a commit list")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create_new", nargs=2)
    group.add_argument("--update_to")

    parser.add_argument("--path", default="results/commitlist.csv")
    args = parser.parse_args()

    if args.create_new:
        create_new(args.path, args.create_new[0], args.create_new[1])
        return
    if args.update_to:
        update_existing(args.path, args.update_to)
        return


if __name__ == "__main__":
    main()
