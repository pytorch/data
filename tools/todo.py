# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Scrip can be used with
# find -name '*.py' | grep -v third_party | perl -ne'print "python tools/todo.py $_"' | head -n 5 | bash

import configparser
import os
import re
import shutil
import sys
import tempfile

from github import Github  # pip install PyGithub

file_name = sys.argv[1]

config = configparser.ConfigParser(allow_no_value=True)
with open(os.path.join(os.path.expanduser("~"), ".ghstackrc")) as stream:
    config.read_string(stream.read())

GITHUB_KEY = config["ghstack"]["github_oauth"]


def get_git_branch_hash():
    stream = os.popen("git rev-parse origin/main")
    return stream.read().rstrip()


def generate_issue_id(id_or_name, title, file_name, line_number):
    git_branch_hash = get_git_branch_hash()
    # print(file_name, line_number, title, id_or_name)
    match = re.match(r"\((\d+)\)", id_or_name)
    if match:
        return int(match.group(1))
    match = re.match(r"\((.*)\)", id_or_name)
    name = None
    if match:
        name = match.group(1)
    if name is not None:
        owner = f"cc @{name}"
    else:
        owner = ""
    g = Github(GITHUB_KEY)
    repo = g.get_repo("pytorch/data")
    # label_be = repo.get_label("better-engineering" )
    # labels = [label_be]
    line_reference = f"https://github.com/pytorch/data/blob/{git_branch_hash}/{file_name}#L{line_number}"
    line_reference = line_reference.replace("/./", "/")
    body = """
This issue is generated from the TODO line

{line_reference}

{owner}
    """.format(
        owner=owner,
        line_reference=line_reference,
    )
    title = f"[TODO] {title}"
    issue = repo.create_issue(title=title, body=body, labels=[])
    print(f"Created issue https://github.com/pytorch/data/issues/{issue.number}")
    return issue.number


def update_file(file_name):
    try:
        f = tempfile.NamedTemporaryFile(delete=False)
        shutil.copyfile(file_name, f.name)
        with open(f.name) as f_inp:
            with open(file_name, "w") as f_out:
                for line_number, line in enumerate(f_inp.readlines()):
                    if not re.search(r"ignore-todo", line, re.IGNORECASE):
                        match = re.search(r"(.*?)#\s*todo\s*(\([^)]+\)){0,1}:{0,1}(.*)", line, re.IGNORECASE)
                        if match:
                            # print(line)
                            prefix = match.group(1)
                            text = match.group(3)
                            issue_id = generate_issue_id(str(match.group(2)), text, file_name, line_number + 1)
                            line = f"{prefix}# TODO({issue_id}):{text}\n"  # ignore-todo
                    f_out.write(line)
    except Exception as e:
        shutil.copyfile(f.name, file_name)
        raise e
    finally:
        os.unlink(f.name)


update_file(file_name)
