# Scrip can be used with
# find -name '*.py' | perl -ne'print "python ../todo.py $_"' | head -n 5 | bash

from github import Github # pip install PyGithub
import sys
import tempfile
import shutil
import os
import re
import configparser

file_name = sys.argv[1]

import yaml

import os

config = configparser.ConfigParser(allow_no_value=True)
with open(os.path.join(os.path.expanduser("~"),".ghstackrc"), "r") as stream:
    config.read_string(stream.read())

GITHUB_KEY = config['ghstack']['github_oauth']

def get_git_branch_hash():
    stream = os.popen("git rev-parse origin/main")
    return stream.read().rstrip()

def generate_issue_id(id_or_name, title, file_name, line_number):
    git_branch_hash = get_git_branch_hash()
    print(git_branch_hash)
    if re.match(r'\(\d+\)', id_or_name):
        return int(id_or_name)
    match = re.match('\((.*)\)', id_or_name)
    name = match.group(1)
    g = Github(GITHUB_KEY)
    repo = g.get_repo("pytorch/data")
    # label_be = repo.get_label("better-engineering" )
    # labels = [label_be]
    body = """
This issue is generated from the TODO line
https://github.com/pytorch/data/blob/{git_branch_hash}/{file_name}#L{line_number}
cc @{owner}
    """.format(owner = name, git_branch_hash= git_branch_hash, line_number=line_number,file_name=file_name)
    issue = repo.create_issue(title=title, body=body, labels = [])
    print(issue)
    return issue.number

def update_file(file_name):
    try:
        f = tempfile.NamedTemporaryFile(delete=False)
        shutil.copyfile(file_name, f.name)
        with open(f.name, "r") as f_inp:
            with open(file_name, "w") as f_out:
                for  line_number, line in enumerate(f_inp.readlines()):
                    if not re.search(r'ignore-todo', line, re.IGNORECASE):
                        match = re.search(r'(.*?)#\s*todo(\([^)]+\)){0,1}:{0,1}(.*)', line, re.IGNORECASE)
                        if match:
                            prefix = match.group(1)
                            text = match.group(3)
                            issue_id = generate_issue_id(str(match.group(2)),text, file_name, line_number+1)
                            line = "{}# TODO({}):{}\n".format(prefix, issue_id, text) # ignore-todo
                    f_out.write(line)
    except Exception as e:
        shutil.copyfile(f.name, file_name)
        print(e)
    finally:
        os.unlink(f.name)

update_file(file_name)


