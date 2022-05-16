from github import Github # pip install PyGithub
import sys
import tempfile
import shutil
import os
import re

file_name = sys.argv[1]

GITHUB_KEY = "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"

def get_git_branch_hash():
    stream = os.popen("git rev-parse origin/main")
# output =
    return stream.read().rstrip()

# def find_owner(file_name, line_number):
#     command = "git blame {file_name}".format(file_name=file_name)
#     print(command)
#     stream = os.popen(command)
#     for line_n, line in enumerate(stream.readlines()):
#         print(line)
#         if line_n == line_number:
#             print("I blame". line)

def generate_issue_id(id_or_name, title, file_name, line_number):
    git_branch_hash = get_git_branch_hash()
    # print(git_branch_hash)
    match = re.match(r'\((\d+)\)', id_or_name)
    if match:
        return int(match.group(1))
    match = re.match('\((.*)\)', id_or_name)
    if match:
        cc = "cc @{}".format(match.group(1))
    else:
        cc = ""

    # find_owner(file_name, line_number)
    # name = match.group(1)
    g = Github(GITHUB_KEY)
    repo = g.get_repo("pytorch/data")

    label_todo = repo.get_label("todo")
    # label_porting = repo.get_label("topic: porting" )
    # label_operators = repo.get_label("module: operators" )
    # label_be = repo.get_label("better-engineering" )

    labels = [label_todo]

    body = """
This issue is generated from the TODO line
https://github.com/pytorch/data/blob/{git_branch_hash}/{file_name}#L{line_number}
{cc}
    """.format(cc = cc, git_branch_hash= git_branch_hash, line_number=line_number+1,file_name=file_name)
    # print(body)
    # print(title)
    title = "[TODO] {}".format(title)
    issue = repo.create_issue(title=title, body=body, labels = labels)
    print(issue)
    # die
    return issue.number

def update_file(file_name):
    try:
        f = tempfile.NamedTemporaryFile(delete=False)
        shutil.copyfile(file_name, f.name)
        with open(f.name, "r") as f_inp:
            with open(file_name, "w") as f_out:
                for  line_number, line in enumerate(f_inp.readlines()):
                    if not re.search(r'ignore-todo', line, re.IGNORECASE):
                        match = re.search(r'(.*?)#\s*todo\s*(\([^)]+\)){0,1}:{0,1}\s*(.*)', line, re.IGNORECASE)
                        # print(line)
                        if match:
                            prefix = match.group(1)
                            text = match.group(3)
                            issue_id = generate_issue_id(str(match.group(2)),text, file_name, line_number)
                            line = "{}# TODO({}): {}\n".format(prefix, issue_id, text) # ignore-todo
                    f_out.write(line)
    except Exception as e:
        shutil.copyfile(f.name, file_name)
        print(e)
    finally:
        os.unlink(f.name)
file_name = os.path.normpath(file_name)
# print('processing ', file_name)
update_file(file_name)


