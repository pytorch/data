import os
import sys
from zipfile import ZipFile


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dist"
    filepath = os.path.join(path, "torchdata-0.3.0-py3-none-any.whl")
    zipf = ZipFile(filepath)
    for zip_info in zipf.infolist():
        if zip_info.filename.endswith("version.py"):
            print(zip_info.filename)
            f = zipf.open(zip_info)
            for line in f:
                print(line)
            break
