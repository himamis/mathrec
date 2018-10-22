import tarfile
from utilities import parse_arg
import os
import file_utils


data_dir = parse_arg("--data-base-dir", required=True)
collection_name = parse_arg("--database", required=True)

tar = tarfile.open(os.path.join(data_dir, collection_name, "images.tar"), "r")

for tarinfo in tar:
    #print(tarinfo.name, "is", tarinfo.size, "bytes in size and is", end="")
    f = tar.extractfile(tarinfo)
    if f is not None:
        content = f.read()
        img = file_utils.image_from_bytes(content)
        print("read")
    if tarinfo.isreg():

        print("a regular file.")
    elif tarinfo.isdir():
        print("a directory.")
    else:
        print("something else.")
tar.close()