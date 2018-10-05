from utilities import parse_arg
import os

base_dir = parse_arg("--data-base-dir", required=True)

data = open(os.path.join(base_dir, "data.txt"), "r+")
lines = data.readlines()
data.seek(0)

for line in lines:
    splitted = line.replace("\n", "").split("\t")
    if len(splitted) != 2:
        raise Exception("Line has other than 1 tab character: " + line)
    fname, formula = splitted
    if os.path.exists(os.path.join(base_dir, "images", fname)):
        data.write(line)

data.close()
