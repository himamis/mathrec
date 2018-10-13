from utilities import parse_arg
import os
import file_utils

data_dir = parse_arg("--data-base-dir", required=True)
collection_name = parse_arg("--database", required=True)

base_dir = os.path.join(data_dir, collection_name)

file_path = os.path.join(base_dir, "data.txt")
file = open(file_path, "r")
lines = file.readlines()
file.close()

result = {}
for line in lines:
    split_line = line.replace("\n", "").split("\t")
    if len(split_line) != 2:
        print("Problem with line: " + line + " -- continuing\n")
        continue
    image_fname, formula = split_line
    image_path = os.path.join(base_dir, "images", image_fname)
    if not os.path.exists(image_path):
        print("Path does not exist: " + image_path + " -- continuing\n")
    image = file_utils.read_img(image_path)
    result[formula] = image


file = os.path.join(base_dir, "data.pkl")
file_utils.write_pkl(file, result)
