import file_utils
from utilities import parse_arg
import os
from random import shuffle
from shutil import copy

base_dir = parse_arg('--data-base-dir', required=True)
out_dir = parse_arg('--output-dir', required=True)

xainano_images_dir = os.path.join(base_dir)

if not os.path.exists(xainano_images_dir):
    print("Path does not exist: " + xainano_images_dir)
    exit()

if not os.path.exists(out_dir):
    print("Output path does not exist: " + out_dir)
    exit()

xainano_out_dir = os.path.join(out_dir, 'xainano_images')

if xainano_out_dir == xainano_images_dir:
    print("The input and output directories cannot be the same")
    exit()

if not os.path.exists(xainano_out_dir):
    os.makedirs(xainano_out_dir)

image_maps = {}
for file in file_utils.list_dirs(xainano_images_dir):
    symbol_dir = os.path.join(xainano_images_dir, file)
    symbol_files = file_utils.list_files(symbol_dir)
    symbol_files = [file for file in symbol_files if file.lower().endswith('jpg') or file.lower().endswith('jpeg')]
    symbol = os.path.basename(os.path.normpath(file))
    shuffle(symbol_files)
    image_maps[symbol] = symbol_files

names = ['training', 'validation', 'test']
percentages = [(0.0, 0.6), (0.6, 0.8), (0.8, 1.0)]

for name, percentage in zip(names, percentages):
    data_dir = os.path.join(xainano_out_dir, name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for symbol, files in image_maps.items():
        data_symbol_dir = os.path.join(data_dir, symbol)
        if not os.path.exists(data_symbol_dir):
            os.makedirs(data_symbol_dir)
        len_files = len(files)
        split_files = files[int(percentage[0] * len_files):int(percentage[1]*len_files)]
        [copy(split_file, data_symbol_dir) for split_file in split_files]


