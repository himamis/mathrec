import os
import numpy as np
import pickle
from file_utils import common
import tarfile

def read_pkl(file):
    with open(file, 'rb') as input:
        content = pickle.load(input)
    return content


def write_pkl(file, object):
    with open(file, 'wb') as output:
        pickle.dump(object, output, 2)  # 2 = protocol version that works with python 2.7


def read_tar(file):
    return tarfile.open(file, "r")

def read_lines(file):
    with open(file, 'r') as f:
        content = f.read().splitlines()
    return content;


def write_list(file, list):
    with open(file, 'w') as output:
        for token in list:
            output.write("%s\n" % token)


def write_npy(url, arr):
    with open(url, 'wb') as output:
        arr = np.save(output, arr)


def read_npy(url):
    with open(url, 'rb') as inp:
        arr = np.load(inp, encoding='bytes') # encoding that works with python 2.7
    return arr


def read_img(url):
    with open(url, 'rb') as inp:
        cont = inp.read()
        nparr = common.image_from_bytes(cont)
    return nparr


def file_exists(path):
    return os.path.exists(path)

def write_string(path, text):
    with open(path, 'w') as out:
        out.write(text)

def list_files(path):
    return [os.path.join(path, fname) for fname in next(os.walk(path))[2]]
