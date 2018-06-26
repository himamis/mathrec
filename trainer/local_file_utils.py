import os
import numpy as np
import pickle
from PIL import Image
from io import BytesIO


def read_pkl(file):
    with open(file, 'rb') as input:
        content = pickle.load(input)
    return content


def write_pkl(file, object):
    with open(file, 'wb') as output:
        pickle.dump(object, output, 2)  # 2 = protocol version that works with python 2.7


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


def read_img(url, max_size=None):
    with open(url, 'rb') as inp:
        cont = inp.read()
    with Image.open(BytesIO(cont)) as img:
        image = img.convert('YCbCr')
        if(max_size != None):
            image = image.resize(max_size)
        nparr = np.asarray(image)
    return nparr


def file_exists(path):
    return os.path.exists(path)