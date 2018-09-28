import os
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import cv2


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


def read_img(url):
    with open(url, 'rb') as inp:
        cont = inp.read()
        buf = np.frombuffer(cont, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        nparr = np.asarray(img)
    return nparr


def file_exists(path):
    return os.path.exists(path)


def list_files(path):
    return [os.path.join(path, fname) for fname in next(os.walk(path))[2]]