'''
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
from google.cloud import storage
#from apache_beam.io.gcp import gcsio

#_gcs = gcsio.GcsIO()

client = storage.Client("")

def read_pkl(url):

    storage.Bucket()
    with _gcs.open(url, 'rb') as input:
        content = pickle.load(input)
    return content


def write_pkl(url, data):
    with _gcs.open(url, 'wb') as output:
        pickle.dump(data, output, 2)


def read_img(url, max_size=None):
    with _gcs.open(url, 'rb') as inp:
        cont = inp.read()
    with Image.open(BytesIO(cont)) as img:
        image = img.convert('YCbCr')
        if(max_size != None):
            image = image.resize(max_size)
        nparr = np.asarray(image)
    return nparr


def read_lines(url):
    with _gcs.open(url, 'r') as input:
        content = input.read().splitlines()
    return content


def write_list(url, list):
    with _gcs.open(url, 'w') as output:
        for token in list:
            output.write("%s\r\n" % token)


def read_content(url):
    with _gcs.open(url, 'r') as input:
        content = input.read()
    return content


def write_string(url, string):
    with _gcs.open(url, 'w') as output:
        output.write(string)


def read_npy(url):
    with _gcs.open(url, 'rb') as input:
        arr = np.load(input, encoding='bytes')
    return arr


def write_npy(url, arr):
    with _gcs.open(url, 'wb') as output:
        arr = np.save(output, arr)


def file_exists(url):
    return _gcs.exists(url)
'''