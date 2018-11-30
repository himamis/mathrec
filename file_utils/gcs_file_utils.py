import numpy as np
import pickle
import cv2
from io import BytesIO
from google.cloud import storage
from file_utils import common
import tarfile

storage_client = storage.Client()
bucket_name = 'image2latex4'
bucket = storage_client.bucket(bucket_name)
#bucket.blob('file').download

#_gcs = gcsio.GcsIO()

#client = storage.Client("")

def read_pkl(url):
    blob = bucket.blob(url)
    bytes = BytesIO(blob.download_as_string())
    content = pickle.load(bytes)
    return content


def write_pkl(url, data):
    data_string = pickle.dumps(data, 2)
    blob = bucket.blob(url)
    blob.upload_from_string(data_string)
    #with _gcs.open(url, 'wb') as output:
    #    pickle.dump(data, output, 2)


def read_img(url, max_size=None):
    blob = bucket.blob(url)
    bytes = blob.download_as_string()
    return common.image_from_bytes(bytes)

def list_files(url):
    prefix = "gs://" + bucket.name
    return [blob.name for blob in bucket.list_blobs(prefix=url[url.find(prefix)+len(prefix)+1:])]

def read_tar(url):
    blob = bucket.blob(url)
    bytes = BytesIO(blob.download_as_string())
    return tarfile.open(fileobj=bytes)


#def read_lines(url):
#    with _gcs.open(url, 'r') as input:
#        content = input.read().splitlines()
#    return content


#def write_list(url, list):
#    with _gcs.open(url, 'w') as output:
#        for token in list:
#            output.write("%s\r\n" % token)


#def read_content(url):
#    with _gcs.open(url, 'r') as input:
#        content = input.read()
#    return content


def write_string(url, string):
    with _gcs.open(url, 'w') as output:
        output.write(string)


def read_npy(url):
    return None
    #with _gcs.open(url, 'rb') as input:
    #    arr = np.load(input, encoding='bytes')
    #return arr


def write_npy(url, arr):
    output = BytesIO()
    np.save(output, arr)
    blob = bucket.blob(url)
    blob.upload_from_string(output.getvalue())
    #with _gcs.open(url, 'wb') as output:
    #    arr = np.save(output, arr)


def file_exists(url):
    return False
    #return _gcs.exists(url)