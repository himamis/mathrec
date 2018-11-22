from utilities import parse_arg
from file_utils import *
import os
from random import shuffle
import pickle

data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/')
pkl_file = os.path.join(data_base_dir, "images.pkl")

if not file_exists(pkl_file):
    print("file does not exist" + pkl_file)
    exit(1)

train_f_p = os.path.join(data_base_dir, "images_train.pkl")
train_f = open(train_f_p, "wb")

test_f_p = os.path.join(data_base_dir, "images_test.pkl")
test_f = open(test_f_p, "wb")


images = pickle.load(open(pkl_file, 'rb'))

shuffle(images)
pct = 0.8

l = len(images)
div = int(l * pct)

train = images[:div]
test = images[div:]

pickle.dump(train, train_f)
pickle.dump(test, test_f)