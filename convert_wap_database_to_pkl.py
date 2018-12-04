from file_utils import list_files
from graphics.augment import Augmentor
import pickle

import os
import cv2

test_res = True

data_dir = '/Users/balazs/university/WAP/data'

database_name = 'off_image_16'
caption_name = '16_caption.txt'
out_file = '/Users/balazs/new_data/vocabulary.pkl'

captions = open(os.path.join(data_dir, caption_name))
captions = captions.readlines()

dict = open(os.path.join(data_dir, "dictionary.txt"))
dict = dict.readlines()
decoding = {}
for line in dict:
    l = line.strip().split()
    decoding[int(l[1])] = l[0]

del decoding[0]
decoding[0] = '<end>'
decoding[len(decoding)] = '<start>'

encoding = {value:key for value, key in zip(decoding.values(), decoding.keys())}


#caps = {}
#for caption in captions:
#    c = caption.strip().split()
#    caps[c[0]] = list(c[1:])



#aug = Augmentor()

#images = []

#for file in list_files(os.path.join(data_dir, database_name)):
#    image = cv2.imread(file)
##    image = aug.grayscale(image)
#    bname = os.path.splitext(os.path.basename(file))[0][:-2]

#    images.append((image, caps[bname], bname))

#if test_res:

##    for im in images[:10] + images[-10:]:
 #       print(im[1])
 #       print(im[2])
  #      cv2.imshow('im', im[0])
  #      cv2.waitKey(0)

  #  for im in images:
  ##      for word in im[1]:
   #         if not word in words:
   #             print("Problem with " + word)

f = open(out_file, 'wb')
pickle.dump((encoding, decoding), f)
f.close()
