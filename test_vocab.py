from trainer.sequence import create_parser
from pickle import load, dump
from utilities import progress_bar
import cv2

vocabulary = load(open('/Users/balazs/vocabulary.pkl', 'rb'))
vocabulary |= {'^', '_', '{', '}', '\\frac', '\\mbox', '\\left', '\\right', '\\lbrack', '\\rbrack', \
               '\\cdot', '\\to', '\\Big', '\\;', '\\mathrm', '\'', '\ ', '\\parallel', '\\vtop', '\\hbox', \
               ''}


parser = create_parser(vocabulary)


images_train = load(open('/Users/balazs/images_test_clean.pkl', 'rb'))

cpis = []
es = []

for index, (image, string) in enumerate(images_train):
    progress_bar("Processing images", index, len(images_train))
    try:
        tokens = parser.parse(string)
    except Exception as e:
        cpis.append(index)
        es.append(e)

print("Found " + str(len(cpis)) + " number of instances")

with open('/Users/balazs/cleanup.pkl', 'wb') as f:
    dump(cpis, f)
    dump(es, f)


