from trainer.sequence import create_parser
from pickle import load, dump
from utilities import progress_bar
import cv2

vocabulary = load(open('/Users/balazs/vocabulary.pkl', 'rb'))
vocabulary |= {'^', '_', '{', '}', '\\frac', '\\mbox', '\\left', '\\right', '\\lbrack', '\\rbrack', \
               '\\cdot', '\\to', '\\Big', '\\;', '\\mathrm', '\'', '\ ', '\\parallel', '\\vtop', '\\hbox'}


with open('/Users/balazs/cleanup.pkl', 'rb') as f:
    cpis = load(f)
    es = load(f)

images_train = load(open('/Users/balazs/images_test.pkl', 'rb'))

replacements = []

for index, e in zip(cpis, es):
    image, string = images_train[index]
    found = False
    for old, new in replacements:
        if old == string:
            replacements.append((old, new))
            found = True
            break
    if found:
        continue

    fixed = string.strip()
    if fixed.endswith("\\!"):
        fixed = fixed[:-2]
    fixed = fixed.strip()
    if fixed.endswith("$"):
        fixed = fixed[:-1]
    fixed = fixed.strip()
    if fixed.startswith("\\ "):
        fixed = fixed[2:]
    fixed = fixed.strip()
    fixed = fixed.replace("$", "")
    fixed = fixed.replace("\\!", "")
    fixed = fixed.strip()
    print(str(e)[-10:])
    print("old: \t\t- " + string)
    print("fixed: \t\t- " + fixed)
    ok = 'f'
    while ok != 'y':
        new_input = input("Expression: ")
        if new_input == '':
            print("using old value")
            new_input = fixed
        ok = input("Sure? \'%s\' " % new_input)
        if ok == 'i':
            cv2.imshow("Window", image)
            cv2.waitKey(0)

    replacements.append((string, new_input))

for index, replacement in zip(cpis, replacements):
    images_train[index] = (images_train[index][0], replacement[1])

print("Writing")
with open('/Users/balazs/images_test_clean.pkl', 'wb') as f:
    dump(images_train, f)