import pickle
import os
import file_utils as utils


basedir = "/Users/balazs/university/models/"

vals = []

for dir in utils.list_dirs(basedir):
    file = os.path.join(dir, "results.pkl")
    numbers = pickle.load(open(file, 'rb'))
    vals.append((os.path.basename(dir), numbers.val_accuracy_masked[-1], numbers.val_accuracy[-1], numbers.val_losses[-1]))

def getkey(item):
    return item[3]

vals = sorted(vals, key=getkey)

for i in range(4):
    s = ""
    for val in vals:
        s += str(val[i])
        s += "\t\t\t\t\t\t"
    print(s)


#print(bestres)
#numbers = pickle.load(open(file, 'rb'))

#print(numbers[])