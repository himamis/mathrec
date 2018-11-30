import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import file_utils as utils
import os
import pickle

basedir = "/Users/balazs/university/models/model-att2-conv32-rowbilstm"
file = os.path.join(basedir, "results.pkl")
numbers = pickle.load(open(file, 'rb'))


plt.plot(numbers.accuracy[0::441])
plt.plot(numbers.val_accuracy)

plt.show()