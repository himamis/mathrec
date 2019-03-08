import os

path_prefix = "/Users/balazs/Documents/datasets/CROHME2019/Data/Task1_onlineRec/Train/INKMLs/"
paths = [
    "TestINKMLGT_2012",
    "TestINKMLGT_2013",
    "Train_2014"
]

for path in paths:
    inkmls_path = os.path.join(path_prefix, path)
    inkml_files = os.listdir(inkmls_path)
    pass
