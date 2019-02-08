from file_utils import read_pkl
from trainer.params import data_base_dir
from os import path

# START = "<ST>"
PAD = "<PAD>"
PAD_ID = 0
END = "<END>"
EOS_ID = 1
# START_ID = 111  # last token

encoding_vocabulary_file = path.join(data_base_dir, 'vocabulary.pkl')
encoding_vocabulary = read_pkl(encoding_vocabulary_file)
# increment values with 1
for k, v in encoding_vocabulary.items():
    encoding_vocabulary[k] = v + 1
encoding_vocabulary[END] = EOS_ID
encoding_vocabulary[PAD] = PAD_ID
# encoding_vocabulary[START] = len(encoding_vocabulary)

decoding_vocabulary = {v: k for k, v in encoding_vocabulary.items()}


def decode_formula(formula, join=True):
    res = [decoding_vocabulary[f] for f in formula]
    if join:
        res = "".join(res)
    return res
