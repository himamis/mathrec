from file_utils import read_pkl
from trainer.params import data_base_dir
from os import path

PAD = "<PAD>"
PAD_ID = 0
END = "<END>"
EOS_ID = 1

encoding_vocabulary_file = path.join(data_base_dir, 'vocabulary.pkl')
encoding_vocabulary = read_pkl(encoding_vocabulary_file)
encoding_vocabulary = sorted(list(encoding_vocabulary))
encoding_vocabulary = {token: index + 2 for index, token in enumerate(encoding_vocabulary)}
encoding_vocabulary[END] = EOS_ID
encoding_vocabulary[PAD] = PAD_ID

decoding_vocabulary = {v: k for k, v in encoding_vocabulary.items()}


def decode_formula(formula, join=True, joiner=""):
    res = [decoding_vocabulary[f] for f in formula]
    if join:
        res = joiner.join(res)
    return res
