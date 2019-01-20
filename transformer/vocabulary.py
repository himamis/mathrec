from utilities import parse_arg
from file_utils import read_pkl
from trainer.params import data_base_dir
from os import path

# START = "<ST>"
END = "<END>"
EOS_ID = 0
# START_ID = 111  # last token

encoding_vocabulary_file = path.join(data_base_dir, 'vocabulary.pkl')
encoding_vocabulary = read_pkl(encoding_vocabulary_file)
encoding_vocabulary[END] = EOS_ID
# encoding_vocabulary[START] = len(encoding_vocabulary)
