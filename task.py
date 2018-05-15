import utils
import preprocess
import model
import os.path
import keras
import numpy as np
from AttentionDecoderLSTM import AttentionDecoderLSTMCell
from SequenceGenerator import SequenceGenerator
from keras.callbacks import ModelCheckpoint
import datetime

data_base_dir = '../../data/'
formulas_file = data_base_dir+'formulas.norm.lst'
norm_formulas_file = data_base_dir+'norm_formulas.lst'
train_file = data_base_dir+'train.lst'
validate_file = data_base_dir+'validate.lst'
test_file = data_base_dir+'test.lst'
vocabulary_file = data_base_dir+'vocabulary.txt'
images_dir = data_base_dir+'images/'
sizes_file = data_base_dir+'sizes.txt'
model_architecture_file = data_base_dir+'model/architecture.json'
model_weights_file = data_base_dir+'model/weights.h5'

embedding_size = 80
encoder_size = 256
batch_size = 32
imgH = 240
imgW = 640

if not os.path.isfile(norm_formulas_file):
    preprocess.create_norm_formulas_file(formulas_file, norm_formulas_file)

if not os.path.isfile(sizes_file):
    vocabulary_size = preprocess.create_vocabulary(norm_formulas_file, vocabulary_file)
    total_train, total_validate, total_test = preprocess.create_pkl_files(vocabulary_file, norm_formulas_file, train_file, validate_file, test_file, data_base_dir, images_dir)
    utils._write_list(sizes_file, [vocabulary_size, total_train, total_validate, total_test])
else:
    sizes = utils._read_lines(sizes_file)
    vocabulary_size = int(sizes[0])
    total_train = int(sizes[1])
    total_validate = int(sizes[2])
    total_test = int(sizes[3])

training_data = SequenceGenerator(data_base_dir, images_dir, 'train', batch_size)
validation_data = SequenceGenerator(data_base_dir, images_dir, 'validate', batch_size)

if os.path.isfile(model_architecture_file):
    print("Start load model:", datetime.datetime.now().time())
    json = utils._read_content(model_architecture_file)
    model = keras.models.model_from_json(json, custom_objects={'AttentionDecoderLSTMCell': AttentionDecoderLSTMCell})
    print("End load model:", datetime.datetime.now().time())
else:
    print("Start create model:", datetime.datetime.now().time())
    model = model.create(vocabulary_size, embedding_size, encoder_size, imgH, imgW)
    # utils._write_string(model_architecture_file, model.to_json())
    print("End create model:", datetime.datetime.now().time())

if os.path.isfile(model_weights_file): #predict
    tokens = utils._read_lines(vocabulary_file)
    vocabulary = {}
    vocabulary_reverse = {}
    for i, val in enumerate(tokens):
        vocabulary[val] = i
        vocabulary_reverse[i] = val
    # do something
else: #train
    checkpointer = ModelCheckpoint(filepath=model_weights_file, verbose=1, save_weights_only=True, save_best_only=True)
    history = model.fit_generator(training_data, epochs=2, verbose=1, validation_data=validation_data, callbacks=[checkpointer], shuffle=True)
    print(history.epoch)
    print(history.history)