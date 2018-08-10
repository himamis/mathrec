from trainer import utils
from trainer import AttentionDecoderLSTMCell, SequenceGenerator, ModelCheckpointer
from trainer import preprocess
from trainer import model
import sys
import keras
import numpy as np
import datetime

# set seeds so that every trainingsprocess is starting with same weights.
# it is also needed when creating the model and setting weights from a file, 
# because there must be some kind of randomness in it, because with not setting
# seeds it generates different output every time. But I don't know where :(
from numpy.random import seed
seed(1337)
from tensorflow import set_random_seed
set_random_seed(1337)

# gcs-url with the format: gs://<bucket-name>/<path to resource>
# in my case: gs://image2latex-mlengine/data-simple/
#if '--data-path' in sys.argv:
#    i_arg = sys.argv.index('--data-path') + 1
#    if i_arg >= len(sys.argv):
#        raise Exception('No --data-path argument!')
#    data_base_dir = sys.argv[i_arg]
#else:
#    raise Exception('No --data-path argument!')
start_epoch = 0
if '--start-epoch' in sys.argv:
    i_arg = sys.argv.index('--start-epoch') + 1
    if i_arg < len(sys.argv):
        start_epoch = int(sys.argv[i_arg])
if data_base_dir[-1] != '/':
    data_base_dir += '/'

train_file = data_base_dir+'train.pkl'
validate_file = data_base_dir+'validate.pkl'
test_file = data_base_dir+'test.pkl'
vocabulary_file = data_base_dir+'vocabulary.txt'
images_dir = data_base_dir+'images/'
model_architecture_file = data_base_dir+'model/architecture.json'
model_weights_file = data_base_dir+'model/weights_{epoch}.h5'

embedding_size = 80 # not needed in current version
encoder_size = 256
batch_size = 32

# check if data files exists
if not (utils.file_exists(train_file) and utils.file_exists(validate_file) and utils.file_exists(test_file) and utils.file_exists(vocabulary_file)):
    raise Exception('Image2Latex: One ore more files of [train.pkl, validate.pkl, test.pkl, vocabulary.txt] are missing')

vocabulary = {index:token for index, token in enumerate(utils.read_lines(vocabulary_file))} # {0:<st>, 1:<et>, ...}

# generate data generators
training_data = SequenceGenerator(train_file, images_dir, batch_size)
validation_data = SequenceGenerator(validate_file, images_dir, batch_size)
testing_data = SequenceGenerator(test_file, images_dir, batch_size)

if utils.file_exists(model_architecture_file):
    print("Image2Latex:", "Start load model:", datetime.datetime.now().time())
    json = utils.read_content(model_architecture_file)
    model = keras.models.model_from_json(json, custom_objects={'AttentionDecoderLSTMCell': AttentionDecoderLSTMCell})
    print("Image2Latex:", "End load model:", datetime.datetime.now().time())
else:
    print("Image2Latex:", "Start create model:", datetime.datetime.now().time())
    model = model.create(len(vocabulary), embedding_size, encoder_size)
    # I don't do this, because I think there are some bugs, when saving RNN with constants
    # utils.write_string(model_architecture_file, model.to_json())
    print("Image2Latex:", "End create model:", datetime.datetime.now().time())
    # utils.write_npy(model_weights_file.format(epoch=0), model.get_weights())

if start_epoch!=0 and utils.file_exists(model_weights_file.format(epoch=start_epoch)):
    print("Image2Latex:", "Start loading weights of epoch", start_epoch)
    weights = utils.read_npy(model_weights_file.format(epoch=start_epoch))
    print("Image2Latex:", "Weights loaded")
    model.set_weights(weights)
    print("Image2Latex:", "Weights set to model")
    
checkpointer = ModelCheckpointer(filepath=model_weights_file, verbose=1)
print("Image2Latex:", "Start training...")
history = model.fit_generator(training_data, len(training_data), epochs=10, verbose=1, 
                              validation_data=validation_data, validation_steps=len(validation_data), 
                              callbacks=[checkpointer], initial_epoch=start_epoch)
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print("Image2Latex:", "Start evaluating...")
losses = model.evaluate_generator(testing_data, len(testing_data))
print(model.metrics_names)
print(losses)