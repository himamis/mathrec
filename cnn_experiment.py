from trainer import model, predictor
from trainer.defaults import *
import file_utils as utils
from utilities import parse_arg
from trainer.logger import NBatchLogger
from graphics import augment
from os import path
from trainer.sequence import create_default_sequence_generator
from xainano_graphics import postprocessor
from numpy.random import seed
from tensorflow import set_random_seed
from generator import single_token_generator
from keras.layers import Dense, Reshape, Flatten, GlobalMaxPooling2D
from keras import Model
from datetime import datetime


seed(1337)
set_random_seed(1337)

date_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
folder_str = 'cnn_experiment-' + date_str


data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/split/xainano_images')
weights_file = parse_arg('--weights', "/Users/balazs/university/weights_20.h5")
background_dir = parse_arg('--background-dir', '/Users/balazs/university/split_backgrounds_dir')
experiment_dir = parse_arg('--experiment_dir', '/Users/balazs/university/cnn_experiment')

output_dir = path.join(experiment_dir, folder_str)
history_file = path.join(output_dir, "history.pkl")


generator = create_generator()
actual_generator = single_token_generator()
config = create_config()
vocabulary = create_vocabulary(generator, config)
encoding_vb, decoding_vb = create_vocabulary_maps(vocabulary)
train_augmentor = augment.Augmentor(path.join(background_dir, 'training/backgrounds'), path.join(background_dir, 'training/grids'))
post_processor = postprocessor.Postprocessor()

batch_size = 32

train_token_parser = create_token_parser(path.join(data_base_dir, 'training'))
validation_token_parser = create_token_parser(path.join(data_base_dir, 'validation'))

# generate data generators
train_augmentor = augment.Augmentor(path.join(background_dir, 'training/backgrounds'), path.join(background_dir, 'training/grids'))
validation_augmentor = augment.Augmentor(path.join(background_dir, 'validation/backgrounds'), path.join(background_dir, 'validation/grids'))
post_processor = postprocessor.Postprocessor()
training_data = create_default_sequence_generator(train_token_parser, train_augmentor, post_processor, actual_generator, config, batch_size, [encoding_vb, decoding_vb], single=True)
validation_data = create_default_sequence_generator(validation_token_parser, validation_augmentor, post_processor, actual_generator, config, batch_size, [encoding_vb, decoding_vb], single=True)

print('Start creating model')
default_model, encoder, decoder = model.create_default(len(vocabulary))
print('Model created')
if utils.file_exists(weights_file):
    print('Start loading weights')
    weights = utils.read_npy(weights_file)
    default_model.set_weights(weights)
    print('Weights loaded and set')
else:
    print("Weights file does not exist")
    exit()

for layer in encoder.layers:
    layer.trainable = False

encoder_cnn_layer = encoder.layers[31]
encoder_cnn_end = encoder_cnn_layer.output
max_pooling = GlobalMaxPooling2D()(encoder_cnn_end)
dense_layer_out = Dense(len(vocabulary), activation="softmax", name="output_softmax")(max_pooling)
new_model = Model(encoder.input, dense_layer_out)
new_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

logger = NBatchLogger(1)
print("Image2Latex:", "Start training...")
history = new_model.fit_generator(training_data, 100, epochs=20, verbose=2,
                              validation_data=validation_data, validation_steps=100,
                              callbacks=[logger], initial_epoch=0)
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print(new_model.metrics_names)
del history.model
utils.write_pkl(history_file, history)

print("done")