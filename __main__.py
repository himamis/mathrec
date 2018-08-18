from file_utils import utils
from trainer import AttentionDecoderLSTMCell, SequenceGenerator, ModelCheckpointer
from trainer import model
import sys
import datetime
from numpy.random import seed

from generator import *
from token_parser import Parser
from xainano_graphics import create_graphics_factory

from tensorflow import set_random_seed
from trainer.sequence import xainano_sequence_generator
from trainer.logger import NBatchLogger


#config = Config(np.random.choice([",", "."]), c(["x"]), np.random.choice([None, "\\times"]))
#tokens = []
#gen = random_long_expression()
#gen.generate_formula(tokens, config)

#image = token_parser.parse(tokens)
#s = reduce((lambda a, b: a + " " + b), tokens)
#print(s)
#print(gen.vocabulary(config))

#cv2.imshow('image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# set seeds so that every trainingsprocess is starting with same weights.
# it is also needed when creating the model and setting weights from a file,
# because there must be some kind of randomness in it, because with not setting
# seeds it generates different output every time. But I don't know where :(

seed(1337)

set_random_seed(1337)

start_epoch = 0

if '--start-epoch' in sys.argv:
    i_arg = sys.argv.index('--start-epoch') + 1
    if i_arg < len(sys.argv):
        start_epoch = int(sys.argv[i_arg])

data_base_dir = "/Users/balazs/university/"
#train_file = data_base_dir + 'train.pkl'
#validate_file = data_base_dir + 'validate.pkl'
#test_file = data_base_dir + 'test.pkl'
#vocabulary_file = data_base_dir + 'vocabulary.txt'
#images_dir = data_base_dir + 'images/'
model_architecture_file = data_base_dir + 'model/architecture.json'
model_weights_file = data_base_dir + 'model/weights_{epoch}.h5'

embedding_size = 80  # not needed in current version
encoder_size = 256
batch_size = 32

# check if data files exists
#if not (utils.file_exists(train_file) and utils.file_exists(validate_file) and utils.file_exists(
#        test_file) and utils.file_exists(vocabulary_file)):
#    raise Exception(
#        'Image2Latex: One ore more files of [train.pkl, validate.pkl, test.pkl, vocabulary.txt] are missing')

base = "/Users/balazs/university/extracted_images"

config = Config(".", c(["x"]), None)
generator = random_generator()
# End token
vocabulary = generator.vocabulary(config) | {"<start>", "<end>"}
vocabulary_map = {val: idx for idx, val in enumerate(vocabulary)}
token_parser = Parser(create_graphics_factory(base))



# generate data generators
training_data = xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map)
validation_data = xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map)
testing_data = xainano_sequence_generator(generator, config, token_parser, batch_size, vocabulary_map)

#if utils.file_exists(model_architecture_file):
#    print("Image2Latex:", "Start load model:", datetime.datetime.now().time())
#    json = utils.read_content(model_architecture_file)
#    model = keras.models.model_from_json(json, custom_objects={ 'AttentionDecoderLSTMCell': AttentionDecoderLSTMCell })
#    print("Image2Latex:", "End load model:", datetime.datetime.now().time())
#else:
print("Image2Latex:", "Start create model:", datetime.datetime.now().time())
model, encoder, decoder = model.create(len(vocabulary), embedding_size, encoder_size)
# I don't do this, because I think there are some bugs, when saving RNN with constants
# utils.write_string(model_architecture_file, model.to_json())
print("Image2Latex:", "End create model:", datetime.datetime.now().time())
# utils.write_npy(model_weights_file.format(epoch=0), model.get_weights())

if start_epoch != 0 and utils.file_exists(model_weights_file.format(epoch=start_epoch)):
    print("Image2Latex:", "Start loading weights of epoch", start_epoch)
    weights = utils.read_npy(model_weights_file.format(epoch=start_epoch))
    print("Image2Latex:", "Weights loaded")
    model.set_weights(weights)
    print("Image2Latex:", "Weights set to model")

checkpointer = ModelCheckpointer(filepath=model_weights_file, verbose=1)
logger = NBatchLogger(1)
print("Image2Latex:", "Start training...")
history = model.fit_generator(training_data, 1000, epochs=10, verbose=2,
                              validation_data=validation_data, validation_steps=1000,
                              callbacks=[checkpointer, logger], initial_epoch=start_epoch)
print("Image2Latex:", history.epoch)
print("Image2Latex:", history.history)
print("Image2Latex:", "Start evaluating...")
losses = model.evaluate_generator(testing_data, 1000)
print(model.metrics_names)
print(losses)
