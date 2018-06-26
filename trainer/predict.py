from trainer import model
from trainer import utils
import numpy as np

# set seeds so that every trainingsprocess is starting with same weights.
# it is also needed when creating the model and setting weights from a file, 
# because there must be some kind of randomness in it, because with not setting
# seeds it generates different output every time. But I don't know where :(
from numpy.random import seed
seed(1337)
from tensorflow import set_random_seed
set_random_seed(1337)


embedding_size = 80
encoder_size = 256

print('Enter base dir:')
data_base_dir = input()
vocabulary_file = data_base_dir + 'vocabulary.txt'
imgs_dir = data_base_dir + 'images/'
weights_file = data_base_dir + 'model/weights_{epoch}.h5'
vocabulary = {index:token for index, token in enumerate(utils.read_lines(vocabulary_file))}
print('Vocabulary read. Size is', len(vocabulary))
print('Start creating model')
model = model.create(len(vocabulary), embedding_size, encoder_size, True)
print('Model created')
for epoch in reversed(range(10)):
    file = weights_file.format(epoch=epoch + 1)
    if utils.file_exists(file):
        print('Start loading weights', epoch+1)
        weights = utils.read_npy(file)
        model.set_weights(weights)
        print('Weights loaded and set')
        break

while True:
    print('Enter image file name:')
    img_file = imgs_dir + input()
    img = utils.read_img(img_file)[:, :, 0][:, :, None]
    imgs = np.array([img])
    x = [imgs, np.zeros((1, 30, 1))]
    y = model.predict(x, batch_size=1)
    for seq in y:
        outp = ''
        for tok in seq:
            outp += vocabulary[np.argmax(tok)] + ' '
        print(outp)
    print('')