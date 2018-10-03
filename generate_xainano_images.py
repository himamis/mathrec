from trainer.defaults import *
from utilities import parse_arg, progress_bar
from numpy.random import seed
from os import path, makedirs
import png


seed_nr = 123
seed(seed_nr)


dir_name = 'xainano_images'
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/')
output_dir = parse_arg('--output-dir', '/Users/balazs/university/handwritten_images')
number_of_images = int(parse_arg('--count', 200000))

generator = create_generator()
config = create_config()
vocabulary = create_vocabulary(generator, config)
vocabulary_maps = create_vocabulary_maps(vocabulary)
token_parser = create_token_parser(data_base_dir)

data_file = "data.txt"
filename_format = 'formula_{:06d}.jpg'
data = ""
images_path = path.join(output_dir, dir_name, "images")
if not path.exists(images_path):
    makedirs(images_path)

for index in range(number_of_images):
    tokens = []
    generator.generate_formula(tokens, config)
    image = token_parser.parse(tokens)
    filename = filename_format.format(index)
    file_path = path.join(images_path, filename)
    png.from_array(image, 'RGB').save(file_path)
    data += filename + "\t" + ''.join(tokens) + "\n"
    progress_bar("Writing images", index + 1, number_of_images)

file = open(path.join(output_dir, dir_name, data_file), "w")
file.write(data)
file.close()

