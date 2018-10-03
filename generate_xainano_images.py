from trainer.defaults import *
from utilities import parse_arg, progress_bar
from numpy.random import seed
from os import path, makedirs
import png
import multiprocessing as mp


dir_name = 'xainano_images'
data_base_dir = parse_arg('--data-base-dir', '/Users/balazs/university/')
output_dir = parse_arg('--output-dir', '/Users/balazs/university/handwritten_images')
number_of_images = int(parse_arg('--count', 200000))
ncores = int(parse_arg('--ncores', 4))

data_file = "data.txt"
filename_format = 'formula_{:06d}.jpg'
data = ""
images_path = path.join(output_dir, dir_name, "images")
if not path.exists(images_path):
    makedirs(images_path)


def file_writer(q):
    '''listens for messages on the q, writes to file. '''
    f = open(path.join(output_dir, dir_name, data_file), "w")
    while 1:

        text = q.get()
        if text == "\n":
            break
        #print("Writing " + text)
        f.write(text + "\n")
        f.flush()
    f.close()

def image_saver(q):
    '''listens for messages on the q, writes image to file. '''
    while 1:
        (file_path, image) = q.get()
        if file_path == "\n":
            break
        #print("saving " + file_path)
        png.from_array(image, 'RGB').save(file_path)


def worker_thread(index):
    '''Genrates formulas and images and posts them to a queue '''

    tokens = []
    worker_thread.generator.generate_formula(tokens, worker_thread.config)
    image = worker_thread.token_parser.parse(tokens)
    filename = filename_format.format(index)
    file_path = path.join(images_path, filename)
    worker_thread.text_q.put(filename + "\t" + ''.join(tokens))
    worker_thread.image_q.put((file_path, image))


def worker_init(text_q, image_q):
    seed()
    worker_thread.text_q = text_q
    worker_thread.image_q = image_q
    worker_thread.generator = create_generator()
    worker_thread.config = create_config()
    worker_thread.token_parser = create_token_parser(data_base_dir)


def main():
    manager = mp.Manager()
    text_q = manager.Queue()
    image_q = manager.Queue()
    file_image_pool = mp.Pool(5)

    file_image_pool.apply_async(file_writer, (text_q,))
    file_image_pool.apply_async(image_saver, (image_q,))
    file_image_pool.apply_async(image_saver, (image_q,))
    file_image_pool.apply_async(image_saver, (image_q,))
    file_image_pool.apply_async(image_saver, (image_q,))

    worker_pool = mp.Pool(mp.cpu_count() * 2, worker_init, [text_q, image_q])
    worker_pool.map(worker_thread, range(number_of_images))

    worker_pool.close()
    file_image_pool.close()

    image_q.put(("\n", "\n"))
    image_q.put(("\n", "\n"))
    image_q.put(("\n", "\n"))
    image_q.put(("\n", "\n"))
    image_q.put(("\n", "\n"))
    text_q.put("\n")

    worker_pool.join()
    file_image_pool.join()



if __name__ == '__main__':
    #seed_nr = 123
    #seed(seed_nr)
    main()