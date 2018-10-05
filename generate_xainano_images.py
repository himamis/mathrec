from trainer.defaults import *
from utilities import parse_arg
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
filename_format = 'formula_{:06d}.png'
data = ""
images_path = path.join(output_dir, dir_name, "images")
if not path.exists(images_path):
    makedirs(images_path)


def image_file_saver(q):
    '''listens for messages on the q, writes image to file. '''
    f = open(path.join(output_dir, dir_name, data_file), "w")
    while 1:
        (file_path, text, image) = q.get()
        if file_path == "\n":
            break
        try:
            png.from_array(image, 'RGB').save(file_path)
            f.write(text + "\n")
            f.flush()
        except Exception:
            print("There was an exception. Arrgh")
    f.close()



def worker_thread(index):
    '''Genrates formulas and images and posts them to a queue '''

    tokens = []
    worker_thread.generator.generate_formula(tokens, worker_thread.config)
    image = worker_thread.token_parser.parse(tokens)
    filename = filename_format.format(index)
    file_path = path.join(images_path, filename)
    worker_thread.queue.put((file_path, filename + "\t" + ''.join(tokens), image))


def worker_init(queue):
    seed()
    worker_thread.queue = queue
    worker_thread.generator = create_generator()
    worker_thread.config = create_config()
    worker_thread.token_parser = create_token_parser(data_base_dir)


def main():
    manager = mp.Manager()
    queue = manager.Queue()
    file_image_pool = mp.Pool(1)
    file_image_pool.apply_async(image_file_saver, (queue,))

    worker_pool = mp.Pool(mp.cpu_count() * 2, worker_init, [queue])
    worker_pool.map(worker_thread, range(number_of_images))

    worker_pool.close()
    file_image_pool.close()
    worker_pool.join()
    queue.join()

    queue.put(("\n", "\n"))
    file_image_pool.join()


if __name__ == '__main__':
    main()