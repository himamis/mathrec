from generator import Config, c, random_generator
from token_parser import Parser
from xainano_graphics import create_graphics_factory


def create_config():
    return Config(".", c(["a", "b", "c", "d", "e", "f", "g", "x", "y", "z"]), c(["\\times"]))


def create_generator():
    return random_generator()


def create_vocabulary(generator=create_generator(), config=create_config()):
    return sorted(generator.vocabulary(config) | {"<start>", "<end>"})


def create_vocabulary_maps(vocabulary=create_vocabulary()):
    encoder = {val: idx for idx, val in enumerate(vocabulary)}
    decoder = {idx: val for idx, val in enumerate(vocabulary)}
    return encoder, decoder


def create_token_parser(data_base_dir):
    return Parser(create_graphics_factory(data_base_dir))
