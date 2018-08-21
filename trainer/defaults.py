from generator import Config, c, random_generator
from token_parser import Parser
from xainano_graphics import create_graphics_factory

def create_config():
    return Config(".", c(["x"]), None)


def create_generator():
    return random_generator()


def create_vocabulary(generator=create_generator(), config=create_config()):
    return generator.vocabulary(config) | {"<start>", "<end>"}


def create_vocabulary_map(vocabulary=create_vocabulary()):
    return {val: idx for idx, val in enumerate(vocabulary)}


def create_token_parser(data_base_dir):
    return Parser(create_graphics_factory(data_base_dir))
