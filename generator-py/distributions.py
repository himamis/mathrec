import numpy as np


class Distribution:

    def __init__(self):
        self.objects = []
        self.p = []

    def set_uniform(self, objects):
        self.set(objects, np.repeat(1.0 / len(objects), len(objects)))

    def set(self, objects, p):
        self.objects = objects
        self.p = p

    def get(self):
        return np.random.choice(self.objects, p=self.p)


def uniform(objects):
    distribution = Distribution()
    distribution.set_uniform(objects)
    return distribution


def single(object):
    distribution = Distribution()
    distribution.set_uniform([object])
    return distribution
