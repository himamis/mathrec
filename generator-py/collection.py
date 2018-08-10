import numpy as np


class Collection:

    def __init__(self, values):
        self.values = values

    def get(self):
        return np.random.choice(self.values)

    def all(self):
        return set(self.values)


def collection(values):
    if isinstance(values, Collection):
        return values
    elif isinstance(values, list):
        return Collection(values)
    else:
        return Collection([values])
