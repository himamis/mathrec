from tensorflow.python import VarianceScaling
from tensorflow.python.framework import dtypes
import tensorflow as tf

class GlorotNormal(VarianceScaling):

    def __init__(self,
                 seed=None,
                 dtype=dtypes.float32):
        super(GlorotNormal, self).__init__(
            scale=1.0,
            mode="fan_avg",
            distribution="normal",
            seed=seed,
            dtype=dtype)

    def get_config(self):
        return {
            "seed": self.seed,
            "dtype": self.dtype.name
        }


def he_normal(seed=None, dtype=tf.float32):
    return VarianceScaling(scale=2., mode="fan_in", distribution="normal", seed=seed, dtype=dtype)


glorot_normal = GlorotNormal