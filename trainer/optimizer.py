import keras.optimizers
# import tensorflow as tf


class PrintAdadelta(keras.optimizers.Adadelta):

    def get_gradients(self, loss, params):
        grads = super().get_gradients(loss, params)
        #index = -43
        #grads[index] = tf.Print(grads[index], [tf.shape(grads[index]), grads[index]], summarize=999999)
        return grads