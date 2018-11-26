from keras import callbacks
from keras.callbacks import Callback
from trainer.predictor import create_predictor
import numpy as np
import logging


class EvaluateModel(callbacks.Callback):

    def __init__(self, encoder, decoder, vocabulary, encoder_vb, decoder_vb, generator):
        super().__init__()
        self.predictor = create_predictor(encoder, decoder, vocabulary, encoder_vb, decoder_vb)
        self.decoder_vb = decoder_vb
        self.generator = generator

    def on_epoch_end(self, epoch, logs={}):
        log = ""
        images, sequences = next(self.generator)[0]
        for input_img,  input_sequence in zip(images, sequences):
            prediction = self.predictor(input_img)
            expected = [self.decoder_vb[np.argmax(character)] for character in input_sequence]
            expected = "".join(expected)
            log += "Testing:\nExpected:\t\t\t" + expected + "\nGot\t\t\t" + prediction + "\n"
        logs['evals'] = log
        print(log)


class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            logging.debug('step: {}/{} ... {}'.format(self.step,
                                              self.params['steps'],
                                              metrics_log))
            self.metric_cache.clear()