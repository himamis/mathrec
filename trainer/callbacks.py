from keras import callbacks
from trainer.predictor import create_predictor
import numpy as np


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

