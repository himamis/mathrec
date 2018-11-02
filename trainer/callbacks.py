from keras import callbacks
from trainer.predictor import create_predictor


class EvaluateModel(callbacks.Callback):

    def __init__(self, encoder, decoder, vocabulary, encoder_vb, decoder_vb, generator):
        super().__init__()
        self.predictor = create_predictor(encoder, decoder, vocabulary, encoder_vb, decoder_vb)
        self.decoder_vb =decoder_vb
        self.generator = generator

    def on_epoch_end(self, epoch, logs={}):
        log = ""
        inputs = next(self.generator)
        for (input_img,  input_sequence), _ in inputs:
            prediction = self.predictor(input_img)
            expected = [self.decoder_vb[character] for character in input_sequence]
            expected = expected[:-1]
            log += "Tesing:\nExpected:\t\t\t" + expected + "\nGot\t\t\t" + prediction + "\n"
        if logs.get('evals') is not None:
            logs['evals'] += logs
        else:
            logs['evals'] = logs
