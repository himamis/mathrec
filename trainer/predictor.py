import numpy as np


def create_predictor(encoder, decoder, vocabulary, encoding_vb, decoding_cv, max_length = 100):

    def predict(image):
        input_image = np.expand_dims(image, 0)
        feature_grid = encoder.predict(input_image)

        sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
        sequence[0, 0, encoding_vb["<start>"]] = 1.

        h = np.zeros((1, 256 * 2), dtype="float32")
        c = np.zeros((1, 256 * 2), dtype="float32")
        states = [h, c]

        decoded_sentence = ""
        while True:
            output, h, c = decoder.predict([feature_grid, sequence] + states)

            # Sample token
            sampled_token_index = np.argmax(output[0, -1, :])
            sampled_char = decoding_cv[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: hit max length, or find stop character
            if sampled_char == "<end>" or len(decoded_sentence) > max_length:
                break

            # Update sequence
            sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
            sequence[0, 0, sampled_token_index] = 1.

            states = [h, c]

        return decoded_sentence

    return predict