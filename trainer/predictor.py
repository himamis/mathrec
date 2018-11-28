import numpy as np

def create_predictor(encoder, decoder, vocabulary, encoding_vb, decoding_vb, max_length = 300):

    def predict(image):
        input_image = np.expand_dims(image, 0)
        feature_grid = encoder.predict(input_image)

        encoded_output = []
        sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
        sequence[0, 0, encoding_vb["<start>"]] = 1.

        h = np.zeros((1, 512), dtype="float32")
        c = np.zeros((1, 512), dtype="float32")
        states = [h, c]

        decoded_output = ""
        while True:
            output, h, c = decoder.predict(feature_grid + [sequence] + states)

            # Sample token
            sampled_token_index = np.argmax(output[0, -1, :])
            sampled_char = decoding_vb[sampled_token_index]


            # Exit condition: hit max length, or find stop character
            if sampled_char == "<end>" or len(decoded_output) >= max_length:
                break

            encoded_output.append(output[0, -1, :])
            decoded_output += sampled_char

            # Update sequence
            sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
            sequence[0, 0, sampled_token_index] = 1.

            states = [h, c]

        return (decoded_output, encoded_output)

    return predict