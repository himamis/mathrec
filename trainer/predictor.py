import numpy as np

def create_predictor(encoder, decoder, vocabulary, encoding_vb, decoding_vb, max_length = 300,
                     k=100, alpha=0.1):

    def predict(image):
        input_image = np.expand_dims(image, 0)
        feature_grid = encoder.predict(input_image)

        decoded_list = []
        sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
        sequence[0, 0, encoding_vb["<start>"]] = 1.

        h = np.zeros((1, 256), dtype="float32")
        c = np.zeros((1, 256), dtype="float32")
        states = [h, c]

        decoded_output = ""
        while True:
            output, h, c = decoder.predict([feature_grid] + [sequence] + states)

            # Sample token
            sampled_token_index = np.argmax(output[0, -1, :])
            sampled_char = decoding_vb[sampled_token_index]


            # Exit condition: hit max length, or find stop character
            if sampled_char == "<end>" or len(decoded_output) >= max_length:
                break

            decoded_list.append(sampled_char)
            decoded_output += sampled_char

            # Update sequence
            sequence = np.zeros((1, 1, len(vocabulary)), dtype="float32")
            sequence[0, 0, sampled_token_index] = 1.

            states = [h, c]

        return (decoded_output, decoded_list)

    def predict_beam_search(image):
        input_image = np.expand_dims(image, 0)
        feature_grid = encoder.predict(input_image)

        sequence = np.zeros((len(vocabulary),), dtype="float32")
        sequence[encoding_vb["<start>"]] = 1.
        h = np.zeros((1, 256), dtype="float32")
        c = np.zeros((1, 256), dtype="float32")
        state = [h, c]

        sequences = [[[sequence], state, 0.0]]
        finished = [False]

        should_continue = True
        while should_continue:
            candidates = []
            for i, seqs in enumerate(sequences):
                seq, state, score = seqs
                if finished[i]:
                    candidates.append([seq, state, score])
                    continue
                last = seq[-1]
                inp = np.reshape(last, (1, 1, -1))
                output, h, c = decoder.predict([feature_grid] + [inp] + state)
                top_n_indices = np.argpartition(output[0, 0, :], -k)[-k:]
                for j in range(k):
                    index = top_n_indices[j]
                    sequence = np.zeros((len(vocabulary),), dtype="float32")
                    sequence[index] = 1.0
                    sc = np.log(output[0, 0, index])
                    candidates.append([seq + [sequence], [h, c], score + sc])
            def ke(a):
                return a[2] / np.power(len(a[0]), alpha)
            ordered = sorted(candidates, key=ke)
            sequences = ordered[-k:]
            finished = []
            for i in range(k):
                seq, state, score = sequences[i]
                finish = len(seq) > max_length or np.argmax(seq[-1]) == encoding_vb['<end>']
                finished.append(finish)
            should_continue = not np.all(finished)

        sequence = sequences[-1]
        ret = [decoding_vb[np.argmax(s)] for s in sequence[0][1:-1]]
        return ret

    return predict_beam_search