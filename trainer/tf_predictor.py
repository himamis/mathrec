import numpy as np


def create_predictor(sess, feature_grid_input_params, feature_grid_decoder_init_output_params,
                     decoder_input_params, decoder_output_params,
                     encoding_vb, decoding_vb, max_length=100, k=100, alpha=0.7):

    (pl_single_input_image, pl_single_image_mask) = feature_grid_input_params

    (eval_feature_grid, eval_masking, eval_calculate_h0, eval_calculate_alphas) = \
        feature_grid_decoder_init_output_params

    (pl_feature_grid, pl_feature_grid_mask, pl_single_input_character,
     pl_eval_init_h, pl_eval_init_alpha) = decoder_input_params

    (eval_output_softmax, eval_state_h, eval_alpha) = decoder_output_params

    def predict_beam_search(image, mask):
        sequence = np.zeros((1,), dtype=np.int32)
        sequence[0] = encoding_vb["<start>"]

        dictionary = {pl_single_input_image: image, pl_single_image_mask: mask}
        feature_grid, mask, h, a = sess.run([eval_feature_grid, eval_masking,
                                             eval_calculate_h0, eval_calculate_alphas], feed_dict=dictionary)

        state = [h, a]

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
                inp = np.reshape(last, (1, -1))
                dictionary = {
                    pl_feature_grid: feature_grid,
                    pl_feature_grid_mask: mask,
                    pl_eval_init_h: state[0],
                    pl_eval_init_alpha: state[1],
                    pl_single_input_character: inp
                }
                h, a, output = sess.run([eval_state_h, eval_alpha, eval_output_softmax], feed_dict=dictionary)
                top_n_indices = np.argpartition(output[0, 0, :], -k)[-k:]
                for j in range(k):
                    index = top_n_indices[j]
                    sequence = np.zeros((1,), dtype=np.int32)
                    sequence[0] = index
                    sc = np.log(output[0, 0, index])
                    candidates.append([seq + [sequence], [h, a], score + sc])

            def key(entry):
                return entry[2] / np.power(len(entry[0]), alpha)
            ordered = sorted(candidates, key=key)
            sequences = ordered[-k:]
            finished = []
            for i in range(k):
                seq, state, score = sequences[i]
                finish = len(seq) > max_length or seq[-1] == encoding_vb['<end>']
                finished.append(finish)
            should_continue = not np.all(finished)

        sequence = sequences[-1]
        ret = [decoding_vb[s[0]] for s in sequence[0][1:-1]]
        return ret

    return predict_beam_search
