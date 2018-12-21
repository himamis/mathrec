import numpy as np

def create_predictor(sess, input_params, output_params, encoding_vb, decoding_vb, max_length = 100, k=100, alpha=0.7):
    single_image, single_image_mask, eval_init_h, \
    feature_grid_input, masking_input, single_char_input = input_params
    eval_feature_grid, eval_masking, eval_calculate_h0, eval_output_softmax, \
    eval_state_h = output_params

    def predict_beam_search(image, mask):
        sequence = np.zeros((1,), dtype=np.int32)
        sequence[0] = encoding_vb["<start>"]

        feature_grid, mask, init_h = sess.run([eval_feature_grid, eval_masking,
                                                       eval_calculate_h0], feed_dict={
            single_image: image,
            single_image_mask: mask
        })

        h = init_h
        state = [h]

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
                h, c, output = sess.run([eval_state_h, eval_output_softmax], feed_dict={
                    feature_grid_input: feature_grid,
                    masking_input: mask,
                    eval_init_h: state[0],
                    single_char_input: inp
                })
                top_n_indices = np.argpartition(output[0, 0, :], -k)[-k:]
                for j in range(k):
                    index = top_n_indices[j]
                    sequence = np.zeros((1,), dtype=np.int32)
                    sequence[0] = index
                    sc = np.log(output[0, 0, index])
                    candidates.append([seq + [sequence], [h, c], score + sc])
            def ke(a):
                return a[2] / np.power(len(a[0]), alpha)
            ordered = sorted(candidates, key=ke)
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