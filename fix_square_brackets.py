import pickle
import os
from normalize_data2 import skip_brackets
import transformer.vocabulary


def update_vocabulary():
    vocab_fname = "vocabulary.pkl"
    dir = "/Users/balazs/token_trace_normalized"
    vocab = pickle.load(open(os.path.join(dir, vocab_fname), 'rb'))
    a = len(vocab)
    vocab['\\['] = a + 1
    vocab['\\]'] = a + 2

    dir_out = "/Users/balazs/token_trace_normalized_square"
    pickle.dump(vocab, open(os.path.join(dir_out, vocab_fname), 'wb'))

def main():
    update_vocabulary()
    return
    token_file = "testing_data.pkl"

    input_tokens_dir = "/Users/balazs/token_trace_normalized"
    input_tokens_path = os.path.join(input_tokens_dir, token_file)
    tokens = pickle.load(open(input_tokens_path, 'rb'))

    for index, (formula, input) in enumerate(tokens):
        for j, (char, box) in enumerate(input):
            if char == '[':
                char = '\\['
            elif char == ']':
                char = '\\]'
            input[j] = (char, box)

        for j, char in enumerate(formula):
            if char == '[' and j > 0:
                if formula[j - 1] != '\\sqrt':
                    k = skip_brackets(formula, j, ('[', ']')) - 1
                    formula[j] = '\\['
                    formula[k] = '\\]'
        tokens[index] = (formula, input)

    output_tokens_dir = "/Users/balazs/token_trace_normalized_square"
    output_tokens_path = os.path.join(output_tokens_dir, token_file)
    # pickle.dump(tokens, open(output_tokens_path, 'wb'))


if __name__ == "__main__":
    main()
