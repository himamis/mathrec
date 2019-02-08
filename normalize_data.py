import os
import pickle


# TODO Add logging
def normalize_formula(formula):
    new_formula = list(formula)
    offset = 0
    changed = False
    for index, token in enumerate(formula):
        if (token == "^" or token == "_") and formula[index + 1] != "{":
            changed = True
            new_formula.insert(index + offset + 1, "{")
            new_formula.insert(index + offset + 3, "}")
            offset += 2

    if changed:
        print("Fixed from \n{}\n \tto \n{}\n\n".format("".join(formula), "".join(new_formula)))

    return new_formula


tokens_dir = "/Users/balazs/token_trace"
token_file = "validating_data.pkl"

tokens_path = os.path.join(tokens_dir, token_file)
tokens = pickle.load(open(tokens_path, 'rb'))

for index, (formula, input) in enumerate(tokens):
    formula = normalize_formula(formula)
    tokens[index] = (formula, input)

pickle.dump(tokens, open(tokens_path, 'wb'))
