import os
import pickle
import string


safe_to_wrap = set(string.ascii_letters) | set(string.digits) | {
    "\\infty",
    "\\prime",
    "\\theta",
    "\\alpha",
    "\\gamma",
    "\\mu",
    "\\beta",
    "\\phi",
    "\\pi",
    "\\lambda",
    "\\Delta"
}


def skip_brackets(formula, index, brackets=("{", "}"), direction=1):
    """
    Returns new index with brackets skipped.
    Assumes that the formula does have closing brackets after index.

    :param formula: list of tokens
    :param index: index
    :param brackets: tuple of brackets
    :param direction: the order of traversing (+1, or -1)
    :return: new index with skipped brackets
    """
    assert formula[index] == brackets[0], "Formula {} index {} does not start with bracket {}".format("".join(formula),
                                                                                                      index,
                                                                                                      brackets[0])
    assert direction == 1 or direction == -1, "Direction must be between -1 or 1"

    index += direction
    opening = 0
    while formula[index] != brackets[1] or opening != 0:
        if formula[index] == brackets[0]:
            opening += 1
        elif formula[index] == brackets[1]:
            opening -= 1
        index += direction
    return index + direction


def wrap(formula, index):
    """
    Wraps the token in formula with { } brackets.

    :param formula: list of tokens
    :param index: index
    :return: formula with wrapped token
    """
    assert formula[index] in safe_to_wrap, "Not safe to wrap in {} at index {}".format("".join(formula), index)
    formula.insert(index, "{")
    formula.insert(index + 2, "}")

    return formula


# Has dependency on fraction normalization
def normalize_under_superscript(formula):
    new_formula = list(formula)
    index = 0
    while index < len(new_formula):
        token = new_formula[index]
        if (token == "^" or token == "_") and new_formula[index + 1] != "{":
            if new_formula[index + 1] == "\\frac":
                new_formula.insert(index + 1, "{")
                index += 3

                # Since we've normalized
                # We jump over the frac and two parameters
                for i in range(2):
                    index = skip_brackets(new_formula, index)
                new_formula.insert(index, "}")
            else:
                wrap(new_formula, index + 1)
                index += 3
        index += 1

    return new_formula


def remove_brackets(formula, index):
    closing = skip_brackets(formula, index) - 1
    del formula[closing]
    del formula[index]

    return formula


def normalize_fractions(formula):
    new_formula = list(formula)
    index = 0
    while index < len(new_formula):
        if new_formula[index] == "\\frac":
            if index > 0 and new_formula[index - 1] == "{" and index < len(new_formula) and new_formula[index + 1] == "}":
                # Unwrap frac
                del new_formula[index + 1]
                del new_formula[index - 1]
                index -= 1

            index += 1
            for i in range(2):
                if new_formula[index] != "{":
                    wrap(new_formula, index)
                    index += 3
                else:
                    index = skip_brackets(new_formula, index)
        else:
            index += 1

    return new_formula


def remove_unnecessary_brackets(formula):
    new_formula = list(formula)
    index = 0
    while index < len(new_formula):
        token = new_formula[index]
        if token == "{":
            if index == 0:
                new_formula = remove_brackets(new_formula, index)
            else:
                prev_token = new_formula[index - 1]
                if prev_token == "}":
                    prev_index = skip_brackets(new_formula, index - 1, ("}", "{"), -1)
                    if new_formula[prev_index] != "\\frac":
                        new_formula = remove_brackets(new_formula, index)
                elif prev_token not in {"\\frac", "_", "^", "\\sqrt", "]"}:
                    new_formula = remove_brackets(new_formula, index)
                    index -= 1
        index += 1

    return new_formula


def normalize_sqrt(formula):
    new_formula = list(formula)
    index = 0
    while index < len(new_formula):
        if new_formula[index] == "\\sqrt":
            index += 1
            if new_formula[index] == "[":
                index = skip_brackets(new_formula, index, ("[", "]"))
            if new_formula[index] != "{":
                wrap(new_formula, index)
                index += 3
        index += 1
    return new_formula


def normalize_under_superscript_order(formula):
    new_formula = list(formula)
    index = 0
    while index < len(new_formula):
        if new_formula[index] == "_":
            next_ind = skip_brackets(new_formula, index + 1)
            if next_ind < len(new_formula) and new_formula[next_ind] == "^":
                last_inde = skip_brackets(new_formula, next_ind + 1)
                underscript = new_formula[index+1:next_ind]
                overscript = new_formula[next_ind+1:last_inde]

                cur = index
                new_formula[cur] = "^"
                cur += 1
                new_formula[cur:cur+len(overscript)] = overscript
                cur += len(overscript)
                new_formula[cur] = "_"
                cur += 1
                new_formula[cur:cur+len(underscript)] = underscript

        index += 1
    return new_formula


def normalize_square_brackets(formula, input):
    for j, (char, box) in enumerate(input):
        if char == '[':
            char = '\\['
        elif char == ']':
            char = '\\]'
        input[j] = (char, box)

    for j, char in enumerate(formula):
        if char == '[':
            if j == 0 or formula[j - 1] != '\\sqrt':
                k = skip_brackets(formula, j, ('[', ']')) - 1
                formula[j] = '\\['
                formula[k] = '\\]'
    return formula, input


def normalize(formula):
    """
    Applies normalizations to formula
    :param formula: list of tokens
    :return: new formula
    """
    for norm in [
        normalize_fractions,
        normalize_under_superscript,
        normalize_sqrt,
        remove_unnecessary_brackets,
        normalize_under_superscript_order
    ]:
        new_formula = norm(formula)
        if new_formula != formula:
            print("Changed from \n{}\nto\n{}\n".format("".join(formula), "".join(new_formula)))
        formula = new_formula

    return formula


def main():
    token_file = "validating_data.pkl"

    input_tokens_dir = "/Users/balazs/token_trace_orig"
    input_tokens_path = os.path.join(input_tokens_dir, token_file)
    tokens = pickle.load(open(input_tokens_path, 'rb'))

    for index, (formula, input) in enumerate(tokens):
        try:
            formula = normalize(formula)
        except AssertionError as e:
            print("Could not parse")
            print(formula)
            raise e

        tokens[index] = normalize_square_brackets(formula, input)

    output_tokens_dir = "/Users/balazs/token_trace_normalized2"
    output_tokens_path = os.path.join(output_tokens_dir, token_file)
    pickle.dump(tokens, open(output_tokens_path, 'wb'))


if __name__ == "__main__":
    main()
