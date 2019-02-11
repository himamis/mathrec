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


def skip_brackets(formula, index, brackets=("{", "}")):
    """
    Returns new index with brackets skipped.
    Assumes that the formula does have closing brackets after index.

    :param formula: list of tokens
    :param index: index
    :param brackets: tuple of brackets
    :return: new index with skipped brackets
    """
    assert formula[index] == brackets[0], "Formula {} index {} does not start with bracket {}".format("".join(formula),
                                                                                                      index,
                                                                                                      brackets[0])
    index += 1
    opening = 0
    while formula[index] != brackets[1] or opening != 0:
        if formula[index] == brackets[0]:
            opening += 1
        elif formula[index] == brackets[1]:
            opening -= 1
        index += 1
    return index + 1


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
                if prev_token not in {"\\frac", "_", "^", "\\sqrt", "}", "]"}:
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
    ]:
        new_formula = norm(formula)
        if new_formula != formula:
            print("Changed from \n{}\nto\n{}\n".format("".join(formula), "".join(new_formula)))
        formula = new_formula

    return formula


def main():
    token_file = "validating_data.pkl"

    input_tokens_dir = "/Users/balazs/token_trace_normalized"
    input_tokens_path = os.path.join(input_tokens_dir, token_file)
    tokens = pickle.load(open(input_tokens_path, 'rb'))

    for index, (formula, input) in enumerate(tokens):
        if "".join(formula).startswith("\\int\\sin2"):
            print(input)
        # formula = normalize(formula)
        # tokens[index] = (formula, input)

    output_tokens_dir = "/Users/balazs/token_trace_normalized"
    output_tokens_path = os.path.join(output_tokens_dir, token_file)
    # pickle.dump(tokens, open(output_tokens_path, 'wb'))


if __name__ == "__main__":
    main()
