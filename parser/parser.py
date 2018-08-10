def _token_sequence(tokens, index, sequence):
    seq_index = 0
    ret_val = True
    while len(tokens) > index and len(sequence) > seq_index and ret_val:
        ret_val = tokens[index] == sequence[seq_index]
        index += 1
        seq_index += 1

    return ret_val and len(sequence) == seq_index


def _find_closing_bracket(tokens, index):
    opening = 0
    while index < len(tokens) and (opening != 0 or tokens[index] != "}"):
        if tokens[index] == "{":
            opening += 1
        elif tokens[index] == "}":
            opening -= 1
        index += 1
    return index


function_tokens = ["sin", "cos", "tan"]
command_tokens = ["\\frac"]


class Parser:

    def __init__(self, graphics_factory):
        self.graphics_factory = graphics_factory

    def parse(self, tokens):
        graphics = self.graphics_factory()
        self._expression(tokens, graphics)

        return graphics.draw()

    def _expression(self, tokens, graphics, index=0):
        while index < len(tokens):
            for cmd in function_tokens:
                if _token_sequence(tokens, index, cmd):
                    graphics.expression(cmd)
                    index += len(cmd)
                    break

            if tokens[index] == "^":
                assert tokens[index + 1] == "{"
                closing = _find_closing_bracket(tokens, index + 2)
                power_graphics = self.graphics_factory()
                self._expression(tokens[index + 2: closing], power_graphics)
                graphics.power(power_graphics)
                index = closing + 1
            elif tokens[index] == "\\frac":
                assert tokens[index + 1] == "{"
                closing = _find_closing_bracket(tokens, index + 2)
                assert tokens[closing + 1] == "{"
                closing_denom = _find_closing_bracket(tokens, closing + 2)
                graphics_num = self.graphics_factory()
                self._expression(tokens[index + 2: closing], graphics_num)
                graphics_denom = self.graphics_factory()
                self._expression(tokens[closing + 2: closing_denom], graphics_denom)
                graphics.fraction(graphics_num, graphics_denom)
                index = closing_denom + 1
            else:
                graphics.expression(tokens[index])
                index += 1
