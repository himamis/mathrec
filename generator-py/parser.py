import graphics as g
import images as i


def _token_sequence(tokens, index, sequence):
    seq_index = 0
    ret_val = True
    while len(tokens) > index and len(sequence) > seq_index and ret_val:
        ret_val = tokens[index] == sequence[seq_index]
        index += 1
        seq_index += 1

    return ret_val and len(sequence) == seq_index


def _find_closing_bracket(tokens, index):
    while index < len(tokens) and tokens[index] != "}":
        index += 1
    return index


function_tokens = ["sin", "cos", "tan"]
command_tokens = ["\\frac"]


class Parser:

    def __init__(self):
        self.image = i.Images()

    def parse(self, tokens):
        graphics = g.Graphics()
        self._expression(tokens, graphics)

        return graphics.draw()

    def _expression(self, tokens, graphics, index=0):
        while index < len(tokens):
            for cmd in function_tokens:
                if _token_sequence(tokens, index, cmd):
                    image = self.image.image(cmd)
                    graphics.expression(image)
                    index += len(cmd)
                    break

            if tokens[index] == "^":
                assert tokens[index + 1] == "{"
                closing = _find_closing_bracket(tokens, index + 2)
                power_graphics = g.Graphics()
                self._expression(tokens[index + 2: closing], power_graphics)
                graphics.power(power_graphics.draw())
                index += 1
            elif _token_sequence(tokens, index, "\\frac"):
                assert tokens[index + 6] == "{"
                closing = _find_closing_bracket(tokens, index + 7)
                assert tokens[closing + 1] == "{"
                closing_denom = _find_closing_bracket(tokens, closing + 2)
                graphics_num = g.Graphics()
                self._expression(tokens[index + 7: closing], graphics_num)
                graphics_denom = g.Graphics()
                self._expression(tokens[closing + 2: closing_denom], graphics_denom)
                graphics.fraction(graphics_num.draw(), graphics_denom.draw(), self.image.image("-"))
                index += closing_denom + 1
            else:
                image = self.image.image(tokens[index])
                graphics.expression(image)
                index += 1



