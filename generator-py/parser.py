class Parser:

    def parse(self, tokens):
        self._expression(tokens, 0)

    def _expression(self, tokens, index):
        tokens[index]