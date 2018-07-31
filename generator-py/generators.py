import numpy as np


class TokenGenerator:

    def __init__(self, token="x"):
        self.token = token

    def generate_formula(self, tokens):
        tokens += [self.token]


class NumberGenerator:

    def __init__(self, n=5, p=0.05, p_real=0.1, separator=".", p_neg=0.0):
        self.n = n
        self.p = p
        self.p_real = p_real
        self.separator = separator
        self.p_neg = p_neg

    def generate_formula(self, tokens):
        if np.random.random() < self.p_neg:
            tokens += "-"
        self.generate_number(tokens)
        if np.random.random() < self.p_real:
            tokens += self.separator
            self.generate_number(tokens)

    def generate_number(self, tokens):
        length = np.random.binomial(self.n, self.p) + 1
        if length > 0:
            tokens += [str(np.random.random_integers(1, 9))]
            tokens += [str(np.random.random_integers(0, 9)) for _ in range(length - 1)]


class ExpressionGenerator:

    def __init__(self, generators=[], operators=[]):
        self.generators = generators
        self.operators = operators

    def generate_formula(self, tokens):
        for index in range(len(self.generators)):
            if index < len(self.operators) and self.operators[index] is not None:
                tokens += [self.operators[index]]
            self.generators[index].generate_formula(tokens)


class CommandGenerator:

    def __init__(self, name, generators):
        self.name = name
        self.generators = generators

    def generate_formula(self, tokens):
        tokens += [self.name]
        for generator in self.generators:
            tokens += "{"
            generator.generate_formula(tokens)
            tokens += "}"


class CallableGenerator:

    def __init__(self, name, generators, brackets):
        self.name = name
        self.generators = generators
        self.brackets = brackets

    def generate_formula(self, tokens):
        if self.name is not None:
            tokens += [self.name]
        tokens += [self.brackets[0]]
        for index in range(len(self.generators)):
            self.generators[index].generate_formula(tokens)
            if index < len(self.generators) - 1:
                tokens += [","]
        tokens += [self.brackets[1]]


class RelationGenerator:

    def __init__(self, operator="=", generators=[TokenGenerator(), TokenGenerator()]):
        self.operator = operator
        self.generators = generators

    def generate_formula(self, tokens):
        self.generators[0].generate_formula(tokens)
        tokens += [self.operator]
        self.generators[1].generate_formula(tokens)


class VariableGenerator:

    def __init__(self, variable_name="x", multiplier="\\times", variable_wrapper=None, scale_generator=None):
        self.variable_name = variable_name
        self.multiplier = multiplier
        self.variable_wrapper = variable_wrapper
        self.scale_generator = scale_generator

    def generate_formula(self, tokens):
        if self.scale_generator is not None:
            self.scale_generator.generate_formula(tokens)
            if self.multiplier is not None:
                tokens += [self.multiplier]
        if self.variable_wrapper is not None:
            self.variable_wrapper.generator = TokenGenerator(self.variable_name)
            self.variable_wrapper.generate_formula(tokens)
        else:
            tokens += [self.variable_name]


class PowerGenerator:

    def __init__(self, generator=TokenGenerator(), power_generator=TokenGenerator("2")):
        self.generator = generator
        self.power_generator = power_generator

    def generate_formula(self, tokens):
        new_tokens = []
        self.generator.generate_formula(new_tokens)
        if len(new_tokens) > 1:
            tokens += "("
        tokens += new_tokens
        if len(new_tokens) > 1:
            tokens += ")"
        tokens += ["^", "{"]
        self.power_generator.generate_formula(tokens)
        tokens += ["}"]


# These build on the ones before
class PolynomialGenerator:

    def __init__(self, length=3, p_miss=0.1, p_minus=0.3):
        self.length = length
        self.p_miss = p_miss
        self.p_minus = p_minus

        self.power = TokenGenerator("1")
        self.power_gen = PowerGenerator(power_generator=self.power)
        self.power_gen.power_generator = self.power

        self.number_gen = NumberGenerator()
        self.var_gen = VariableGenerator()
        self.var_gen.scale_generator = self.number_gen
        self.var_gen.variable_wrapper = self.power_gen

    def generate_formula(self, tokens):
        new_tokens = []
        for ind in range(self.length, 0, -1):
            if not (ind == 2 and self.length == 2) and np.random.random() < self.p_miss:
                continue
            if np.random.random() < self.p_minus:
                new_tokens += ["-"]
            elif len(new_tokens) > 0:
                new_tokens += ["+"]
            if ind > 2:
                self.power.token = str(ind - 1)
                self.var_gen.generate_formula(new_tokens)
            elif ind == 2:
                self.var_gen.variable_wrapper = None
                self.var_gen.generate_formula(new_tokens)
            else:
                self.number_gen.generate_formula(new_tokens)
        tokens += new_tokens


square_brackets = ("[", "]")
round_brackets = ("(", ")")
curly_brackets = ("{", "}")


def function_generator(name, generators): return CallableGenerator(name, generators, round_brackets)


def fraction_generator(numerator, denominator): return CommandGenerator("\\frac", [numerator, denominator])
