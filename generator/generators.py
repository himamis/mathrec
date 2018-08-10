import numpy as np
import string
from collection import collection

'''
Generators should be classes that have at least two public methods with the following signatures
generate_formula([String])
vocabulary(config)
The first one should generate a random and valid mathematical expression in latex
broken down into the smallest tokens possible (e.g. sin should be 's', 'i', 'n', instead of 'sin').
The second method should return all tokens that the generator can generate in a set.
'''


class Config:

    def __init__(self, separator, variables, multiplier):
        self.separator = separator
        self.variables = variables
        self.multiplier = multiplier

    def vocabulary(self):
        return self.variables.all() | {self.separator, self.multiplier}


class TokenGenerator:

    def __init__(self, token="x"):
        self.token = token

    def generate_formula(self, tokens, config):
        tokens += [self.token]

    def vocabulary(self, config):
        return {self.token}


class NumberGenerator:

    def __init__(self, n=5, p=0.05, p_real=0.1, p_neg=0.0):
        self.n = n
        self.p = p
        self.p_real = p_real
        self.p_neg = p_neg

    def generate_formula(self, tokens, config):
        if np.random.random() < self.p_neg:
            tokens += "-"
        self.generate_number(tokens)
        if np.random.random() < self.p_real:
            tokens += config.separator
            self.generate_number(tokens)

    def generate_number(self, tokens):
        length = np.random.binomial(self.n, self.p) + 1
        if length > 0:
            tokens += [str(np.random.random_integers(1, 9))]
            tokens += [str(np.random.random_integers(0, 9)) for _ in range(length - 1)]

    def vocabulary(self, config):
        vocab = {str(num) for num in range(0, 10)}
        if self.p_neg > 0:
            vocab = vocab | {'-'}
        if self.p_real > 0:
            vocab = vocab | {config.separator}
        return vocab


class ExpressionGenerator:

    def __init__(self, generators=[], operators=collection([])):
        self.generators = generators
        self.operators = operators
        self.randomize_order = False

    def generate_formula(self, tokens, config):
        if self.randomize_order:
            np.random.shuffle(self.generators)
        for index in range(len(self.generators)):
            if index < len(self.operators) and self.operators[index] is not None:
                operator = self.operators[index].get()
                if operator is not None:
                    tokens += [operator]
            self.generators[index].generate_formula(tokens, config)

    def vocabulary(self, config):
        vocab = set()
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        for operator in self.operators:
            if operator is not None:
                vocab = vocab | operator.all()
        if None in vocab:
            vocab.remove(None)
        return vocab


class CommandGenerator:

    def __init__(self, name, generators):
        self.name = name
        self.generators = generators

    def generate_formula(self, tokens, config):
        tokens += [self.name]
        for generator in self.generators:
            tokens += "{"
            generator.generate_formula(tokens, config)
            tokens += "}"

    def vocabulary(self, config):
        vocab = {self.name}
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        return vocab


class CallableGenerator:

    def __init__(self, name, generators, brackets):
        self._name = collection(name)
        self.generators = generators
        self.brackets = brackets

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = collection(value)

    def generate_formula(self, tokens, config):
        if self.name is not None:
            name = self.name.get()
            if name is not None:
                tokens.append(name)
        tokens += [self.brackets[0]]
        for index in range(len(self.generators)):
            self.generators[index].generate_formula(tokens, config)
            if index < len(self.generators) - 1:
                tokens += [","]
        tokens += [self.brackets[1]]

    def vocabulary(self, config):
        vocab = {self.brackets[0], self.brackets[1]}
        if self.name is not None:
            for name in self.name.all():
                for character in name:
                    vocab.add(character)
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        if len(self.generators) > 1:
            vocab = vocab | {","}
        if None in vocab:
            vocab.remove(None)
        return vocab


class RelationGenerator:

    def __init__(self, operator=collection("="), generators=[TokenGenerator(), TokenGenerator()]):
        self.operator = operator
        self.generators = generators

    def generate_formula(self, tokens, config):
        self.generators[0].generate_formula(tokens, config)
        tokens += [self.operator.get()]
        self.generators[1].generate_formula(tokens, config)

    def vocabulary(self, config):
        return self.generators[0].vocabulary(config) | self.generators[1].vocabulary(config) | self.operator.all()


class VariableGenerator:

    def __init__(self, variable_wrapper=None, scale_generator=None):
        self.variable_wrapper = variable_wrapper
        self.scale_generator = scale_generator

    def generate_formula(self, tokens, config):
        if self.scale_generator is not None:
            self.scale_generator.generate_formula(tokens, config)
            if config.multiplier is not None:
                tokens += [config.multiplier]
        if self.variable_wrapper is not None:
            self.variable_wrapper.generator = TokenGenerator(config.variables.get())
            self.variable_wrapper.generate_formula(tokens, config)
        else:
            tokens += [config.variables.get()]

    def vocabulary(self, config):
        vocab = set()
        if self.scale_generator is not None:
            vocab = vocab | self.scale_generator.vocabulary(config)
            if config.multiplier is not None:
                vocab.add(config.multiplier)
        if self.variable_wrapper is not None:
            vocab = vocab | self.variable_wrapper.vocabulary(config)
            vocab = vocab | config.variables.all()
        return vocab


class PowerGenerator:

    def __init__(self, generator=TokenGenerator(), power_generator=TokenGenerator("2")):
        self.generator = generator
        self.power_generator = power_generator

    def generate_formula(self, tokens, config):
        new_tokens = []
        self.generator.generate_formula(new_tokens, config)
        if len(new_tokens) > 1:
            tokens += "("
        tokens += new_tokens
        if len(new_tokens) > 1:
            tokens += ")"
        tokens += ["^", "{"]
        self.power_generator.generate_formula(tokens, config)
        tokens += ["}"]

    def vocabulary(self, config):
        vocab = {"(", ")", "^", "{", "}"}
        vocab = vocab | self.generator.vocabulary(config)
        vocab = vocab | self.power_generator.vocabulary(config)
        return vocab


# These build on the ones before
class PolynomialGenerator:

    def __init__(self, length=collection(3), p_miss=0.1, p_minus=0.3):
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

    def generate_formula(self, tokens, config):
        new_tokens = []
        length = self.length.get()
        for ind in range(length, 0, -1):
            if not (ind == 2 and length == 2) and np.random.random() < self.p_miss:
                continue
            if np.random.random() < self.p_minus:
                new_tokens += ["-"]
            elif len(new_tokens) > 0:
                new_tokens += ["+"]
            if ind > 2:
                self.power.token = str(ind - 1)
                self.var_gen.generate_formula(new_tokens, config)
            elif ind == 2:
                self.var_gen.variable_wrapper = None
                self.var_gen.generate_formula(new_tokens, config)
            else:
                self.number_gen.generate_formula(new_tokens, config)
        tokens += new_tokens

    def vocabulary(self, config):
        vocab = {"+"} | self.var_gen.vocabulary(config)
        if self.p_minus > 0:
            vocab = vocab | {"-"}
        return vocab


class RandomGenerator:

    def __init__(self, generators):
        self.generators = generators

    def generate_formula(self, tokens, config):
        generator = np.random.choice(self.generators)
        generator.generate_formula(tokens, config)

    def vocabulary(self, config):
        vocab = set()
        for generator in self.generators:
            vocab = vocab | generator.vocabulary(config)
        return vocab


square_brackets = ("[", "]")
round_brackets = ("(", ")")
curly_brackets = ("{", "}")


def function_generator(name, generators): return CallableGenerator(name, generators, round_brackets)


def fraction_generator(numerator, denominator): return CommandGenerator("\\frac", [numerator, denominator])


def random_simple_expression():
    num = NumberGenerator()

    var1 = VariableGenerator()
    var1.scale_generator = num
    var2 = VariableGenerator()
    rand = RandomGenerator([var1, var2])

    operators = [collection(None), collection(["+", "-"])]
    expr = ExpressionGenerator([rand, num])
    expr.randomize_order = True
    expr.operators = operators

    return expr


def random_polynomial():
    poly = PolynomialGenerator()
    poly.length = collection([2, 3, 4])

    return poly


def random_simple_equation():
    num = NumberGenerator()
    var = VariableGenerator()
    var.scale_generator = num

    expr = ExpressionGenerator()
    expr.generators = [var, num]
    expr.randomize_order = True
    expr.operators = [collection([None, "+", "-"]), collection(["+", "-"])]

    rel = ExpressionGenerator()
    rel.generators = [expr, num]
    rel.operators = [collection(None), collection(["=", "\\leq", "\\geq"])]

    return rel


def random_coord():
    num = NumberGenerator()
    num.p_neg = 0.5
    co = function_generator(None, [num, num])

    generators = []
    for character in string.ascii_uppercase:
        generators.append(TokenGenerator(character))
    rand = RandomGenerator(generators)

    rel = RelationGenerator()
    rel.operator = "="
    rel.generators = [rand, co]

    return rel


def random_fraction():
    pow_num = NumberGenerator(0, p_real=0)
    pow_num.p_neg = 0.4

    num = NumberGenerator()
    num.p_neg = 0.5

    var = VariableGenerator()
    var.scale_generator = num
    var.variable_wrapper = PowerGenerator(power_generator=pow_num)

    fraction = fraction_generator(var, pow_num)

    e = TokenGenerator("e")
    pow_e = PowerGenerator(e, pow_num)

    tok = TokenGenerator("y")
    rel = ExpressionGenerator()
    rel.operators = [collection(None), collection("=")]

    #TODO Factor out pow_e
    rel.generators = [tok, RandomGenerator([fraction])]

    return rel


def random_long_expression_no_frac():
    commands = ["sin", "cos", "tan"]
    simple = random_simple_expression()
    command_gen = function_generator(collection(commands), [simple])
    random = RandomGenerator([command_gen, simple])

    return random


def _random_long_expr_item():
    expression = random_long_expression_no_frac()
    frac = fraction_generator(expression, expression)

    return RandomGenerator([expression, frac])


def random_long_expression():
    generators = []

    for i in range(4, 8):
        expr = ExpressionGenerator()
        expr.generators = [_random_long_expr_item() for _ in range(i)]
        expr.operators = [collection(None)] + [collection(["+", "-"]) for _ in range(i - 1)]
        generators.append(expr)
    #length = np.random.choice([4, 5, 6, 7])
    ##expr.generators = [long_expr_item() for _ in range(length)]
    #e#xpr.operators = [None]+[np.random.choice(["+", "-"]) for _ in range(length-1)]

    return RandomGenerator(generators)